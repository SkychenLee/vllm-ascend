import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

NUM_LORAS = 64


def bgmv_expand_cpu_impl(
    x: torch.Tensor, w: torch.Tensor, indices: torch.Tensor, y: torch.Tensor, slice_offset: int, slice_size: int
) -> torch.Tensor:
    result = y.clone()
    valid = indices >= 0
    output_slice = slice(slice_offset, slice_offset + slice_size)
    weights = w[indices[valid]].to(torch.float64)
    delta = torch.bmm(weights, x[valid].to(torch.float64).unsqueeze(-1)).squeeze(-1)
    result[valid, output_slice] = (delta + y[valid, output_slice].to(torch.float64)).to(y.dtype)
    return result


def make_inputs(batch_size, rank, output_dim, full_dim, dtype, index_mode="mixed", seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((batch_size, rank), dtype=torch.float32, generator=generator)
    w = torch.randn((NUM_LORAS, output_dim, rank), dtype=dtype, generator=generator)
    # Include repeated slots, boundary slots, and multiple negative adapter ids.
    pattern = torch.tensor([0, NUM_LORAS - 1, -1, 32, 31, -7, 42, 17], dtype=torch.int64)
    indices = pattern.repeat((batch_size + pattern.numel() - 1) // pattern.numel())[:batch_size].clone()
    if index_mode == "inactive":
        indices.fill_(-1)
    elif index_mode != "mixed":
        raise ValueError(f"Unknown index mode: {index_mode}")
    y = torch.randn((batch_size, full_dim), dtype=dtype, generator=generator)
    return x, w, indices, y


def assert_result(actual, expected, indices, initial_y, slice_offset, slice_size):
    # Use the output dtype's compute tolerance, with stricter absolute tolerance
    # and a 100% element match. Keep the standard absolute-error cap as well.
    if actual.dtype == torch.float16:
        atol, rtol, max_abs_error = 2**-12, 2**-9, 0.1
    else:
        atol, rtol, max_abs_error = 2**-8, 2**-6, 1.0
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert (actual.double() - expected.double()).abs().max().item() <= max_abs_error

    protected = torch.ones_like(actual, dtype=torch.bool)
    protected[:, slice_offset : slice_offset + slice_size] = False
    protected[indices < 0] = True
    torch.testing.assert_close(
        actual.view(torch.int16)[protected], initial_y.view(torch.int16)[protected], atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size,rank,output_dim,slice_offset,full_dim,index_mode",
    [
        # Preserve the original single-row, rank-16, 64-adapter slice case.
        pytest.param(1, 16, 128, 0, 384, "mixed", id="original-single-row"),
        pytest.param(1, 8, 96, 0, 96, "mixed", id="small-rank-full-width"),
        pytest.param(8, 16, 96, 96, 192, "mixed", id="decode-second-slice"),
        pytest.param(17, 32, 256, 16, 288, "mixed", id="odd-batch-middle-slice"),
        pytest.param(17, 64, 256, 32, 320, "mixed", id="large-rank-middle-slice"),
        pytest.param(8, 8, 4080, 16, 4112, "mixed", id="below-output-tile"),
        pytest.param(17, 16, 4096, 0, 4096, "mixed", id="full-output-tile"),
        pytest.param(8, 32, 4112, 16, 4144, "mixed", id="output-tile-tail"),
        pytest.param(17, 64, 4112, 32, 4160, "mixed", id="large-rank-output-tail"),
        pytest.param(1, 64, 4096, 16, 4128, "mixed", id="single-row-output-tile"),
        pytest.param(8, 8, 256, 16, 288, "inactive", id="all-inactive-small-rank"),
        pytest.param(17, 32, 96, 96, 192, "inactive", id="all-inactive-second-slice"),
    ],
)
@torch.inference_mode()
def test_bgmv_expand(dtype, batch_size, rank, output_dim, slice_offset, full_dim, index_mode):
    x, w, indices, y = make_inputs(batch_size, rank, output_dim, full_dim, dtype, index_mode)
    expected = bgmv_expand_cpu_impl(x, w, indices, y, slice_offset, output_dim)
    x_npu, w_npu, indices_npu, y_npu = (tensor.npu() for tensor in (x, w, indices, y))

    returned = torch.ops._C_ascend.bgmv_expand(x_npu, w_npu, indices_npu, y_npu, slice_offset, output_dim)

    assert returned.data_ptr() == y_npu.data_ptr()
    assert_result(y_npu.cpu(), expected, indices, y, slice_offset, output_dim)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size,rank,output_dim,slice_offset,full_dim",
    [(8, 16, 96, 96, 192), (17, 64, 4112, 16, 4144)],
)
@torch.inference_mode()
def test_bgmv_expand_graph_replay(dtype, batch_size, rank, output_dim, slice_offset, full_dim):
    initial = make_inputs(batch_size, rank, output_dim, full_dim, dtype)
    x_npu, w_npu, indices_npu, y_npu = (tensor.npu() for tensor in initial)

    def op():
        torch.ops._C_ascend.bgmv_expand(x_npu, w_npu, indices_npu, y_npu, slice_offset, output_dim)

    op()
    eager_output = y_npu.cpu()
    for _ in range(3):
        op()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        op()

    for replay in range(4):
        index_mode = "inactive" if replay == 2 else "mixed"
        x, w, indices, y = make_inputs(batch_size, rank, output_dim, full_dim, dtype, index_mode, seed=replay)
        # Exercise routing changes and inactive -> active transitions at the
        # captured addresses. Reset nonzero Y before every accumulating call.
        indices = indices.roll(replay)
        expected = bgmv_expand_cpu_impl(x, w, indices, y, slice_offset, output_dim)
        for destination, source in zip((x_npu, w_npu, indices_npu, y_npu), (x, w, indices, y)):
            destination.copy_(source)
        graph.replay()
        actual = y_npu.cpu()
        assert_result(actual, expected, indices, y, slice_offset, output_dim)
        if replay == 0:
            torch.testing.assert_close(actual.view(torch.int16), eager_output.view(torch.int16), atol=0, rtol=0)
