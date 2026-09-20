import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Shrink accumulates and writes FP32 even when its inputs are BF16 or FP16.
DEFAULT_ATOL = 2**-16
DEFAULT_RTOL = 2**-10
NUM_COMBINED_EXPERTS = 256  # Two adapter slots, each containing 128 experts.


def bgmv_shrink_cpu_impl(
    x: torch.Tensor, w: torch.Tensor, indices: torch.Tensor, y: torch.Tensor, scaling: float
) -> torch.Tensor:
    result = y.clone()
    valid = indices >= 0
    weights = w[indices[valid]].to(torch.float64)
    products = torch.bmm(weights, x[valid].to(torch.float64).unsqueeze(-1)).squeeze(-1)
    # Valid rows are overwritten; rows without an adapter retain their input y.
    result[valid] = (products * scaling).to(y.dtype)
    return result


def make_inputs(batch_size, input_dim, rank, dtype, index_mode="mixed", seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((batch_size, input_dim), dtype=dtype, generator=generator)
    w = torch.randn((NUM_COMBINED_EXPERTS, rank, input_dim), dtype=dtype, generator=generator) * 0.125
    # Include both adapter slots, boundary experts, and multiple negative ids.
    pattern = torch.tensor([255, 0, -1, 128, 127, -7, 42, 170], dtype=torch.int64)
    indices = pattern.repeat((batch_size + pattern.numel() - 1) // pattern.numel())[:batch_size].clone()
    if index_mode == "inactive":
        indices.fill_(-1)
    elif index_mode != "mixed":
        raise ValueError(f"Unknown index mode: {index_mode}")
    y = torch.randn((batch_size, rank), dtype=torch.float32, generator=generator)
    return x, w, indices, y


def assert_result(actual, expected, indices, initial_y):
    torch.testing.assert_close(actual, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    inactive = indices < 0
    torch.testing.assert_close(actual[inactive], initial_y[inactive], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size,input_dim,rank,scaling,index_mode",
    [
        pytest.param(1, 96, 16, 0.5, "mixed", id="single-row"),
        pytest.param(8, 96, 16, 0.5, "mixed", id="decode"),
        pytest.param(17, 96, 16, -1.0, "mixed", id="odd-batch-negative-scale"),
        pytest.param(64, 96, 16, 0.0, "mixed", id="zero-scale"),
        pytest.param(513, 96, 16, 0.5, "mixed", id="core-tail"),
        pytest.param(4096, 96, 16, -1.0, "mixed", id="prefill"),
        pytest.param(17, 96, 16, 0.5, "inactive", id="all-inactive"),
        pytest.param(8, 2048, 2, 0.5, "mixed", id="wide-input"),
        pytest.param(17, 128, 16, -1.0, "mixed", id="multiple-vector-repeats"),
        pytest.param(8, 1024, 15, 0.5, "mixed", id="below-batched-cost-boundary"),
        # The public shrink API requires input_dim > rank. Exercise GM row
        # padding and vector masks independently of the production MoE shape.
        pytest.param(1, 2, 1, 0.5, "mixed", id="minimal-shrink"),
        pytest.param(8, 15, 2, 0.5, "mixed", id="below-copy-block"),
        pytest.param(17, 16, 7, -1.0, "mixed", id="aligned-copy-block"),
        pytest.param(8, 17, 8, 0.5, "mixed", id="above-copy-block"),
        pytest.param(8, 63, 9, 0.5, "mixed", id="below-vector-repeat"),
        pytest.param(8, 64, 15, 0.5, "mixed", id="full-vector-repeat"),
        pytest.param(17, 65, 17, -1.0, "mixed", id="vector-repeat-tail"),
        pytest.param(8, 95, 31, 0.5, "mixed", id="padded-input-odd-rank"),
        pytest.param(17, 97, 32, 0.5, "mixed", id="padded-input-even-rank"),
        pytest.param(8, 127, 64, -1.0, "mixed", id="nearly-two-vector-repeats"),
        pytest.param(8, 129, 127, 0.5, "mixed", id="small-shrink-ratio"),
        pytest.param(17, 256, 128, 0.5, "mixed", id="multiple-rank-tiles"),
        pytest.param(8, 512, 257, -1.0, "mixed", id="rank-above-repeat-count"),
        # The runtime UB budget limits wide-row rank tiles; cover the cost
        # guard together with full and partial rank tiles that pass it.
        pytest.param(8, 2032, 31, 0.5, "mixed", id="wide-row-cost-fallback"),
        pytest.param(8, 1024, 16, 0.5, "mixed", id="full-rank-tile-cost-boundary"),
        pytest.param(17, 1024, 17, -1.0, "mixed", id="wide-row-rank-tile-tail"),
        pytest.param(8, 2033, 16, 0.5, "mixed", id="padded-stride-fallback"),
        pytest.param(17, 2049, 7, 0.5, "mixed", id="unaligned-fallback-rows"),
        pytest.param(8, 11775, 2, 0.5, "mixed", id="below-input-tile"),
        pytest.param(8, 11776, 2, 0.5, "mixed", id="full-input-tile"),
        pytest.param(17, 11777, 2, -1.0, "mixed", id="incremental-input-tail"),
    ],
)
@torch.inference_mode()
def test_bgmv_shrink(dtype, batch_size, input_dim, rank, scaling, index_mode):
    x, w, indices, y = make_inputs(batch_size, input_dim, rank, dtype, index_mode)
    expected = bgmv_shrink_cpu_impl(x, w, indices, y, scaling)
    x_npu, w_npu, indices_npu, y_npu = (tensor.npu() for tensor in (x, w, indices, y))

    torch.ops._C_ascend.bgmv_shrink(x_npu, w_npu, indices_npu, y_npu, scaling)

    assert_result(y_npu.cpu(), expected, indices, y)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "input_dim,rank",
    [(96, 16), (65, 17), (512, 255), (512, 256), (2033, 16)],
)
@torch.inference_mode()
def test_bgmv_shrink_graph_replay(dtype, input_dim, rank):
    batch_size, scaling = 17, 0.5
    initial = make_inputs(batch_size, input_dim, rank, dtype)
    x_npu, w_npu, indices_npu, y_npu = (tensor.npu() for tensor in initial)

    def op():
        torch.ops._C_ascend.bgmv_shrink(x_npu, w_npu, indices_npu, y_npu, scaling)

    for _ in range(3):
        op()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        op()

    for replay in range(1, 4):
        x, w, indices, y = make_inputs(batch_size, input_dim, rank, dtype, seed=replay)
        # Change which rows use an adapter without changing captured addresses.
        indices = indices.roll(replay)
        expected = bgmv_shrink_cpu_impl(x, w, indices, y, scaling)
        for destination, source in zip((x_npu, w_npu, indices_npu, y_npu), (x, w, indices, y)):
            destination.copy_(source)
        graph.replay()
        assert_result(y_npu.cpu(), expected, indices, y)
