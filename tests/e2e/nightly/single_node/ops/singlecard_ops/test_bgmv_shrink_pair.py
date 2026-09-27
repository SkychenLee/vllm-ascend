import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def make_inputs(rows, hidden, rank, dtype, seed=0, inactive=False):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, hidden), dtype=dtype, generator=generator)
    weights = tuple(torch.randn((8, rank, hidden), dtype=dtype, generator=generator) * 0.125 for _ in range(2))
    pattern = torch.tensor([7, 0, -1, 3, -(2**63), 6, 2, 7], dtype=torch.int64)
    indices = pattern.repeat((rows + 7) // 8)[:rows].roll(seed).clone()
    if inactive:
        indices.fill_(-1)
    y = torch.randn((2, rows, rank), dtype=torch.float32, generator=generator)
    # Inactive rows must retain even the sign bit of zero.
    y[:, indices < 0] = -0.0
    return x, *weights, indices, y


def assert_output(actual, inputs, scale):
    x, weight0, weight1, indices, initial = inputs
    assert torch.isfinite(actual).all()
    valid = indices >= 0
    expected = initial.double()
    for projection, weight in enumerate((weight0, weight1)):
        product = torch.bmm(weight[indices[valid]].double(), x[valid].double().unsqueeze(-1)).squeeze(-1)
        expected[projection, valid] = product * scale
    torch.testing.assert_close(actual.double(), expected, atol=2**-16, rtol=2**-10)
    if actual.numel():
        assert (actual.double() - expected.double()).abs().max().item() <= 0.01
    torch.testing.assert_close(
        actual.view(torch.int32)[:, ~valid], initial.view(torch.int32)[:, ~valid], atol=0, rtol=0
    )


def compare_with_singles(device_inputs, initial_y, actual, scale):
    x, weight0, weight1, indices, _ = device_inputs
    singles = initial_y.npu()
    for projection, weight in enumerate((weight0, weight1)):
        torch.ops._C_ascend.bgmv_shrink(x, weight, indices, singles[projection], scale)
    torch.testing.assert_close(actual.view(torch.int32), singles.cpu().view(torch.int32), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,hidden,rank,scale,inactive,output_offset",
    [
        pytest.param(0, 65, 17, 0.5, False, 0, id="empty"),
        pytest.param(8, 2048, 2, 0.5, False, 0, id="decode"),
        pytest.param(64, 2048, 2, -1.0, False, 0, id="decode-larger-batch"),
        pytest.param(19, 129, 16, 0.0, False, 0, id="odd-batch-zero-scale"),
        pytest.param(21, 129, 16, -1.0, False, 0, id="paired-core-tail"),
        pytest.param(8, 512, 257, 0.5, False, 0, id="paired-rank-tile-tail"),
        pytest.param(21, 512, 257, -1.0, False, 0, id="rank-tail-plane-fallback"),
        pytest.param(41, 11777, 2, 0.5, False, 0, id="wide-input-tail"),
        pytest.param(8, 65, 17, 0.5, False, 1, id="output-base-fallback"),
        pytest.param(17, 96, 16, 0.5, True, 0, id="all-inactive"),
    ],
)
@torch.inference_mode()
def test_bgmv_shrink_pair(dtype, rows, hidden, rank, scale, inactive, output_offset):
    inputs = make_inputs(rows, hidden, rank, dtype, inactive=inactive)
    device_inputs = [value.npu() for value in inputs]
    # Keep protected padding on both sides of a contiguous output view. Offset
    # one deliberately exercises the native fallback without changing shape.
    backing = torch.full((inputs[-1].numel() + output_offset + 1,), 123.0, device="npu", dtype=torch.float32)
    output = backing[output_offset : output_offset + inputs[-1].numel()].view_as(inputs[-1])
    output.copy_(inputs[-1])
    device_inputs[-1] = output

    returned = torch.ops._C_ascend.bgmv_shrink_pair(*device_inputs, scale)

    assert returned is None
    actual = output.cpu()
    assert_output(actual, inputs, scale)
    if rows:
        compare_with_singles(device_inputs, inputs[-1], actual, scale)
    padding = backing.cpu()
    assert (padding[:output_offset] == 123.0).all()
    assert (padding[output_offset + inputs[-1].numel() :] == 123.0).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,hidden,rank", [(64, 2048, 2), (17, 65, 17)])
@torch.inference_mode()
def test_bgmv_shrink_pair_graph_replay(dtype, rows, hidden, rank):
    scale = -0.5
    initial = make_inputs(rows, hidden, rank, dtype)
    device_inputs = tuple(value.npu() for value in initial)

    def op():
        torch.ops._C_ascend.bgmv_shrink_pair(*device_inputs, scale)

    for _ in range(3):
        op()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        op()

    for replay in range(3):
        # Change every input at the captured addresses, including both weights,
        # routing and protected output rows; then reactivate after all-inactive.
        inputs = make_inputs(rows, hidden, rank, dtype, seed=replay + 1, inactive=replay == 1)
        for destination, source in zip(device_inputs, inputs):
            destination.copy_(source)
        graph.replay()
        actual = device_inputs[-1].cpu()
        assert_output(actual, inputs, scale)
        compare_with_singles(device_inputs, inputs[-1], actual, scale)
