import pytest
import torch

from vllm_ascend.lora import cube_bgmv
from vllm_ascend.lora.cube_bgmv import cube_bgmv_expand, cube_bgmv_shrink, prepare_cube_bgmv_routing
from vllm_ascend.lora.lora_ops import bgmv_expand_slice, bgmv_shrink
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.fixture(autouse=True)
def enable_cube_candidate(monkeypatch):
    monkeypatch.setattr(cube_bgmv, "ENABLE_CUBE_BGMV", True)


@pytest.mark.parametrize("rows", [2048, 2049, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden,rank,groups", [(80, 8, 1), (80, 8, 5), (96, 16, 4), (512, 32, 8), (2048, 2, 128)])
@torch.inference_mode()
def test_cube_shrink_matches_vector_and_fp64(rows, dtype, hidden, rank, groups):
    gen = torch.Generator().manual_seed(rows + hidden)
    x = torch.randn(rows, hidden, generator=gen, dtype=dtype)
    weights = torch.randn(groups, rank, hidden, generator=gen, dtype=dtype) * 0.125
    indices = (torch.arange(rows, dtype=torch.int64) * 17) % groups
    indices[::11] = -1
    indices[1::17] = -7
    initial = torch.randn(rows, rank, generator=gen, dtype=torch.float32)
    x_npu, w_npu, ids_npu, y_npu = (tensor.npu() for tensor in (x, weights, indices, initial))
    vector = y_npu.clone()

    bgmv_shrink(x_npu, w_npu, y_npu, ids_npu, 0.5)
    torch.ops._C_ascend.bgmv_shrink(x_npu, w_npu, ids_npu, vector, 0.5)
    valid = indices >= 0
    expected = initial.clone()
    expected[valid] = (
        torch.bmm(weights[indices[valid]].double(), x[valid].double().unsqueeze(-1)).squeeze(-1) * 0.5
    ).float()
    result = y_npu.cpu()
    torch.testing.assert_close(result, expected, atol=2**-14, rtol=2**-10)
    torch.testing.assert_close(result, vector.cpu(), atol=2**-14, rtol=2**-10)
    torch.testing.assert_close(result[~valid], initial[~valid], atol=0, rtol=0)


@pytest.mark.parametrize("rows", [2048, 2049, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rank,width,groups", [(2, 96, 5), (7, 65, 5), (8, 160, 1), (8, 160, 5), (16, 96, 4), (32, 128, 8)]
)
@torch.inference_mode()
def test_cube_expand_restores_rows_and_preserves_inactive(rows, dtype, rank, width, groups):
    gen = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, rank, generator=gen, dtype=torch.float32)
    weights = torch.randn(groups, width, rank, generator=gen, dtype=dtype) * 0.125
    indices = (torch.arange(rows, dtype=torch.int64) * 7) % groups
    indices[::11] = -1
    initial = torch.randn(rows, 2 * width, generator=gen, dtype=dtype)
    initial[indices < 0, width:] = -0.0
    x_npu, w_npu, ids_npu, y_npu = (tensor.npu() for tensor in (x, weights, indices, initial))
    vector = y_npu.clone()

    bgmv_expand_slice(x_npu, w_npu, y_npu, ids_npu, width, width)
    if rank in (8, 16, 32, 64):
        torch.ops._C_ascend.bgmv_expand(x_npu, w_npu, ids_npu, vector, width, width)
    valid = indices >= 0
    expected = initial.clone()
    delta = torch.bmm(weights[indices[valid]].double(), x[valid].double().unsqueeze(-1)).squeeze(-1)
    expected[valid, width:] = (initial[valid, width:].double() + delta).to(dtype)
    result = y_npu.cpu()
    atol = 2**-11 if dtype == torch.float16 else 2**-8
    torch.testing.assert_close(result, expected, atol=atol, rtol=2**-6)
    if rank in (8, 16, 32, 64):
        torch.testing.assert_close(result, vector.cpu(), atol=atol, rtol=2**-6)
    torch.testing.assert_close(result[~valid].view(torch.int16), initial[~valid].view(torch.int16), atol=0, rtol=0)


@pytest.mark.parametrize("rows", [2049, 4096])
@torch.inference_mode()
def test_cube_shrink_graph_replay_consumes_changed_indices(rows):
    hidden, rank, groups = 96, 16, 4
    x = torch.randn(rows, hidden, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(groups, rank, hidden, device="npu", dtype=torch.bfloat16)
    ids = (torch.arange(rows, device="npu") % groups).long()
    output = torch.empty(rows, rank, device="npu", dtype=torch.float32)
    for _ in range(2):
        bgmv_shrink(x, weights, output, ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        bgmv_shrink(x, weights, output, ids)
    for shift in (1, 2, 3, 4, 5):
        current = ((torch.arange(rows, device="npu") + shift) % groups).long()
        if shift == 4:
            current.zero_()
        elif shift == 5:
            current.fill_(-1)
        else:
            current[::13] = -1
        previous = output.clone()
        ids.copy_(current)
        graph.replay()
        expected = torch.einsum("mk,mrk->mr", x.float(), weights.float().index_select(0, ids.clamp_min(0)))
        expected[ids < 0] = previous[ids < 0]
        torch.testing.assert_close(output.cpu(), expected.cpu(), atol=2**-10, rtol=2**-10)


@pytest.mark.parametrize("rows", [2049, 4096])
@torch.inference_mode()
def test_shared_cube_routing_replays_changed_indices(rows):
    groups, hidden, rank, width = 5, 80, 8, 64
    x = torch.randn(rows, hidden, device="npu", dtype=torch.bfloat16)
    a = [torch.randn(groups, rank, hidden, device="npu", dtype=torch.bfloat16) for _ in range(2)]
    b = [torch.randn(groups, width, rank, device="npu", dtype=torch.bfloat16) for _ in range(2)]
    ids = (torch.arange(rows, device="npu") % groups).long()
    shrink = [torch.zeros(rows, rank, device="npu", dtype=torch.float32) for _ in range(2)]
    output = torch.zeros(rows, 2 * width, device="npu", dtype=torch.bfloat16)

    def run_shared():
        routing = prepare_cube_bgmv_routing(ids, groups)
        for i in range(2):
            cube_bgmv_shrink(x, a[i], shrink[i], ids, 1.0, routing=routing)
            cube_bgmv_expand(shrink[i], b[i], output, ids, i * width, width, routing=routing)

    for _ in range(2):
        run_shared()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run_shared()
    for shift in (1, 2, 3):
        current = ((torch.arange(rows, device="npu") + shift) % groups).long()
        current[::11] = -1
        ids.copy_(current)
        output.zero_()
        for buffer in shrink:
            buffer.zero_()
        graph.replay()

        expected = torch.zeros_like(output)
        expected_shrink = []
        for i in range(2):
            reference_shrink = torch.zeros_like(shrink[i])
            bgmv_shrink(x, a[i], reference_shrink, ids)
            expected_shrink.append(reference_shrink)
            bgmv_expand_slice(reference_shrink, b[i], expected, ids, i * width, width)
        for actual, reference in zip(shrink, expected_shrink):
            torch.testing.assert_close(actual, reference, atol=2**-14, rtol=2**-10)
        torch.testing.assert_close(output, expected, atol=2**-8, rtol=2**-6)
