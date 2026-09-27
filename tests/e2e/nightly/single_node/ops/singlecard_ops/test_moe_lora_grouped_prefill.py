# SPDX-License-Identifier: Apache-2.0
"""Independent routing and precision gates for grouped W8A8 MoE LoRA."""

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

from vllm_ascend.lora.grouped_prefill import prepare_grouped_moe_lora_routing
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    assert enable_custom_op(), "Build the Ascend custom operators before running this test."


def make_ids(rows, experts, adapters, mode="mixed", seed=912):
    """Leave expert and adapter groups empty, interleave adapters, then shuffle."""
    generator = torch.Generator().manual_seed(seed)
    expert = torch.arange(rows) % max(1, experts - 1)
    slot = torch.arange(rows) // max(1, experts - 1) % adapters
    ids = slot * experts + expert
    if mode == "single":
        ids = expert + (adapters - 1) * experts
    elif mode == "disabled":
        ids.fill_(-1)
    if mode != "disabled":
        ids[::11] = -1
        if rows == 1:
            ids[0] = (adapters - 1) * experts
    return ids[torch.randperm(rows, generator=generator)].long()


def cpu_shrink_reference(q, scales, weights, ids, positions):
    """FP64 projection after the contract's FP32 q*scale multiplication."""
    result = torch.zeros(len(positions), weights.shape[1], dtype=torch.float64)
    for destination, row in enumerate(positions):
        slot = int(ids[row])
        if 0 <= slot < weights.shape[0]:
            dequantized = q[row].float() * scales[row].float()
            result[destination] = weights[slot].double() @ dequantized.double()
    return result.float()


@pytest.mark.parametrize(
    "rows,experts,adapters,mode",
    [
        (0, 4, 3, "mixed"),
        (19, 7, 3, "mixed"),
        (2053, 256, 3, "single"),
        (49152, 256, 3, "mixed"),
        (2053, 128, 3, "disabled"),
    ],
)
def test_independent_combined_routing(rows, experts, adapters, mode):
    ids = make_ids(rows, experts, adapters, mode)
    if rows > 1:
        ids[-1] = experts * adapters  # Out-of-range adapters also have zero delta.
    route = prepare_grouped_moe_lora_routing(ids.npu(), experts * adapters)
    order, inverse, sorted_ids = route.order.cpu(), route.inverse.cpu(), route.sorted_indices.cpu()
    torch.testing.assert_close(order.sort().values, torch.arange(rows), atol=0, rtol=0)
    torch.testing.assert_close(order[inverse], torch.arange(rows), atol=0, rtol=0)
    safe = ids.clone()
    safe[(safe < 0) | (safe >= experts * adapters)] = -1
    torch.testing.assert_close(sorted_ids[inverse], safe, atol=0, rtol=0)
    valid = sorted_ids >= 0
    assert torch.all(sorted_ids[valid][1:] >= sorted_ids[valid][:-1])
    if valid.any():
        assert torch.all(valid[: int(valid.sum())])
    # Single adapter is deliberately in the last slot; expert-only indexing
    # would select the wrong weights despite a seemingly identical row order.
    if mode == "single":
        assert torch.all(sorted_ids[valid] >= (adapters - 1) * experts)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "rows,hidden,rank,experts,mode",
    [
        (0, 64, 2, 4, "mixed"),
        (1, 4096, 2, 256, "single"),
        (17, 2048, 2, 128, "mixed"),
        (2053, 4096, 2, 256, "mixed"),
        (2053, 2048, 2, 128, "single"),
        (2053, 256, 16, 256, "disabled"),
        (49152, 4096, 2, 256, "mixed"),
    ],
)
@torch.inference_mode()
def test_grouped_paired_shrink_matches_existing_and_fp32_boundary(dtype, rows, hidden, rank, experts, mode):
    torch.manual_seed(913)
    groups = experts * 3
    ids_cpu = make_ids(rows, experts, 3, mode)
    q_cpu = torch.randint(-128, 128, (rows, hidden), dtype=torch.int8)
    scales_cpu = torch.rand(rows) * 0.01973
    scales_cpu[::3].neg_()
    scales_cpu[::17] = 0
    weights_cpu = torch.randn(2, groups, rank, hidden, dtype=dtype) * 0.01
    q, scales, ids, weights = q_cpu.npu(), scales_cpu.npu(), ids_cpu.npu(), weights_cpu.npu()
    route = prepare_grouped_moe_lora_routing(ids, groups)
    out = torch.full((rows, 2 * rank), float("nan"), device="npu")
    old = torch.empty_like(out)
    torch.ops._C_ascend.bgmv_shrink_int8_pair_grouped(q, weights, route.sorted_indices, route.order, scales, out)
    torch.ops._C_ascend.bgmv_shrink_int8_pair(q, weights, ids, scales, old)
    actual = out[route.inverse].cpu()
    torch.testing.assert_close(actual, old.cpu(), atol=2e-5, rtol=2e-4)
    positions = sorted(set(range(min(rows, 37))) | ({rows - 1} if rows else set()))
    for plane in range(2):
        ref = cpu_shrink_reference(q_cpu, scales_cpu, weights_cpu[plane], ids_cpu, positions)
        torch.testing.assert_close(actual[positions, plane * rank : (plane + 1) * rank], ref, atol=2e-5, rtol=2e-4)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[ids_cpu < 0]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,width,experts", [(0, 128, 128), (17, 128, 128), (2053, 256, 256), (49152, 256, 256)])
@torch.inference_mode()
def test_w2_legacy_shrink_grouped_expand_preserve_unmodified_columns(dtype, rows, width, experts):
    torch.manual_seed(915)
    rank, groups, offset, output_width = 16, experts * 3, 32, 512
    ids = make_ids(rows, experts, 3).npu()
    route = prepare_grouped_moe_lora_routing(ids, groups)
    q = torch.randint(-128, 128, (rows, width), dtype=torch.int8, device="npu")
    scale = torch.rand(rows, device="npu") * 0.013
    a = torch.randn(groups, rank, width, dtype=dtype, device="npu") * 0.01
    b = torch.randn(groups, output_width, rank, dtype=dtype, device="npu") * 0.01
    projected = torch.empty(rows, rank, device="npu")
    torch.ops._C_ascend.bgmv_shrink_int8(q, a, ids, scale, projected)
    base = torch.randn(rows, output_width + 2 * offset, dtype=dtype, device="npu")
    actual, expected = base.clone(), base.clone()
    torch.ops._C_ascend.bgmv_expand_grouped(projected, b, route.sorted_indices, route.order, actual, offset)
    if rows:
        torch.ops._C_ascend.bgmv_expand(projected, b, ids, expected, offset, output_width)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-5, rtol=8e-3)
    torch.testing.assert_close(actual[:, :offset], base[:, :offset], atol=0, rtol=0)
    torch.testing.assert_close(actual[:, -offset:], base[:, -offset:], atol=0, rtol=0)
    torch.testing.assert_close(actual[ids < 0], base[ids < 0], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "rows,width,experts,weighted",
    [(0, 256, 256, False), (17, 96, 128, True), (2053, 256, 256, False), (49152, 256, 256, True)],
)
@torch.inference_mode()
def test_grouped_expand_quant_consumes_original_rank_major_layout(dtype, rows, width, experts, weighted):
    torch.manual_seed(916)
    shards, local_rank, rank, groups = 8, 2, 16, experts * 3
    ids = make_ids(rows, experts, 3).npu()
    route = prepare_grouped_moe_lora_routing(ids, groups)
    base = torch.randn(rows, 2 * width, dtype=dtype, device="npu") * 8
    pair = torch.randn(shards, rows, 2 * local_rank, device="npu") * 0.1
    bg = torch.randn(groups, width, rank, dtype=dtype, device="npu") * 0.01
    bu = torch.randn_like(bg) * 0.02
    topk = torch.linspace(-1, 1, rows, device="npu") if weighted else None
    quantized, scale = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair_grouped(
        base, pair, bg, bu, route.sorted_indices, route.order, topk, 10.0
    )
    expected, expected_scale = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(
        base, pair, bg, bu, ids, topk, 10.0
    )
    actual, actual_scale = quantized[route.inverse], scale[route.inverse]
    torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(), atol=2, rtol=0)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale.cpu(), atol=1e-5, rtol=8e-3)


@torch.inference_mode()
def test_grouped_graph_replay_rebuilds_route_and_reads_weight_updates():
    torch.manual_seed(918)
    rows, hidden, rank, width, groups = 2053, 256, 16, 96, 12
    q = torch.randint(-128, 128, (rows, hidden), dtype=torch.int8, device="npu")
    scale = torch.rand(rows, device="npu") * 0.01973
    ids = torch.full((rows,), -1, dtype=torch.int64, device="npu")
    weights = torch.randn(2, groups, rank, hidden, dtype=torch.bfloat16, device="npu") * 0.01
    base = torch.randn(rows, width * 2, dtype=torch.bfloat16, device="npu")
    bg = torch.randn(groups, width, rank, dtype=torch.bfloat16, device="npu") * 0.01
    bu = torch.randn_like(bg) * 0.02

    def candidate():
        route = prepare_grouped_moe_lora_routing(ids, groups)
        shrink = torch.empty(rows, 2 * rank, device="npu")
        torch.ops._C_ascend.bgmv_shrink_int8_pair_grouped(q, weights, route.sorted_indices, route.order, scale, shrink)
        unsorted = shrink[route.inverse].contiguous()
        out, scales = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair_grouped(
            base, unsorted.unsqueeze(0), bg, bu, route.sorted_indices, route.order, None, 10.0
        )
        return out[route.inverse], scales[route.inverse], unsorted

    for _ in range(3):
        candidate()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, actual_scale, actual_shrink = candidate()
    for step, mode in enumerate(("disabled", "mixed", "single", "mixed", "disabled")):
        ids.copy_(make_ids(rows, 4, 3, mode, seed=step + 10).npu())
        weights[:, step % groups].fill_(0.0137 * (step + 1))
        bg[step % groups].fill_(-0.017 * (step + 1))
        scale.mul_(-0.7)
        graph.replay()
        old = torch.empty_like(actual_shrink)
        torch.ops._C_ascend.bgmv_shrink_int8_pair(q, weights, ids, scale, old)
        expected, expected_scale = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(
            base, old.unsqueeze(0), bg, bu, ids, None, 10.0
        )
        torch.testing.assert_close(actual_shrink.cpu(), old.cpu(), atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(), atol=2, rtol=0)
        torch.testing.assert_close(actual_scale.cpu(), expected_scale.cpu(), atol=1e-5, rtol=8e-3)


@pytest.mark.parametrize("device", ["meta", "npu"])
def test_grouped_native_meta_and_invalid_layouts(device):
    rows, hidden, groups, rank = 17, 256, 4, 2
    q = torch.zeros(rows, hidden, device=device, dtype=torch.int8)
    w = torch.zeros(2, groups, rank, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.zeros(rows, device=device, dtype=torch.int64)
    order = torch.arange(rows, device=device)
    scales = torch.ones(rows, device=device)
    output = torch.empty(rows, 2 * rank, device=device)
    torch.ops._C_ascend.bgmv_shrink_int8_pair_grouped(q, w, ids, order, scales, output)
    invalid = (
        (q.float(), w, ids, ids, scales, output),
        (q, w, ids.int(), ids, scales, output),
        (q, w, ids, ids[:-1], scales, output),
        (q, w[:1], ids, ids, scales, output),
        (q, w, ids, ids, scales, output[:, ::2]),
        (q[:, ::2], w[:, :, :, ::2], ids, ids, scales, output),
    )
    for arguments in invalid:
        with pytest.raises(RuntimeError):
            torch.ops._C_ascend.bgmv_shrink_int8_pair_grouped(*arguments)

    base = torch.zeros(rows, 64, device=device, dtype=torch.bfloat16)
    paired = torch.zeros(8, rows, 4, device=device)
    b = torch.zeros(groups, 32, 16, device=device, dtype=torch.bfloat16)
    out, scale = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair_grouped(
        base, paired, b, b, ids, order, None, 10.0
    )
    assert out.shape == (rows, 32) and scale.shape == (rows,)
    for arguments in (
        (base, paired[:7], b, b, ids, order, None, 10.0),
        (base, paired, b, b, ids, order[:-1], None, 10.0),
        (base, paired, b, b, ids, order, None, -1.0),
        (base.float(), paired, b, b, ids, order, None, 10.0),
    ):
        with pytest.raises(RuntimeError):
            torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair_grouped(*arguments)
    projected = torch.zeros(rows, 16, device=device)
    for offset in (-16, 1, 48):
        with pytest.raises(RuntimeError):
            torch.ops._C_ascend.bgmv_expand_grouped(projected, b, ids, order, base, offset)
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.bgmv_expand_grouped(projected[:, :7], b[:, :, :7], ids, order, base, 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank,width", [(8, 512), (16, 512), (32, 256), (64, 128)])
@torch.inference_mode()
def test_grouped_expand_add_matches_legacy_reduction_exactly(dtype, rank, width):
    torch.manual_seed(921)
    rows, groups, offset = 4097, 21, 32
    ids_cpu = make_ids(rows, 7, 3)
    ids = ids_cpu.npu()
    route = prepare_grouped_moe_lora_routing(ids, groups)
    projected = torch.randn(rows, rank, device="npu") * 0.3
    weights = torch.randn(groups, width, rank, device="npu", dtype=dtype) * 0.05
    base = torch.randn(rows, width + 2 * offset, device="npu", dtype=dtype)
    expected, actual = base.clone(), base.clone()
    torch.ops._C_ascend.bgmv_expand(projected, weights, ids, expected, offset, width)
    torch.ops._C_ascend.bgmv_expand_grouped(projected, weights, route.sorted_indices, route.order, actual, offset)
    # Identical inputs isolate the B reduction from upstream A error.
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(actual[ids < 0], base[ids < 0], atol=0, rtol=0)
    for row in (1, 7, 19, rows - 1):
        slot = int(ids_cpu[row])
        if slot < 0:
            continue
        ref = (
            base[row, offset : offset + width].cpu().double()
            + weights[slot].cpu().double() @ projected[row].cpu().double()
        ).to(dtype)
        torch.testing.assert_close(actual[row, offset : offset + width].cpu(), ref, atol=2e-5, rtol=8e-3)
