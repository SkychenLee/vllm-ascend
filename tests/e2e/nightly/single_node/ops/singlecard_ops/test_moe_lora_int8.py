# SPDX-License-Identifier: Apache-2.0
"""INT8 LoRA shrink and fused B/add/SwiGLU/quant, including graph replay."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.lora.lora_ops import bgmv_expand_slice, bgmv_shrink
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    assert enable_custom_op(), "Build and load the Ascend custom operators before running this test."


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,hidden,rank",
    [
        (0, 17, 2),
        (1, 17, 1),
        (19, 255, 2),
        (48, 4096, 2),
        (65, 256, 16),
        (9, 4101, 64),
        (8, 1024, 128),
        # Batched rank reductions, tail rank tiles and row scheduling boundary.
        (1, 64, 8),
        (6, 128, 16),
        (48, 256, 8),
        (65, 512, 32),
        (9, 256, 24),
        (9, 128, 40),
        (9, 512, 512),
        (385, 256, 16),
        (769, 256, 16),
        # Both sides of the vector-path guards keep the generic result.
        (9, 63, 8),
        (9, 65, 8),
        (9, 255, 16),
        (9, 257, 16),
        (9, 511, 16),
        (9, 513, 16),
        (9, 256, 7),
        (9, 256, 9),
        (9, 8193, 16),
        (9, 1024, 24),
        (9, 2048, 8),
        (9, 2048, 128),
    ],
)
def test_int8_shrink(dtype, rows, hidden, rank):
    torch.manual_seed(87)
    x = torch.randint(-128, 128, (rows, hidden), dtype=torch.int8)
    x[:, 0] = -128
    scale = torch.rand(rows) * 0.02
    # Input-side router weights may make a prequantized row's scale negative.
    scale[1::2].neg_()
    w = torch.randn(4, rank, hidden, dtype=dtype) * 0.1
    ids = torch.arange(rows) % 5 - 1
    if rows == 1:
        ids.fill_(0)
    ref = torch.zeros(rows, rank)
    for j in range(rows):
        if ids[j] >= 0:
            ref[j] = w[ids[j]].float() @ (x[j].float() * scale[j])
    out = torch.full((rows, rank), float("nan"), device="npu")
    torch.ops._C_ascend.bgmv_shrink_int8(x.npu(), w.npu(), ids.npu(), scale.npu(), out)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden,rank", [(256, 16), (1024, 24), (257, 9)])
def test_int8_shrink_zero_rows_and_invalid_adapters(dtype, hidden, rank):
    x = torch.randint(-127, 128, (5, hidden), device="npu", dtype=torch.int8)
    x[0].zero_()
    scales = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.01], device="npu")
    ids = torch.tensor([0, 1, -1, 2, 1], device="npu")
    weights = torch.randn(2, rank, hidden, device="npu", dtype=dtype) * 0.01
    output = torch.full((5, rank), float("nan"), device="npu")
    torch.ops._C_ascend.bgmv_shrink_int8(x, weights, ids, scales, output)
    expected = weights[1].float().cpu() @ (x[4].float().cpu() * 0.01)
    torch.testing.assert_close(output[:4].cpu(), torch.zeros(4, rank), atol=0, rtol=0)
    torch.testing.assert_close(output[4].cpu(), expected, atol=2e-5, rtol=2e-4)


def fused_reference(base, gate, up, bg, bu, ids, topk, limit):
    # Independent FP64 CPU B projection; retain the old BF16/FP16 add boundary.
    out = base.cpu().clone()
    for plane, (a, b) in enumerate(((gate, bg), (up, bu))):
        width = b.shape[1]
        for row, slot in enumerate(ids.cpu().tolist()):
            if slot >= 0:
                delta = b[slot].cpu().double() @ a[row].cpu().double()
                out[row, plane * width : (plane + 1) * width] = (
                    out[row, plane * width : (plane + 1) * width].double() + delta
                ).to(out.dtype)
    g, u = out.npu().chunk(2, -1)
    if limit > 0:
        g = g.clamp(max=limit)
        u = u.clamp(-limit, limit)
    activated = torch_npu.npu_swiglu(torch.cat((g, u), -1))
    if topk is not None:
        activated *= topk[:, None]
    return torch_npu.npu_dynamic_quant(activated)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,width,rank,limit,weighted",
    [
        (0, 17, 16, 10.0, False),
        (1, 17, 16, 10.0, False),
        (19, 255, 16, 0.0, True),
        (48, 256, 16, 10.0, False),
        (65, 257, 8, 10.0, True),
        (9, 512, 32, 10.0, False),
        (8, 1024, 64, 10.0, False),
        (9, 31, 3, 10.0, True),
        (8, 256, 128, 10.0, False),
        (1, 8192, 64, 10.0, False),
        (1, 32, 512, 10.0, False),
        (9, 256, 16, 10.1, False),
        (33, 257, 128, 10.0, True),
        (9, 511, 256, 10.0, True),
        (9, 65, 512, 10.0, False),
        (1, 8192, 512, 10.0, False),
        (1, 1, 128, 10.0, False),
        (9, 255, 1, 10.0, True),
        (9, 257, 2, 10.0, False),
        (9, 129, 4, 10.0, False),
    ],
)
def test_expand_swiglu_quant(dtype, rows, width, rank, limit, weighted):
    torch.manual_seed(123)
    base = (torch.randn(rows, width * 2, dtype=dtype) * 8).npu()
    g = torch.randn(rows, rank, device="npu")
    u = torch.randn_like(g)
    bg = (torch.randn(4, width, rank, dtype=dtype) * 0.2).npu()
    bu = torch.randn_like(bg) * 0.2
    ids = (torch.arange(rows) % 5 - 1).npu()
    if rows == 1:
        ids.fill_(0)
    topk = torch.linspace(-1, 1, rows, device="npu") if weighted else None
    args = (base, g, u, bg, bu, ids, topk, limit)
    actual, scale = torch.ops._C_ascend.moe_lora_expand_swiglu_quant(*args)
    if rows == 0:
        assert actual.shape == (0, width) and scale.shape == (0,)
        return
    ref, refscale = fused_reference(*args)
    torch.testing.assert_close(scale.cpu(), refscale.cpu(), atol=1e-5, rtol=8e-3)
    torch.testing.assert_close(actual.cpu().float(), ref.cpu().float(), atol=2, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,rank", [(256, 16), (64, 8), (256, 128), (512, 512), (257, 3)])
def test_int8_lora_graph_replay(dtype, width, rank):
    rows = 17
    q = torch.zeros(rows, width, device="npu", dtype=torch.int8)
    scale = torch.zeros(rows, device="npu")
    ids = torch.full((rows,), -1, device="npu", dtype=torch.int64)
    a = torch.randn(4, rank, width, device="npu", dtype=dtype)
    shrink = torch.empty(rows, rank, device="npu")
    base = torch.zeros(rows, width * 2, device="npu", dtype=dtype)
    b = torch.randn(4, width, rank, device="npu", dtype=dtype) * 0.01
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        torch.ops._C_ascend.bgmv_shrink_int8(q, a, ids, scale, shrink)
        out, s = torch.ops._C_ascend.moe_lora_expand_swiglu_quant(base, shrink, shrink, b, b, ids, None, 10.0)
    graph.replay()
    assert torch.count_nonzero(out) == 0 and torch.count_nonzero(s) == 0
    q.random_(-127, 128)
    scale.fill_(0.01)
    ids.copy_(torch.arange(rows, device="npu") % 4)
    graph.replay()
    expected = (a[ids].float() * (q.float() * scale[:, None])[:, None, :]).sum(-1)
    torch.testing.assert_close(shrink.cpu(), expected.cpu(), atol=2e-5, rtol=2e-4)
    ref, rs = fused_reference(base, shrink, shrink, b, b, ids, None, 10.0)
    torch.testing.assert_close(s.cpu(), rs.cpu(), atol=1e-5, rtol=8e-3)
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), atol=2, rtol=0)


def test_prequantized_allgather_routing_keeps_row_scales():
    rows, width, experts = 9, 32, 3
    x = torch.arange(1, rows + 1, dtype=torch.int8).view(-1, 1).expand(rows, width).contiguous().npu()
    scale = (torch.arange(1, rows + 1).float() / 10).npu()
    topk = torch.tensor([[i % experts, (i + 1) % experts] for i in range(rows)], dtype=torch.int32).npu()
    routed, _, counts, routed_scale = torch_npu.npu_moe_init_routing_v2(
        x,
        topk,
        scale=scale,
        active_num=rows * 2,
        expert_num=experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, experts],
        quant_mode=-1,
    )
    assert routed.dtype == torch.int8
    torch.testing.assert_close(routed_scale.cpu(), routed[:, 0].cpu().float() / 10)
    torch.testing.assert_close(counts.cpu(), torch.full((experts,), 6, dtype=counts.dtype))


def test_int8_lora_metadata_validation():
    x = torch.empty((3, 32), device="meta", dtype=torch.int8)
    w = torch.empty((4, 2, 32), device="meta", dtype=torch.bfloat16)
    ids = torch.empty(3, device="meta", dtype=torch.int64)
    scale = torch.empty(3, device="meta")
    y = torch.empty((3, 2), device="meta")
    torch.ops._C_ascend.bgmv_shrink_int8(x, w, ids, scale, y)
    with pytest.raises(RuntimeError, match="row/scale"):
        torch.ops._C_ascend.bgmv_shrink_int8(x, w, ids, scale[:2], y)
    with pytest.raises(RuntimeError, match="dtype"):
        torch.ops._C_ascend.bgmv_shrink_int8(x, w, ids, scale.bfloat16(), y)
    base = torch.empty(3, 16, device="meta", dtype=torch.bfloat16)
    b = torch.empty(4, 8, 2, device="meta", dtype=torch.bfloat16)
    q, s = torch.ops._C_ascend.moe_lora_expand_swiglu_quant(base, y, y, b, b, ids, None, 10.0)
    assert q.shape == (3, 8) and q.dtype == torch.int8
    assert s.shape == (3,) and s.dtype == torch.float32


@pytest.mark.parametrize("adapter_enabled", [[1, 1], [1, 0], [0, 0]])
def test_quantized_mlp_matches_separate_int8_lora_stages(adapter_enabled):
    # Exercise real routing recovery, both GMMs and the Punica interfaces.
    torch.manual_seed(451)
    rows, hidden, intermediate, rank, experts = 32, 128, 64, 16, 2
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._token_lora_indices = (torch.arange(rows, device="npu") % 3 - 1).long()
    wrapper.indices_len = [rows, 0, 0, 0]
    wrapper.bgmv_shrink = bgmv_shrink
    wrapper.bgmv_expand_slice = bgmv_expand_slice

    def weight(*shape):
        return torch.randn(*shape, device="npu", dtype=torch.bfloat16) * 0.05

    context = SimpleNamespace(
        use_ep=False,
        top_k=1,
        tp_rank=0,
        tp_size=1,
        fully_sharded=False,
        punica_wrapper=wrapper,
        adapter_enabled=torch.tensor(adapter_enabled, device="npu", dtype=torch.int32),
        w13_lora_a_stacked=[weight(2, experts, rank, hidden) for _ in range(2)],
        w13_lora_b_stacked=[weight(2, experts, intermediate, rank) for _ in range(2)],
        w2_lora_a_stacked=[weight(2, experts, rank, intermediate)],
        w2_lora_b_stacked=[weight(2, experts, hidden, rank)],
    )
    x, scale = torch_npu.npu_dynamic_quant(weight(rows, hidden))
    payload = MoEMlpComputeInput(
        hidden_states=x,
        dynamic_scale=scale,
        output_dtype=torch.bfloat16,
        group_list=torch.tensor([rows // experts] * experts, device="npu", dtype=torch.int64),
        group_list_type=1,
        topk_scales=None,
        weights=MoEWeights(
            w1=[torch.randint(-20, 21, (experts, hidden, intermediate * 2), device="npu", dtype=torch.int8)],
            w2=[torch.randint(-20, 21, (experts, intermediate, hidden), device="npu", dtype=torch.int8)],
            w1_scale=[torch.full((experts, intermediate * 2), 0.01, device="npu", dtype=torch.bfloat16)],
            w2_scale=[torch.full((experts, hidden), 0.01, device="npu", dtype=torch.bfloat16)],
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
        fusion=True,
        swiglu_limit=10.0,
        expanded_row_idx=torch.arange(rows, device="npu", dtype=torch.int32),
        topk_ids=(torch.arange(rows, device="npu", dtype=torch.int32) // (rows // experts))[:, None],
        lora_context=context,
    )
    with patch("vllm_ascend.lora.quant_moe._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)):
        actual, _ = quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
        expected, _ = quant_apply_mlp_with_moe_lora(mlp_compute_input=replace(payload, fusion=False))
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-5, rtol=0.03)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,hidden,rank",
    [
        (0, 17, 2),
        (1, 17, 1),
        (6, 4096, 2),
        (19, 4101, 3),
        (48, 4096, 2),
        (65, 256, 16),
        (9, 64, 24),
        (9, 1024, 24),
        (9, 8193, 8),
        (9, 128, 512),
        (9, 4096, 64),
        (9, 1024, 128),
        (9, 256, 256),
    ],
)
def test_paired_int8_shrink(dtype, rows, hidden, rank):
    torch.manual_seed(733)
    x = torch.randint(-128, 128, (rows, hidden), dtype=torch.int8)
    scales = torch.linspace(-0.02, 0.02, rows)
    weights = torch.randn(2, 4, rank, hidden, dtype=dtype) * 0.01
    ids = torch.arange(rows) % 6 - 1
    if rows == 1:
        ids[0] = 0
    expected = torch.zeros(rows, 2, rank)
    for row in range(rows):
        if 0 <= ids[row] < 4:
            expected[row] = (weights[:, ids[row]].double() @ (x[row].double() * scales[row])).float()
    out = torch.full((rows, 2 * rank), float("nan"), device="npu")
    torch.ops._C_ascend.bgmv_shrink_int8_pair(x.npu(), weights.npu(), ids.npu(), scales.npu(), out)
    torch.testing.assert_close(out.cpu().reshape(rows, 2, rank), expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,width,shards,local_rank",
    [
        (0, 256, 8, 2),
        (1, 17, 1, 3),
        (19, 255, 2, 3),
        (48, 256, 8, 2),
        (65, 257, 8, 1),
        (9, 512, 2, 16),
        (9, 32, 8, 64),
        (1, 8192, 8, 64),
        (9, 17, 512, 1),
        (9, 17, 127, 1),
        (9, 128, 8, 8),
        (9, 64, 8, 16),
        (9, 32, 8, 32),
    ],
)
def test_paired_expand_consumes_rank_major_layout(dtype, rows, width, shards, local_rank):
    torch.manual_seed(442)
    rank = shards * local_rank
    base = torch.randn(rows, width * 2, device="npu", dtype=dtype) * 8
    paired = torch.randn(shards, rows, 2 * local_rank, device="npu")
    gate = paired[:, :, :local_rank].permute(1, 0, 2).reshape(rows, rank).contiguous()
    up = paired[:, :, local_rank:].permute(1, 0, 2).reshape(rows, rank).contiguous()
    bg = torch.randn(4, width, rank, device="npu", dtype=dtype) * 0.1
    bu = torch.randn_like(bg) * 0.3
    ids = (torch.arange(rows, device="npu") % 6 - 1).long()
    if rows == 1:
        ids.fill_(0)
    topk = torch.linspace(-1, 1, rows, device="npu")
    q, s = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(base, paired, bg, bu, ids, topk, 10.1)
    ref, rs = torch.ops._C_ascend.moe_lora_expand_swiglu_quant(base, gate, up, bg, bu, ids, topk, 10.1)
    # Only the low-rank input layout changes; reduction/rounding stay identical.
    torch.testing.assert_close(q.cpu(), ref.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(s.cpu(), rs.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paired_graph_mapping_and_weight_updates(dtype):
    rows, hidden, rank, width = 19, 257, 16, 64
    x = torch.randint(-128, 128, (rows, hidden), device="npu", dtype=torch.int8)
    scale = torch.rand(rows, device="npu") * 0.01
    ids = torch.full((rows,), -1, device="npu", dtype=torch.int64)
    weights = torch.randn(2, 4, rank, hidden, device="npu", dtype=dtype) * 0.01
    out = torch.empty(rows, 2 * rank, device="npu")
    base = torch.randn(rows, 2 * width, device="npu", dtype=dtype)
    bg = torch.randn(4, width, rank, device="npu", dtype=dtype) * 0.01
    bu = torch.randn_like(bg) * 0.1
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        torch.ops._C_ascend.bgmv_shrink_int8_pair(x, weights, ids, scale, out)
        q, s = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(base, out.unsqueeze(0), bg, bu, ids, None, 10.0)
    for iteration in range(3):
        ids.copy_((torch.arange(rows, device="npu") + iteration) % 5 - 1)
        weights[:, 0].fill_(iteration * 0.01)
        bg[0].fill_(iteration * 0.02)
        graph.replay()
        actual_q, actual_s = q.clone(), s.clone()
        reference = torch.empty_like(out)
        torch.ops._C_ascend.bgmv_shrink_int8_pair(x, weights, ids, scale, reference)
        rq, rs = torch.ops._C_ascend.moe_lora_expand_swiglu_quant(
            base, reference[:, :rank].contiguous(), reference[:, rank:].contiguous(), bg, bu, ids, None, 10.0
        )
        torch.testing.assert_close(actual_q.cpu(), rq.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(actual_s.cpu(), rs.cpu(), atol=0, rtol=0)


def test_paired_meta_and_validation():
    x = torch.empty(3, 32, device="meta", dtype=torch.int8)
    weights = torch.empty(2, 4, 2, 32, device="meta", dtype=torch.bfloat16)
    ids = torch.empty(3, device="meta", dtype=torch.int64)
    scales = torch.empty(3, device="meta")
    out = torch.empty(3, 4, device="meta")
    torch.ops._C_ascend.bgmv_shrink_int8_pair(x, weights, ids, scales, out)
    with pytest.raises(RuntimeError, match="weight/rank"):
        torch.ops._C_ascend.bgmv_shrink_int8_pair(x, weights[:1], ids, scales, out)
    with pytest.raises(RuntimeError, match="row/scale"):
        torch.ops._C_ascend.bgmv_shrink_int8_pair(x, weights, ids, scales[:2], out)
    base = torch.empty(3, 64, device="meta", dtype=torch.bfloat16)
    b = torch.empty(4, 32, 16, device="meta", dtype=torch.bfloat16)
    a = torch.empty(8, 3, 4, device="meta")
    q, s = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(base, a, b, b, ids, None, 10.0)
    assert q.shape == (3, 32) and q.dtype == torch.int8 and s.shape == (3,)
    with pytest.raises(RuntimeError, match="rank shape"):
        torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(base, a[:7], b, b, ids, None, 10.0)
    with pytest.raises(RuntimeError, match="non-contiguous"):
        torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(base, a.transpose(0, 1), b, b, ids, None, 10.0)
