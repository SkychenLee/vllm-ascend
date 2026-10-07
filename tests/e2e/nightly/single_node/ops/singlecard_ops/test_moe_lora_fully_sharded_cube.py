# SPDX-License-Identifier: Apache-2.0
"""NPU arithmetic regression for the eager fully-sharded Cube implementation."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.lora.fully_sharded_cube import CubePrefillRouting, prepare_cube_prefill, shrink_cube_prefill


@pytest.mark.parametrize("pair,slot", [(True, 0), (True, 1), (False, 1)])
def test_cube_matches_native_with_empty_experts_and_disabled_rows(monkeypatch, pair, slot):
    torch.manual_seed(71)
    rows, experts = 16384, 256
    hidden, rank = (4096, 2) if pair else (256, 16)
    counts = torch.zeros(experts, dtype=torch.int64)
    counts[1], counts[100], counts[255] = 8193, 1, 8190
    indices = (torch.repeat_interleave(torch.arange(experts), counts) + slot * experts).npu()
    indices[::7] = -1
    x = torch.randint(-127, 128, (rows, hidden), dtype=torch.int8).npu()
    scales = (torch.rand(rows) * 0.01).npu()
    weights = (torch.randn(2 if pair else 1, 2, experts, rank, hidden) * 0.01).bfloat16().npu()
    output = torch.empty(rows, (2 if pair else 1) * rank, dtype=torch.float32, device=x.device)
    reference = torch.empty_like(output)
    routing = CubePrefillRouting(counts.npu().cumsum(0), slot, rows)
    if pair:
        monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_FULLY_SHARDED_CUBE", "1")
        context = SimpleNamespace(
            fully_sharded=True,
            use_ep=False,
            tp_size=8,
            top_k=6,
            w13_lora_a_packed=weights,
            w2_lora_a_stacked=(torch.empty(2, experts, 16, 256, dtype=torch.bfloat16, device=x.device),),
            punica_wrapper=SimpleNamespace(cube_single_slot=slot),
        )
        forward = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.NONE, in_profile_run=False, is_draft_model=False, attn_metadata={}
        )
        routing = prepare_cube_prefill(context, x, counts.npu(), 1, forward)
        assert routing is not None
    for _ in range(2):
        if pair:
            torch.ops._C_ascend.bgmv_shrink_int8_pair(x, weights.view(2, -1, rank, hidden), indices, scales, reference)
        else:
            torch.ops._C_ascend.bgmv_shrink_int8(x, weights[0].view(-1, rank, hidden), indices, scales, reference)
        shrink_cube_prefill(x, weights.unbind(0), indices, scales, output, routing)
        torch.testing.assert_close(output.cpu(), reference.cpu(), atol=3e-6, rtol=2e-5)
        assert torch.equal(output[::7].cpu(), torch.zeros_like(output[::7].cpu()))
        weights[:, slot].mul_(0.5)
