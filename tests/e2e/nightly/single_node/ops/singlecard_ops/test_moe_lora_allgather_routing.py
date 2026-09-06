# SPDX-License-Identifier: Apache-2.0
"""Check actual NPU AllGather dispatch against a CPU token/expert oracle."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

import vllm_ascend.ops  # noqa: F401 -- initializes Ascend/Punica integration
from vllm_ascend.lora.fused_moe import (
    _prepare_moe_lora_bgmv_indices_allgather,
    _recover_moe_lora_routing_allgather,
)
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 128, 512, 513, 2048])
@pytest.mark.parametrize("rank", range(8))
@torch.inference_mode()
def test_ep_allgather_bgmv_indices_match_dispatch(tokens, rank):
    enable_custom_op()
    experts, local_experts, top_k = 256, 32, 8
    start = rank * local_experts
    topk_cpu = (torch.arange(tokens * top_k, dtype=torch.int32).reshape(tokens, top_k) * 29 + 7) % experts
    if tokens == 1 and rank == 7:
        topk_cpu.fill_(0)  # Empty local expert population.
    hidden = torch.arange(tokens, dtype=torch.float32).reshape(-1, 1).expand(-1, 16).contiguous().npu()
    topk = topk_cpu.npu()
    routed, mapping, counts, _ = torch_npu.npu_moe_init_routing_v2(
        hidden,
        topk,
        active_num=tokens * top_k,
        expert_num=experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[start, start + local_experts],
        quant_mode=-1,
        row_idx_type=0,
    )
    expert_map_cpu = torch.full((experts,), -1, dtype=torch.int32)
    expert_map_cpu[start : start + local_experts] = torch.arange(local_experts, dtype=torch.int32)
    expert_map = expert_map_cpu.npu()
    slots_cpu = torch.arange(tokens, dtype=torch.int64) % 4 - 1
    enabled_cpu = torch.tensor([1, 1, 0], dtype=torch.int32)
    context = SimpleNamespace(top_k=top_k, allgather_lora_indices=slots_cpu.npu())
    actual = _prepare_moe_lora_bgmv_indices_allgather(
        context, mapping, topk, expert_map, enabled_cpu.npu(), local_experts
    ).cpu()
    recovered_experts, recovered_slots = _recover_moe_lora_routing_allgather(context, mapping, topk, expert_map)
    recovered_experts, recovered_slots = recovered_experts.cpu(), recovered_slots.cpu()
    recovered = torch.where(
        (recovered_slots >= 0) & enabled_cpu[recovered_slots.clamp(min=0)].bool(),
        recovered_slots * local_experts + recovered_experts,
        -1,
    )
    expected = torch.full((tokens * top_k,), -1, dtype=torch.int64)
    mapping_cpu = mapping.cpu().to(torch.float32).abs().to(torch.int64)
    routed_cpu = routed.cpu()
    local_pairs = 0
    for pair, expert in enumerate(topk_cpu.flatten().tolist()):
        if start <= expert < start + local_experts:
            local_pairs += 1
            destination = int(mapping_cpu[pair])
            token = pair // top_k
            assert int(routed_cpu[destination, 0]) == token
            slot = int(slots_cpu[token])
            if slot >= 0 and int(enabled_cpu[slot]):
                expected[destination] = slot * local_experts + expert - start
    assert local_pairs == int(counts.sum().cpu())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
