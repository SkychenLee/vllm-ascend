# SPDX-License-Identifier: Apache-2.0
"""Bounded, optional fusion of MoE LoRA routing metadata operations."""

import torch

from vllm_ascend import envs

# Scalar INT64 arithmetic preserves every payload bit. Bound dispatch to the
# measured target: max batched tokens 16384 times routed top-k 6.
MAX_FUSED_ROUTING_ROWS = 98304
MAX_FUSED_ROUTING_ADAPTERS = 64


def combined_moe_lora_indices(experts, slots, adapter_enabled, num_experts):
    if (
        envs.VLLM_ASCEND_MOE_LORA_FUSED_ROUTING
        and slots.device.type == "npu"
        and experts.device == slots.device == adapter_enabled.device
        and experts.dim() == slots.dim() == adapter_enabled.dim() == 1
        and experts.numel() == slots.numel()
        and slots.numel() <= MAX_FUSED_ROUTING_ROWS
        and 0 < adapter_enabled.numel() <= MAX_FUSED_ROUTING_ADAPTERS
        and experts.dtype in (torch.int32, torch.int64)
        and slots.dtype == torch.int64
        and adapter_enabled.dtype == torch.int32
        and experts.is_contiguous()
        and slots.is_contiguous()
        and adapter_enabled.is_contiguous()
        and num_experts > 0
        and hasattr(torch.ops._C_ascend, "moe_lora_combined_indices")
    ):
        return torch.ops._C_ascend.moe_lora_combined_indices(experts, slots, adapter_enabled, num_experts)
    # Preserve the old path, including its valid-domain/exception behavior.
    expert_idx = experts.view(-1).to(torch.long)
    safe_slots = slots.clamp(min=0)
    enabled = (slots >= 0) & adapter_enabled[safe_slots].bool()
    return torch.where(enabled, safe_slots * num_experts + expert_idx, torch.full_like(slots, -1)).contiguous()
