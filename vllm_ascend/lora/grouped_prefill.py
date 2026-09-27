# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independent, fixed-size (expert, adapter) grouping for MoE LoRA Prefill."""

from typing import NamedTuple

import torch

from vllm_ascend import envs

ENABLE_GROUPED_PREFILL = envs.VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL
GROUPED_PREFILL_MIN_ROWS = 8192
GROUPED_PREFILL_ROWS_PER_GROUP = 8
GROUPED_A_CACHE_ELEMENTS = 16384


class GroupedMoELoRARouting(NamedTuple):
    order: torch.Tensor
    inverse: torch.Tensor
    sorted_indices: torch.Tensor


def prepare_grouped_moe_lora_routing(combined_indices: torch.Tensor, groups: int) -> GroupedMoELoRARouting:
    """Group every batch independently, including batches with one adapter.

    Only the row mapping is sorted. Full-width activations remain in place.
    Invalid/disabled adapters sort last and retain the -1 no-delta sentinel.
    The fixed shapes permit graph replay with updated mapping buffers.
    """
    if combined_indices.ndim != 1 or combined_indices.dtype != torch.int64:
        raise ValueError("Grouped MoE LoRA requires one-dimensional INT64 indices")
    if not 0 < groups < (1 << 24) or combined_indices.numel() > (1 << 24):
        raise ValueError("Grouped MoE LoRA groups and rows must fit exact FP32 sorting keys")
    active = (combined_indices >= 0) & (combined_indices < groups)
    keys = torch.where(active, combined_indices, groups)
    order = torch.argsort(keys.float())
    # INT64 scatter is expensive on this device; the complete permutation
    # has exact FP32 keys and its argsort is precisely the inverse map.
    inverse = torch.argsort(order.float())
    sorted_indices = torch.where(active, combined_indices, -1).index_select(0, order)
    return GroupedMoELoRARouting(order, inverse, sorted_indices)


def can_use_grouped_moe_lora(lora_context, hidden_states: torch.Tensor, intermediate_size: int) -> bool:
    """Conservative static guard; no host reads of adapter/expert occupancy."""
    if not ENABLE_GROUPED_PREFILL or not lora_context.fully_sharded or getattr(lora_context, "use_ep", True):
        return False
    packed = getattr(lora_context, "w13_lora_a_packed", None)
    if packed is None or hidden_states.dtype != torch.int8 or hidden_states.device.type != "npu":
        return False
    rows, hidden = hidden_states.shape
    local_rank = packed.shape[-2]
    groups = packed.shape[1] * packed.shape[2]
    rank = lora_context.w13_lora_b_stacked[0].shape[-1]
    return (
        (1 << 24) >= rows >= GROUPED_PREFILL_MIN_ROWS
        and rows >= groups * GROUPED_PREFILL_ROWS_PER_GROUP
        and 0 < groups < (1 << 24)
        and 0 < hidden <= 4096
        and hidden % 64 == 0
        and 2 * local_rank * hidden <= GROUPED_A_CACHE_ELEMENTS
        and rank in (8, 16, 32)
        and 0 < intermediate_size <= 256
        and intermediate_size % 32 == 0
        and all(
            hasattr(torch.ops._C_ascend, op)
            for op in (
                "bgmv_shrink_int8_pair_grouped",
                "moe_lora_expand_swiglu_quant_pair_grouped",
                "bgmv_expand_grouped",
            )
        )
    )
