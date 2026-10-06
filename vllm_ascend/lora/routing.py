# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape metadata recovery for AllGather expert-parallel LoRA."""

import torch
import torch_npu

from vllm_ascend import envs
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoELoRARouting

MAX_FUSED_EP_ROUTING_PAIRS = 384
MAX_FUSED_EP_ROUTING_EXPERTS = 512


def recover_allgather_ep_lora_routing(
    *,
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    prepared_lora_indices: torch.Tensor,
    expert_map: torch.Tensor,
    num_dispatched_rows: int,
) -> MoELoRARouting:
    """Invert original-pair -> routed-row indices without reading device scalars.

    Routing capacity may include unwritten rows: only a pair whose destination
    equals the output row and whose expert is local may contribute. In
    particular, negative routing sentinels must never be made positive.
    The returned tensors have exactly the producer's dispatched capacity.
    The producer contract requires unique nonnegative destination indices.
    """
    if topk_ids.ndim != 2 or expanded_row_idx.numel() != topk_ids.numel():
        raise ValueError("AllGather EP LoRA requires one expanded index per (token, topk) pair.")
    if prepared_lora_indices.ndim != 1 or prepared_lora_indices.shape[0] != topk_ids.shape[0]:
        raise ValueError("Prepared adapter slots must align with every prepared token.")
    if prepared_lora_indices.dtype != torch.int64:
        raise ValueError("Prepared adapter slots must have dtype int64.")
    pair_count = topk_ids.numel()
    if not 0 <= num_dispatched_rows <= pair_count:
        raise ValueError("Dispatched capacity must lie between zero and the pair count.")
    max_exact_sort_key = 1 << 24
    if num_dispatched_rows > max_exact_sort_key:
        raise ValueError("AllGather EP LoRA routing capacity exceeds exact FP32 sort-key range.")
    if num_dispatched_rows == 0:
        empty = expanded_row_idx.new_empty((0,), dtype=torch.int64)
        return MoELoRARouting(empty, empty.clone())
    if topk_ids.shape[1] == 0 or expert_map.numel() == 0:
        raise ValueError("Nonempty dispatch requires topk entries and an expert map.")

    if (
        envs.VLLM_ASCEND_MOE_LORA_FUSED_ROUTING
        and pair_count <= MAX_FUSED_EP_ROUTING_PAIRS
        and expert_map.numel() <= MAX_FUSED_EP_ROUTING_EXPERTS
        and expanded_row_idx.ndim == expert_map.ndim == 1
        and expanded_row_idx.dtype in (torch.int32, torch.int64)
        and topk_ids.dtype in (torch.int32, torch.int64)
        and expert_map.dtype in (torch.int32, torch.int64)
        and expanded_row_idx.device.type == "npu"
        and expanded_row_idx.device == topk_ids.device == prepared_lora_indices.device == expert_map.device
        and all(t.is_contiguous() for t in (expanded_row_idx, topk_ids, prepared_lora_indices, expert_map))
        and hasattr(torch.ops._C_ascend, "moe_lora_ep_routing")
        # FakeTensor has no NPU descriptor. During compilation the native
        # runtime guard rejects blocked storage; eager retains its fallback.
        and (
            torch.compiler.is_compiling()
            or all(
                torch_npu.get_npu_format(t) in (0, 2)
                for t in (expanded_row_idx, topk_ids, prepared_lora_indices, expert_map)
            )
        )
    ):
        # InitRouting gives unique nonnegative destinations and negative
        # sentinels for excluded pairs. Single-core UB inverse scatter retains
        # holes and every integer payload without sort or device-host sync.
        return MoELoRARouting(
            *torch.ops._C_ascend.moe_lora_ep_routing(
                expanded_row_idx, topk_ids, prepared_lora_indices, expert_map, num_dispatched_rows
            )
        )

    expanded = expanded_row_idx.reshape(-1).long()
    valid_pair = (expanded >= 0) & (expanded < num_dispatched_rows)
    # INT32/INT64 argsort falls back to AiCPU on A3. Integer keys in
    # [0, capacity] are exactly representable in FP32 within the checked bound.
    keys = torch.where(valid_pair, expanded, num_dispatched_rows).to(torch.float32)
    inverse = torch.argsort(keys)[:num_dispatched_rows]
    rows = torch.arange(num_dispatched_rows, device=expanded.device)
    valid = keys[inverse] == rows
    global_experts = topk_ids.reshape(-1)[inverse].long()
    expert_in_range = (global_experts >= 0) & (global_experts < expert_map.numel())
    # Safe lookup is masked below; it never turns an invalid pair into an expert.
    local_experts = expert_map[global_experts.clamp(0, expert_map.numel() - 1)].long()
    valid = valid & expert_in_range & (local_experts >= 0)
    slots = prepared_lora_indices[inverse // topk_ids.shape[1]]
    return MoELoRARouting(
        torch.where(valid, local_experts, -1),
        torch.where(valid, slots, -1),
    )
