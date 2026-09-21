"""Grouped Cube implementation for large batches of LoRA rows."""

from typing import NamedTuple

import torch
import torch_npu

from vllm_ascend import envs


CUBE_BGMV_MIN_ROWS = 2049
# The grouped Cube candidate changes BF16 model output in the current 8-card
# precision gate. Keep it opt-in until that model-level gate is accepted.
ENABLE_CUBE_BGMV = envs.VLLM_ASCEND_ENABLE_CUBE_BGMV
_MAX_GROUPS = 1024
_MAX_INNER_DIM = 65536
_MOE_ROUTING_MAX_ROWS = 1 << 24


class CubeBGMVRouting(NamedTuple):
    rows: int
    groups: int
    order: torch.Tensor
    inverse: torch.Tensor
    group_ends: torch.Tensor


def prepare_cube_bgmv_routing(indices: torch.Tensor, groups: int) -> CubeBGMVRouting:
    """Build reusable routing for several projections with the same row IDs."""
    rows = indices.shape[0]
    safe_indices = indices.clamp_min(0)
    order = torch.argsort(safe_indices.float())
    inverse = torch.empty_like(order)
    inverse.index_copy_(0, order, torch.arange(rows, device=indices.device, dtype=order.dtype))
    sorted_indices = safe_indices.index_select(0, order)
    group_ends = torch.searchsorted(
        sorted_indices, torch.arange(groups, device=indices.device, dtype=indices.dtype), right=True
    )
    return CubeBGMVRouting(rows, groups, order, inverse, group_ends)


def can_use_cube_bgmv(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Check only static metadata; the route remains capturable in an NPU graph."""
    if inputs.ndim != 2 or output.ndim != 2 or weights.ndim != 3 or indices.ndim != 1:
        return False
    rows = inputs.shape[0]
    groups = weights.shape[0]
    return (
        ENABLE_CUBE_BGMV
        and rows >= CUBE_BGMV_MIN_ROWS
        and inputs.device.type == "npu"
        and indices.shape[0] == output.shape[0] == rows
        and 1 <= groups <= _MAX_GROUPS
        and 0 < inputs.shape[1] < _MAX_INNER_DIM
        and 0 < output.shape[1] < _MAX_INNER_DIM
        and weights.shape[2] == inputs.shape[1]
        and weights.shape[1] <= output.shape[1]
        and inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weights.dtype in (torch.float16, torch.bfloat16)
        and indices.dtype == torch.int64
        and inputs.is_contiguous()
        and weights.is_contiguous()
        and indices.is_contiguous()
        and all(t.device == inputs.device for t in (weights, indices, output))
    )


def _grouped_matmul(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    routing: CubeBGMVRouting | None = None,
) -> torch.Tensor:
    """Sort by LoRA slot, run Cube GMM, then restore the original row order."""
    rows, _ = inputs.shape
    groups, out_dim, _ = weights.shape
    # Negative IDs preserve the existing output. Route them with group zero to
    # keep the sort shape static, then mask their results in the caller.
    if groups == 1:
        return inputs.float() @ weights[0].float().T
    if routing is not None:
        if routing.rows != rows or routing.groups != groups:
            raise ValueError("Cube BGMV routing does not match the input shape")
        sorted_inputs = inputs.index_select(0, routing.order)
        group_ends = routing.group_ends
    else:
        safe_indices = indices.clamp_min(0)
    use_moe_routing = routing is None and rows < _MOE_ROUTING_MAX_ROWS and hasattr(torch_npu, "npu_moe_init_routing")
    if use_moe_routing:
        # This operator fuses the sort and input gather, and returns the map
        # from each original row to its sorted position.
        row_numbers = torch.arange(rows, device=indices.device, dtype=torch.int32).view(rows, 1)
        sorted_inputs, inverse, sorted_indices = torch_npu.npu_moe_init_routing(
            inputs, row_numbers, safe_indices.int().view(rows, 1), rows
        )
        group_ends = torch.searchsorted(
            sorted_indices, torch.arange(groups, device=indices.device, dtype=torch.int32), right=True
        )
    elif routing is None:
        # Int64 argsort runs on AiCPU on this device. IDs bounded by
        # _MAX_GROUPS are exactly representable as FP32 sorting keys.
        order = torch.argsort(safe_indices.float())
        sorted_indices = safe_indices.index_select(0, order)
        sorted_inputs = inputs.index_select(0, order)
        group_ends = torch.searchsorted(
            sorted_indices, torch.arange(groups, device=indices.device, dtype=indices.dtype), right=True
        )
    # The non-quantized GMM on this device returns BF16 for BF16 operands.
    # FP32 operands/output preserve the FP32 shrink buffer and expand add.
    sorted_output = torch_npu.npu_grouped_matmul(
        x=[sorted_inputs.float()],
        weight=[weights.float().transpose(1, 2)],
        group_list=group_ends,
        group_type=0,
        group_list_type=0,
        split_item=2,
        output_dtype=torch.float32,
    )[0]
    if routing is not None:
        return sorted_output.index_select(0, routing.inverse)
    if use_moe_routing:
        return sorted_output.index_select(0, inverse.long())
    output = torch.empty((rows, out_dim), dtype=torch.float32, device=inputs.device)
    output.index_copy_(0, order, sorted_output)
    return output


def cube_bgmv_shrink(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    routing: CubeBGMVRouting | None = None,
) -> None:
    result = _grouped_matmul(inputs, weights, indices, routing=routing) * scale
    output.copy_(torch.where(indices[:, None] >= 0, result, output))


def cube_bgmv_expand(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    indices: torch.Tensor,
    offset: int,
    width: int,
    routing: CubeBGMVRouting | None = None,
) -> None:
    result = _grouped_matmul(inputs, weights, indices, routing=routing)
    target = output[:, offset : offset + width]
    updated = target.float() + result
    target.copy_(torch.where(indices[:, None] >= 0, updated.to(target.dtype), target))
