# SPDX-License-Identifier: Apache-2.0
"""Graph-safe single-adapter specialization of fully-sharded MoE routing.

The native operation reads the current token slots and enable mask on every
execution. Its cached first-slot arithmetic is a hint; other adapters and
inactive rows use the generic device-side calculation. No adapter identity
or batch mapping is cached in Python or baked into a captured graph.
"""

import torch
import torch_npu

from vllm_ascend import envs

MAX_SINGLE_ADAPTER_SCALAR_WORK = 1024
MIN_SINGLE_ADAPTER_LARGE_ROWS = 8161
MAX_SINGLE_ADAPTER_ROWS = 98304
MAX_SINGLE_ADAPTER_SLOTS = 64


def try_single_adapter_indices(lora_context, expanded_row_idx, topk_ids):
    """Return fused combined indices, or None to retain the existing path."""
    if not (
        envs.VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER
        and lora_context is not None
        and getattr(lora_context, "fully_sharded", None) is True
        and getattr(lora_context, "use_ep", None) is False
        and expanded_row_idx is not None
        and topk_ids is not None
        and hasattr(torch.ops._C_ascend, "moe_lora_single_adapter_indices")
    ):
        return None

    slots = lora_context.punica_wrapper.token_lora_indices
    enabled = lora_context.adapter_enabled
    top_k = lora_context.top_k
    w13 = lora_context.w13_lora_a_stacked[0]
    w2 = lora_context.w2_lora_a_stacked[0]
    if not (
        isinstance(top_k, int)
        and top_k > 0
        and expanded_row_idx.ndim == slots.ndim == enabled.ndim == 1
        and topk_ids.ndim == 2
        and topk_ids.shape[1] == top_k
        and expanded_row_idx.numel() == topk_ids.numel()
        and expanded_row_idx.dtype in (torch.int32, torch.int64)
        and topk_ids.dtype in (torch.int32, torch.int64)
        and slots.dtype == torch.int64
        and enabled.dtype == torch.int32
        and expanded_row_idx.device.type == "npu"
        and expanded_row_idx.device == topk_ids.device == slots.device == enabled.device
        and w13.shape[:2] == w2.shape[:2]
        and 0 < w13.shape[0] <= MAX_SINGLE_ADAPTER_SLOTS
        and enabled.numel() in (w13.shape[0], w13.shape[0] + 1)
        and w13.shape[1] > 0
    ):
        return None
    rows = expanded_row_idx.numel()
    if not (
        rows <= MAX_SINGLE_ADAPTER_ROWS
        and (rows == 0 or slots.numel() > 0)
        and (rows + topk_ids.shape[0] <= MAX_SINGLE_ADAPTER_SCALAR_WORK or rows >= MIN_SINGLE_ADAPTER_LARGE_ROWS)
    ):
        return None

    # FakeTensor has no physical storage descriptor. Native dispatch checks
    # the real storage before capture/replay, even for a compiled callable.
    inputs = (expanded_row_idx, topk_ids, slots, enabled)
    if not torch.compiler.is_compiling() and any(torch_npu.get_npu_format(tensor) not in (0, 2) for tensor in inputs):
        return None
    # vLLM reserves one additional enabled entry after the allocated adapters.
    # It is not a weight slot. Limit lookup to the actual weight table while
    # retaining the original descriptor guard and per-replay enable reads.
    if enabled.numel() != w13.shape[0]:
        enabled = enabled[: w13.shape[0]]
        inputs = (expanded_row_idx, topk_ids, slots, enabled)
    return torch.ops._C_ascend.moe_lora_single_adapter_indices(
        *(tensor.contiguous() for tensor in inputs), top_k, w13.shape[1]
    )
