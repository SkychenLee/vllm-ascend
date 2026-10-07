# SPDX-License-Identifier: Apache-2.0
"""Bounded, eager prefill Cube projections for single-adapter TP8 MoE LoRA.

The existing AllGather producer supplies expert-major rows and device counts.
INT8 activations and BF16 weights convert exactly to FP32; the row scale is
applied after the matrix product. Reduction order therefore differs from BGMV.
No expert-count readback, persistent weight copy or shared scratch is needed.
"""

from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend import envs
from vllm_ascend.lora.cube_moe import supports_cube_target_prefill_context
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

MIN_CUBE_ROWS = 16384
MAX_CUBE_ROWS = 98304
NUM_EXPERTS = 256
HIDDEN_SIZE = 4096
LOCAL_RANK = 2
FULL_RANK = 16
INTERMEDIATE_SIZE = 256
MAX_INPUT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class CubePrefillRouting:
    """One forward's expert boundaries and CPU-confirmed adapter slot."""

    ends: torch.Tensor
    slot: int
    rows: int


def prepare_cube_prefill(context, x, counts, group_list_type, forward_context):
    """Return routing only for the measured layout and a real eager prefill."""
    if not (
        envs.VLLM_ASCEND_MOE_LORA_FULLY_SHARDED_CUBE
        and get_ascend_device_type() == AscendDeviceType.A3
        and not torch.compiler.is_compiling()
        and supports_cube_target_prefill_context(forward_context)
        and context is not None
        and getattr(context, "fully_sharded", None) is True
        and getattr(context, "use_ep", None) is False
        and getattr(context, "tp_size", None) == 8
        and getattr(context, "top_k", None) == 6
        and group_list_type == 1
        and counts is not None
        and x.ndim == 2
        and MIN_CUBE_ROWS <= x.shape[0] <= MAX_CUBE_ROWS
        and x.shape[1] == HIDDEN_SIZE
        and x.dtype == torch.int8
        and x.device.type == "npu"
        and x.is_contiguous()
        and counts.shape == (NUM_EXPERTS,)
        and counts.dtype == torch.int64
        and counts.device == x.device
        and counts.is_contiguous()
    ):
        return None
    slot = getattr(context.punica_wrapper, "cube_single_slot", None)
    paired = getattr(context, "w13_lora_a_packed", None)
    if paired is None or type(slot) is not int:
        return None
    down = context.w2_lora_a_stacked[0]
    if not (
        paired.ndim == 5
        and paired.shape[0] == 2
        and paired.shape[2:] == (NUM_EXPERTS, LOCAL_RANK, HIDDEN_SIZE)
        and 0 <= slot < paired.shape[1]
        and down.shape == (paired.shape[1], NUM_EXPERTS, FULL_RANK, INTERMEDIATE_SIZE)
        and all(w.dtype == torch.bfloat16 and w.device == x.device and w.is_contiguous() for w in (paired, down))
        and all(torch_npu.get_npu_format(t) in (0, 2) for t in (x, counts))
        # Five-dimensional packed A storage uses unblocked NCDHW (30).
        # Selecting a projection/slot gives the ND matrices consumed below.
        and all(torch_npu.get_npu_format(t) in (0, 2, 30) for t in (paired, down))
    ):
        return None
    # This is a device operation. No count or routing value crosses to CPU.
    return CubePrefillRouting(counts.cumsum(0), slot, x.shape[0])


def shrink_cube_prefill(x, weights, combined, scale, output, routing):
    """Write FP32 A projections in original expert-row order, including masks.

    Gate/up share one GMM with adjacent output columns. W2 uses one projection.
    Existing callers retain their original AllGather/AllReduce and B kernels.
    Weight conversion is per invocation, so adapter reloads are immediately
    visible without a generation cache or per-layer permanent allocation.
    """
    rows, hidden = x.shape
    if not (
        len(weights) in (1, 2)
        and rows == routing.rows
        and combined.shape == (rows,)
        and combined.dtype == torch.int64
        and scale.shape == (rows,)
        and scale.dtype == torch.float32
        and output.shape == (rows, sum(w.shape[-2] for w in weights))
        and output.dtype == torch.float32
        and x.dtype == torch.int8
        and all(t.device == x.device for t in (combined, scale, output, routing.ends, *weights))
        and all(w.shape == weights[0].shape and w.shape[-1] == hidden for w in weights)
    ):
        raise ValueError("Fully sharded Cube projection metadata does not match its routing")
    selected = tuple(w[routing.slot] for w in weights)
    packed = selected[0] if len(selected) == 1 else torch.cat(selected, dim=1)
    matrix = packed.float().transpose(1, 2)
    experts = weights[0].shape[1]
    first = routing.slot * experts
    # Combined indices re-read the adapter enable mask for each forward.
    # Padding/disabled rows must be exactly zero before TP communication.
    active = (combined >= first) & (combined < first + experts)
    chunk_rows = MAX_INPUT_BYTES // (hidden * torch.float32.itemsize)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        ends = (routing.ends - start).clamp(0, stop - start)
        product = torch_npu.npu_grouped_matmul(
            x=[x[start:stop].float()],
            weight=[matrix],
            group_list=ends,
            group_list_type=0,
            split_item=2,
            group_type=0,
            output_dtype=torch.float32,
        )[0]
        product.mul_(scale[start:stop, None])
        product.masked_fill_(~active[start:stop, None], 0)
        output[start:stop].copy_(product)
