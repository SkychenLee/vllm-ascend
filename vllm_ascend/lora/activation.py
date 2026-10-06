# SPDX-License-Identifier: Apache-2.0
"""Bounded fusion of floating MoE LoRA activation and W8A8 quantization."""

import math

import torch
import torch_npu

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import MoECommType

MAX_FUSED_ACTIVATION_ROWS = 48
FUSED_ACTIVATION_WIDTH = 2048
MAX_FUSED_ACTIVATION_LIMIT = 65504


def can_fuse_moe_lora_activation(gate_up_out, mlp_compute_input, comm_type) -> bool:
    context = mlp_compute_input.lora_context
    return (
        envs.VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION
        and comm_type == MoECommType.ALLGATHER
        and getattr(context, "use_ep", False)
        and not getattr(context, "fully_sharded", False)
        and getattr(mlp_compute_input.activation, "value", mlp_compute_input.activation) == "silu"
        and mlp_compute_input.topk_scales is None
        and math.isfinite(mlp_compute_input.swiglu_limit)
        and 0 <= mlp_compute_input.swiglu_limit <= MAX_FUSED_ACTIVATION_LIMIT
        and gate_up_out.device.type == "npu"
        and gate_up_out.dtype in (torch.bfloat16, torch.float16)
        and gate_up_out.ndim == 2
        and 0 < gate_up_out.shape[0] <= MAX_FUSED_ACTIVATION_ROWS
        and gate_up_out.shape[1] == 2 * FUSED_ACTIVATION_WIDTH
        and gate_up_out.is_contiguous()
        # FakeTensor has no storage descriptor. The runtime C++ dispatch
        # checks physical format before capture; real GMM outputs use ND.
        and (torch.compiler.is_compiling() or torch_npu.get_npu_format(gate_up_out) in (0, 2))
        and hasattr(torch.ops._C_ascend, "moe_lora_activation_quant")
    )
