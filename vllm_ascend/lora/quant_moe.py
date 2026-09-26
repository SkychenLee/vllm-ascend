# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extensible quantized MoE LoRA execution for Ascend.

Each quantization scheme owns its activation contract and MLP implementation.
W8A8 AllGather shares quantized activations between the base experts and LoRA;
AlltoAll retains floating-point LoRA boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.fused_moe import (
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
    moe_lora_apply_w13_swiglu_quant,
    reset_lora_indices,
)
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType

# The native operator supports width <= 8192 and rank <= 512. Dispatch only
# the measured beneficial region: wide projections or scalar-rank fallback
# can cost more than separate B/activation/quant kernels, especially Decode.
MAX_FUSED_LORA_INTERMEDIATE_SIZE = 256
MAX_FUSED_LORA_PROJECTION_ELEMENTS = 8192


def _can_fuse_int8_lora_swiglu(lora_context, intermediate_size: int) -> bool:
    if len(lora_context.w13_lora_a_stacked) != 2:
        return False
    rank = lora_context.w13_lora_b_stacked[0].shape[-1]
    return (
        rank >= 8
        and rank & (rank - 1) == 0
        and intermediate_size % 32 == 0
        and intermediate_size <= MAX_FUSED_LORA_INTERMEDIATE_SIZE
        and intermediate_size * rank <= MAX_FUSED_LORA_PROJECTION_ELEMENTS
    )


QuantMoELoRAApply = Callable[[MoEMlpComputeInput, Any], tuple[torch.Tensor, torch.npu.Event | None]]
QuantMoELoRAActivationValidator = Callable[[torch.Tensor, torch.Tensor | None], None]


@dataclass(frozen=True)
class QuantMoELoRAImpl:
    apply: QuantMoELoRAApply
    validate_activation_input: QuantMoELoRAActivationValidator | None


_QUANT_MOE_LORA_IMPLS: dict[QuantType, QuantMoELoRAImpl] = {}


def register_quant_moe_lora_impl(
    quant_type: QuantType,
    *,
    validate_activation_input: QuantMoELoRAActivationValidator | None = None,
):
    """Register one quantized MoE LoRA implementation."""

    def decorator(apply: QuantMoELoRAApply) -> QuantMoELoRAApply:
        if quant_type in _QUANT_MOE_LORA_IMPLS:
            raise ValueError(f"Quantized MoE LoRA implementation already registered for {quant_type}.")
        _QUANT_MOE_LORA_IMPLS[quant_type] = QuantMoELoRAImpl(
            apply=apply,
            validate_activation_input=validate_activation_input,
        )
        return apply

    return decorator


def _get_quant_moe_lora_impl(quant_type: QuantType) -> QuantMoELoRAImpl:
    impl = _QUANT_MOE_LORA_IMPLS.get(quant_type)
    if impl is None:
        supported = ", ".join(item.name for item in _QUANT_MOE_LORA_IMPLS)
        raise NotImplementedError(
            "Ascend quantized MoE LoRA has no implementation registered for "
            f"{quant_type.name}. Registered quant types: {supported or 'none'}."
        )
    return impl


def quant_apply_mlp_with_moe_lora(
    *,
    mlp_compute_input: MoEMlpComputeInput,
    quant_method=None,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Dispatch an active quantized MoE LoRA batch to its backend."""
    return _get_quant_moe_lora_impl(mlp_compute_input.quant.quant_type).apply(mlp_compute_input, quant_method)


def validate_quant_moe_lora_activation_input(
    *,
    quant_type: QuantType,
    hidden_states: torch.Tensor,
    dynamic_scale: torch.Tensor | None,
    comm_type: MoECommType | None = None,
) -> None:
    """Validate activations before quantized MoE LoRA prepare/dispatch."""
    impl = _get_quant_moe_lora_impl(quant_type)
    if quant_type == QuantType.W8A8 and comm_type == MoECommType.ALLGATHER:
        _validate_int8_scale(hidden_states, dynamic_scale)
        return
    if impl.validate_activation_input is not None:
        impl.validate_activation_input(hidden_states, dynamic_scale)


def _apply_moe_activation(
    gate_up_out: torch.Tensor,
    activation: str | None,
    swiglu_limit: float,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> torch.Tensor:
    """Match the activation semantics of the common unquantized MoE path."""
    act_name = getattr(activation, "value", activation)
    if activation == MoEActivation.SWIGLUOAI:
        return AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out)
    if act_name == "swigluoai_uninterleave":
        return torch_npu.npu_clipped_swiglu(
            gate_up_out,
            interleaved=False,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            bias=swiglu_beta,
        )
    if activation == MoEActivation.SWIGLUSTEP:
        return AscendSwigluStepAndMul.swiglustep_forward(gate_up_out, limit=swiglu_limit or 7.0)
    if activation in (MoEActivation.GELU, MoEActivation.GELU_TANH):
        gate, up = gate_up_out.chunk(2, dim=-1)
        approximate = "tanh" if activation == MoEActivation.GELU_TANH else "none"
        return torch.nn.functional.gelu(gate, approximate=approximate) * up
    if swiglu_limit > 0:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        gate_up_out = torch.cat((gate, up), dim=-1)
    return torch_npu.npu_swiglu(gate_up_out)


def _validate_dynamic_int8_activations(
    hidden_states: torch.Tensor,
    dynamic_scale: torch.Tensor | None,
) -> None:
    if dynamic_scale is not None or hidden_states.dtype == torch.int8:
        raise NotImplementedError("Dynamic INT8 MoE LoRA requires unquantized activations before expert routing.")


def _validate_int8_scale(hidden_states: torch.Tensor, scale: torch.Tensor | None) -> None:
    if hidden_states.dtype == torch.int8:
        if scale is None:
            raise ValueError("INT8 MoE LoRA activations require a per-token dynamic_scale.")
        if scale.dtype != torch.float32 or scale.device != hidden_states.device:
            raise ValueError("INT8 MoE LoRA dynamic_scale must be FP32 on the activation device.")
        if scale.shape not in (hidden_states.shape[:-1], (*hidden_states.shape[:-1], 1)):
            raise ValueError("INT8 MoE LoRA dynamic_scale must contain one scale per activation row.")
    elif hidden_states.dtype not in (torch.bfloat16, torch.float16) or scale is not None:
        raise ValueError("MoE LoRA expects BF16/FP16 without scales or INT8 with per-token scales.")


@register_quant_moe_lora_impl(
    QuantType.W8A8,
    validate_activation_input=_validate_dynamic_int8_activations,
)
def _apply_dynamic_int8_moe_lora(
    mlp_compute_input: MoEMlpComputeInput,
    quant_method=None,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Run INT8 experts, with kernel-side LoRA dequantization on AllGather."""
    comm_type = _EXTRA_CTX.moe_comm_type
    if comm_type not in {MoECommType.ALLGATHER, MoECommType.ALLTOALL}:
        raise NotImplementedError(
            "Ascend quantized MoE LoRA currently supports the AllGather TP and AlltoAll EP paths; "
            "MC2 and FusedMC2 are unsupported."
        )
    lora_context = mlp_compute_input.lora_context
    if mlp_compute_input.dynamic_eplb:
        raise NotImplementedError("Ascend quantized MoE LoRA does not support dynamic EPLB.")

    hidden_states = mlp_compute_input.hidden_states
    allgather_int8 = comm_type == MoECommType.ALLGATHER
    if allgather_int8:
        _validate_int8_scale(hidden_states, mlp_compute_input.dynamic_scale)
    elif mlp_compute_input.dynamic_scale is not None or hidden_states.dtype == torch.int8:
        raise AssertionError(
            "Quantized MoE LoRA requires BF16/FP16 routed activations. "
            "Dispatch-side quantization must be disabled for LoRA batches."
        )
    if comm_type == MoECommType.ALLGATHER and (
        mlp_compute_input.expanded_row_idx is None or mlp_compute_input.topk_ids is None
    ):
        raise AssertionError("Quantized MoE LoRA requires AllGather routing metadata (expanded_row_idx and topk_ids).")
    input_dtype = (mlp_compute_input.output_dtype or hidden_states.dtype) if allgather_int8 else hidden_states.dtype
    if allgather_int8 and input_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("INT8 MoE LoRA requires an explicit BF16/FP16 output_dtype.")
    if hidden_states.shape[0] == 0:
        # An EP rank may receive no routed tokens. Keep participating in the
        # surrounding AlltoAll collectives, but avoid empty-tensor NPU kernels.
        if allgather_int8:
            reset_lora_indices(lora_context)
        return hidden_states.to(input_dtype), None

    # Weights are carried by the routed-expert layer since the MoE MLP refactor;
    # fall back to the payload for direct callers/tests that still build it.
    if quant_method is not None and mlp_compute_input.layer is not None:
        weights = quant_method.get_mlp_weights(mlp_compute_input.layer)
    else:
        weights = mlp_compute_input.weights
    if weights.w1_scale_bias is not None or weights.w2_scale_bias is not None:
        raise NotImplementedError("Quantized MoE LoRA does not support fused scale-bias.")
    if weights.w1_offset is not None or weights.w2_offset is not None:
        raise NotImplementedError("Quantized MoE LoRA does not support antiquant offsets.")
    if weights.w1_scale is None or weights.w2_scale is None:
        raise AssertionError("Quantized MoE LoRA requires w1 and w2 weight scales.")

    w1 = weights.w1 if isinstance(weights.w1, list) else [weights.w1]
    w2 = weights.w2 if isinstance(weights.w2, list) else [weights.w2]
    w1_scale = weights.w1_scale if isinstance(weights.w1_scale, list) else [weights.w1_scale]
    w2_scale = weights.w2_scale if isinstance(weights.w2_scale, list) else [weights.w2_scale]
    if not all(len(values) == 1 for values in (w1, w2, w1_scale, w2_scale)):
        raise NotImplementedError("Quantized MoE LoRA does not support per-expert tensor lists used by dynamic EPLB.")

    if hidden_states.dtype == torch.int8:
        quantized_input = hidden_states
        input_scale = mlp_compute_input.dynamic_scale.reshape(-1).contiguous()
    else:
        quantized_input, input_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=torch.int8,
            use_mxfp_quant=False,
        )
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[quantized_input],
        weight=w1,
        scale=[w1_scale[0].to(w2_scale[0].dtype)],
        per_token_scale=[input_scale],
        split_item=2,
        group_type=0,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
        output_dtype=input_dtype,
    )[0]

    if comm_type == MoECommType.ALLGATHER:
        lora_routing = _recover_moe_lora_routing_allgather(
            lora_context,
            mlp_compute_input.expanded_row_idx,
            mlp_compute_input.topk_ids,
            expert_start=mlp_compute_input.expert_start,
            num_local_experts=mlp_compute_input.num_local_experts,
        )
    else:
        lora_routing = _recover_moe_lora_routing_all2all(
            lora_context,
            group_list=mlp_compute_input.group_list,
        )
    act_name = getattr(mlp_compute_input.activation, "value", mlp_compute_input.activation)
    combined_lora_indices = None
    if (
        allgather_int8
        and mlp_compute_input.fusion
        and act_name == "silu"
        and _can_fuse_int8_lora_swiglu(lora_context, gate_up_out.shape[-1] // 2)
    ):
        quantized_activated, activated_scale, combined_lora_indices = moe_lora_apply_w13_swiglu_quant(
            lora_context,
            gate_up_out=gate_up_out,
            hidden_states=quantized_input,
            input_scale=input_scale,
            lora_routing=lora_routing,
            swiglu_limit=mlp_compute_input.swiglu_limit,
            topk_scales=mlp_compute_input.topk_scales,
        )
    else:
        moe_lora_apply_w13(
            lora_context,
            gate_up_out=gate_up_out,
            hidden_states=quantized_input if allgather_int8 else hidden_states,
            lora_routing=lora_routing,
            **({"input_scale": input_scale} if allgather_int8 else {}),
        )

        activated = _apply_moe_activation(
            gate_up_out,
            mlp_compute_input.activation,
            mlp_compute_input.swiglu_limit,
            mlp_compute_input.swiglu_alpha,
            mlp_compute_input.swiglu_beta,
        )
        if mlp_compute_input.topk_scales is not None:
            activated *= mlp_compute_input.topk_scales

        quantized_activated, activated_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=activated,
            dynamic_scale=None,
            act_quant_type=torch.int8,
            use_mxfp_quant=False,
        )
    before_gmm2_evt = torch.npu.current_stream().record_event()
    down_out = DeviceOperator.npu_grouped_matmul_gmm2(
        hidden_states=quantized_activated,
        weight=w2,
        weight_scale=w2_scale,
        per_token_scale=activated_scale,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
        input_dtype=input_dtype,
        act_quant_type=torch.int8,
        weight_quant_type=None,
        scale_type=None,
        per_token_scale_type=None,
        use_bf16=input_dtype == torch.bfloat16,
        use_mxfp_quant=False,
        bias=None,
        fallback_output_dtype=input_dtype if allgather_int8 else w2_scale[0].dtype,
        mxfp_quant_dtype=None,
    )
    moe_lora_apply_w2(
        lora_context,
        down_out=down_out,
        silu_out=quantized_activated if allgather_int8 else activated,
        lora_routing=lora_routing,
        **({"input_scale": activated_scale} if allgather_int8 else {}),
        **({"combined_indices": combined_lora_indices} if combined_lora_indices is not None else {}),
    )
    return down_out, before_gmm2_evt


__all__ = [
    "quant_apply_mlp_with_moe_lora",
    "register_quant_moe_lora_impl",
    "validate_quant_moe_lora_activation_input",
]
