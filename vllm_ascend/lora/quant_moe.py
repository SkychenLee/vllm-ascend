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

Each quantization scheme registers the dispatch policy and MLP implementation
needed to preserve floating-point LoRA boundaries around quantized base expert
matmuls. The token dispatcher and the common MoE MLP entry therefore do not
need scheme-specific dtype checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.fused_moe import (
    _recover_moe_lora_routing_all2all,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend import envs

QuantMoELoRAApply = Callable[[MoEMlpComputeInput], tuple[torch.Tensor, torch.npu.Event | None]]
QuantMoELoRAActivationValidator = Callable[[torch.Tensor, torch.Tensor | None], None]

# ---------------------------------------------------------------------------
# Aux-stream overlap (AllGather + decode): LoRA applies are vector-only work
# and the grouped matmuls keep the cube cores busy, so the applies (and the
# combined-index metadata build) are hidden behind the GEMMs on one shared
# aux stream. Two variants, picked by _lora_overlap_eligible:
#   "delta" (non-fully-sharded / EP): the single-op applies run entirely on
#     the aux stream into isolated, per-shape cached delta buffers; the main
#     stream folds each delta into the GEMM output with a single add.
#   "phase" (fully-sharded / TP-only): the two-op shrink/expand structure
#     splits at the dependency boundary -- only the shrink halves (plus the
#     metadata build) run on the aux stream; the cross-rank collective and
#     the expands stay on the main stream (w2 needs no collective at all:
#     partial_expand folds its completion into the MoE runner's final TP
#     all-reduce).
# ---------------------------------------------------------------------------
_AUX_STREAM: torch.npu.Stream | None = None
_DELTA_BUFS: dict[tuple, torch.Tensor] = {}


def _lora_aux_stream() -> torch.npu.Stream:
    global _AUX_STREAM
    if _AUX_STREAM is None:
        _AUX_STREAM = torch.npu.Stream()
    return _AUX_STREAM


def _lora_delta_buffer(rows, width, dtype, device) -> torch.Tensor:
    key = (rows, width, dtype, device)
    buf = _DELTA_BUFS.get(key)
    if buf is None:
        buf = torch.empty((rows, width), dtype=dtype, device=device)
        _DELTA_BUFS[key] = buf
    return buf


def _lora_overlap_eligible(lora_context, comm_type) -> str | None:
    """Pick the aux-stream overlap variant for this batch, or None."""
    if not envs.VLLM_ASCEND_LORA_MOE_OVERLAP or comm_type != MoECommType.ALLGATHER:
        return None
    wrapper = getattr(lora_context, "punica_wrapper", None)
    if wrapper is None:
        return None
    use_gmm = getattr(wrapper, "_use_moe_gmm_cpu", None)
    if use_gmm is None or bool(use_gmm.item()):
        return None  # decode only; prefill keeps the sequential gmm path
    if getattr(lora_context, "fully_sharded", False):
        # FS+EP is invalid upstream; FS under TP runs the two-op path whose
        # collective must sit between shrink and expand -> phase variant
        if getattr(lora_context, "use_ep", False):
            return None
        return "phase"
    return "delta"


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
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Dispatch an active quantized MoE LoRA batch to its backend."""
    return _get_quant_moe_lora_impl(mlp_compute_input.quant.quant_type).apply(mlp_compute_input)


def validate_quant_moe_lora_activation_input(
    *,
    quant_type: QuantType,
    hidden_states: torch.Tensor,
    dynamic_scale: torch.Tensor | None,
) -> None:
    """Validate activations before quantized MoE LoRA prepare/dispatch."""
    impl = _get_quant_moe_lora_impl(quant_type)
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


def _fused_w13_epilogue_eligible(mlp_compute_input, lora_context) -> bool:
    """Can add_lora_swiglu_quant replace swiglu -> dynamic_quant for this batch?"""
    if mlp_compute_input.topk_scales is not None:
        return False
    activation = mlp_compute_input.activation
    act_name = getattr(activation, "value", activation)
    if activation in (
        MoEActivation.SWIGLUOAI,
        MoEActivation.SWIGLUSTEP,
        MoEActivation.GELU,
        MoEActivation.GELU_TANH,
    ):
        return False
    if act_name == "swigluoai_uninterleave":
        return False
    return True


def _validate_dynamic_int8_activations(
    hidden_states: torch.Tensor,
    dynamic_scale: torch.Tensor | None,
) -> None:
    if dynamic_scale is not None or hidden_states.dtype == torch.int8:
        raise NotImplementedError("Dynamic INT8 MoE LoRA requires unquantized activations before expert routing.")


@register_quant_moe_lora_impl(
    QuantType.W8A8,
    validate_activation_input=_validate_dynamic_int8_activations,
)
def _apply_dynamic_int8_moe_lora(
    mlp_compute_input: MoEMlpComputeInput,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Run INT8 base experts and inject LoRA at BF16/FP16 boundaries."""
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
    if mlp_compute_input.dynamic_scale is not None or hidden_states.dtype == torch.int8:
        raise AssertionError(
            "Quantized MoE LoRA requires BF16/FP16 routed activations. "
            "Dispatch-side quantization must be disabled for LoRA batches."
        )
    if comm_type == MoECommType.ALLGATHER and (
        mlp_compute_input.expanded_row_idx is None or mlp_compute_input.topk_ids is None
    ):
        raise AssertionError("Quantized MoE LoRA requires AllGather routing metadata (expanded_row_idx and topk_ids).")
    if hidden_states.shape[0] == 0:
        # An EP rank may receive no routed tokens. Keep participating in the
        # surrounding AlltoAll collectives, but avoid empty-tensor NPU kernels.
        return hidden_states, None

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

    input_dtype = hidden_states.dtype

    overlap_variant = _lora_overlap_eligible(lora_context, comm_type)
    if overlap_variant is not None:
        apply_overlap = (
            _apply_dynamic_int8_moe_lora_overlap_phase
            if overlap_variant == "phase"
            else _apply_dynamic_int8_moe_lora_overlap_delta
        )
        return apply_overlap(
            mlp_compute_input=mlp_compute_input,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale[0],
            w2_scale=w2_scale[0],
        )

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
        # Single-pass combined index (replaces the recovery + build chain;
        # see _build_combined_lora_idx_allgather). Stashed for w13/w2.
        from vllm_ascend.lora.fused_moe import _build_combined_lora_idx_allgather

        lora_context.combined_lora_idx = _build_combined_lora_idx_allgather(
            lora_context,
            mlp_compute_input.expanded_row_idx,
            mlp_compute_input.topk_ids,
        )
        lora_routing = None
    else:
        lora_routing = _recover_moe_lora_routing_all2all(
            lora_context,
            group_list=mlp_compute_input.group_list,
        )
    delta = moe_lora_apply_w13(
        lora_context,
        gate_up_out=gate_up_out,
        hidden_states=hidden_states,
        lora_routing=lora_routing,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
    )

    if _fused_w13_epilogue_eligible(mlp_compute_input, lora_context):
        activated, quantized_activated, activated_scale = (
            torch.ops._C_ascend.add_lora_swiglu_quant(
                gate_up_out, delta, mlp_compute_input.swiglu_limit
            )
        )
    else:
        if delta is not None:
            gate_up_out += delta
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
        fallback_output_dtype=w2_scale[0].dtype,
        mxfp_quant_dtype=None,
    )
    moe_lora_apply_w2(
        lora_context,
        down_out=down_out,
        silu_out=activated,
        lora_routing=lora_routing,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
    )
    return down_out, before_gmm2_evt


def _apply_dynamic_int8_moe_lora_overlap_delta(
    *,
    mlp_compute_input: MoEMlpComputeInput,
    hidden_states: torch.Tensor,
    w1,
    w2,
    w1_scale,
    w2_scale,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """AllGather decode path with the LoRA work hidden behind the base GEMMs.

    Layout per layer (main stream M, aux stream B):
      M: ev_x.record                       (hidden + routing ready)
      B: wait ev_x -> build combined_idx -> delta13.zero_ -> w13 apply
         (overwrite mode) -> ev13.record
      M: quant1 -> gmm1 -> wait ev13 -> gate_up_out += delta13
      M: activation (-> topk scale) -> ev_act.record
      B: wait ev_act -> delta2.zero_ -> w2 apply -> ev2.record
      M: quant2 -> gmm2 -> wait ev2 -> down_out += delta2
    When the fused w13 epilogue is eligible, the += / activation / quant2
    trio on M collapses into one add_lora_swiglu_quant(gate_up_out, delta13)
    before ev_act (same kernel and gate as the sequential path).
    The applies and the metadata build only touch tensors produced before
    their GEMM, so B never waits on M mid-flight.
    """
    from vllm_ascend.lora.fused_moe import _build_combined_lora_idx_allgather

    lora_context = mlp_compute_input.lora_context
    wrapper = lora_context.punica_wrapper
    aux = _lora_aux_stream()
    main = torch.npu.current_stream()
    rows = hidden_states.shape[0]
    device = hidden_states.device
    input_dtype = hidden_states.dtype

    def _apply(y_delta: torch.Tensor, x: torch.Tensor, a_stacked, b_stacked) -> None:
        wrapper.add_lora_fused_moe(
            y=y_delta,
            x=x,
            lora_a_stacked=a_stacked,
            lora_b_stacked=b_stacked,
            adapter_enabled=lora_context.adapter_enabled,
            fully_sharded=False,
            combined_idx=lora_context.combined_lora_idx,
            overwrite=True,
        )

    ev_x = torch.npu.Event()
    ev_x.record(main)
    with torch.npu.stream(aux):
        aux.wait_event(ev_x)
        lora_context.combined_lora_idx = _build_combined_lora_idx_allgather(
            lora_context,
            mlp_compute_input.expanded_row_idx,
            mlp_compute_input.topk_ids,
        )
        width13 = sum(b.shape[-2] for b in lora_context.w13_lora_b_stacked)
        delta13 = _lora_delta_buffer(rows, width13, input_dtype, device)
        delta13.zero_()
        _apply(delta13, hidden_states, lora_context.w13_lora_a_stacked,
               lora_context.w13_lora_b_stacked)
        ev13 = torch.npu.Event()
        ev13.record(aux)

    quantized_input, input_scale = DeviceOperator.npu_dynamic_quant(
        hidden_states=hidden_states,
        dynamic_scale=None,
        act_quant_type=torch.int8,
        use_mxfp_quant=False,
    )
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[quantized_input],
        weight=w1,
        scale=[w1_scale.to(w2_scale.dtype)],
        per_token_scale=[input_scale],
        split_item=2,
        group_type=0,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
        output_dtype=input_dtype,
    )[0]
    main.wait_event(ev13)
    fuse_epilogue = _fused_w13_epilogue_eligible(
        mlp_compute_input, lora_context
    ) and delta13.shape == gate_up_out.shape
    if fuse_epilogue:
        activated, quantized_activated, activated_scale = (
            torch.ops._C_ascend.add_lora_swiglu_quant(
                gate_up_out, delta13, mlp_compute_input.swiglu_limit
            )
        )
    else:
        gate_up_out += delta13
        activated = _apply_moe_activation(
            gate_up_out,
            mlp_compute_input.activation,
            mlp_compute_input.swiglu_limit,
            mlp_compute_input.swiglu_alpha,
            mlp_compute_input.swiglu_beta,
        )
        if mlp_compute_input.topk_scales is not None:
            activated *= mlp_compute_input.topk_scales

    ev_act = torch.npu.Event()
    ev_act.record(main)
    with torch.npu.stream(aux):
        aux.wait_event(ev_act)
        width2 = sum(b.shape[-2] for b in lora_context.w2_lora_b_stacked)
        delta2 = _lora_delta_buffer(rows, width2, input_dtype, device)
        delta2.zero_()
        _apply(delta2, activated, lora_context.w2_lora_a_stacked,
               lora_context.w2_lora_b_stacked)
        ev2 = torch.npu.Event()
        ev2.record(aux)

    if not fuse_epilogue:
        quantized_activated, activated_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=activated,
            dynamic_scale=None,
            act_quant_type=torch.int8,
            use_mxfp_quant=False,
        )
    before_gmm2_evt = main.record_event()
    down_out = DeviceOperator.npu_grouped_matmul_gmm2(
        hidden_states=quantized_activated,
        weight=w2,
        weight_scale=[w2_scale],
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
        fallback_output_dtype=w2_scale.dtype,
        mxfp_quant_dtype=None,
    )
    main.wait_event(ev2)
    down_out += delta2
    if hasattr(lora_context, "combined_lora_idx"):
        del lora_context.combined_lora_idx
    return down_out, before_gmm2_evt


def _apply_dynamic_int8_moe_lora_overlap_phase(
    *,
    mlp_compute_input: MoEMlpComputeInput,
    hidden_states: torch.Tensor,
    w1,
    w2,
    w1_scale,
    w2_scale,
) -> tuple[torch.Tensor, torch.npu.Event | None]:
    """Fully-sharded TP-only variant. Layout per layer (main M, aux B):
      M: ev_x.record                        (hidden + routing ready)
      B: wait ev_x -> combined_idx build -> w13 shrink -> ev13
      M: quant1 -> gmm1 -> wait ev13 -> w13 collective + expand -> gate_up_out
      M: activation (-> topk scale) -> ev_act
      B: wait ev_act -> w2 shrink (collective-free via partial_expand) -> ev2
      M: quant2 -> gmm2 -> wait ev2 -> w2 expand into down_out slice
    """
    from vllm_ascend.lora.fused_moe import _build_combined_lora_idx_allgather

    lora_context = mlp_compute_input.lora_context
    wrapper = lora_context.punica_wrapper
    aux = _lora_aux_stream()
    main = torch.npu.current_stream()
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    input_dtype = hidden_states.dtype

    ev_x = torch.npu.Event()
    ev_x.record(main)
    with torch.npu.stream(aux):
        aux.wait_event(ev_x)
        combined_idx = _build_combined_lora_idx_allgather(
            lora_context,
            mlp_compute_input.expanded_row_idx,
            mlp_compute_input.topk_ids,
        )
        buf13 = wrapper.add_lora_fused_moe(
            y=hidden_states,  # unused by phase="shrink"
            x=hidden_states,
            lora_a_stacked=lora_context.w13_lora_a_stacked,
            lora_b_stacked=lora_context.w13_lora_b_stacked,
            adapter_enabled=lora_context.adapter_enabled,
            fully_sharded=True,
            group_list=group_list,
            group_list_type=group_list_type,
            combined_idx=combined_idx,
            phase="shrink",
        )
        ev13 = torch.npu.Event()
        ev13.record(aux)

    quantized_input, input_scale = DeviceOperator.npu_dynamic_quant(
        hidden_states=hidden_states,
        dynamic_scale=None,
        act_quant_type=torch.int8,
        use_mxfp_quant=False,
    )
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[quantized_input],
        weight=w1,
        scale=[w1_scale.to(w2_scale.dtype)],
        per_token_scale=[input_scale],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=group_list_type,
        output_dtype=input_dtype,
    )[0]
    main.wait_event(ev13)
    wrapper.add_lora_fused_moe(
        y=gate_up_out,
        x=hidden_states,  # unused by phase="expand"
        lora_a_stacked=lora_context.w13_lora_a_stacked,
        lora_b_stacked=lora_context.w13_lora_b_stacked,
        adapter_enabled=lora_context.adapter_enabled,
        fully_sharded=True,
        group_list=group_list,
        group_list_type=group_list_type,
        combined_idx=combined_idx,
        phase="expand",
        shrink_buffers=buf13,
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

    # mirror moe_lora_apply_w2: fold w2's cross-rank completion into the MoE
    # runner's final TP all-reduce (partial_expand), expand into this rank's
    # output slice
    partial_expand = (
        lora_context.fully_sharded
        and getattr(lora_context, "tp_size", 1) > 1
        and not getattr(lora_context, "use_ep", False)
    )
    w2_offset = lora_context.w2_lora_b_stacked[0].shape[-2] * lora_context.tp_rank

    ev_act = torch.npu.Event()
    ev_act.record(main)
    with torch.npu.stream(aux):
        aux.wait_event(ev_act)
        buf2 = wrapper.add_lora_fused_moe(
            y=activated,  # unused by phase="shrink"
            x=activated,
            lora_a_stacked=lora_context.w2_lora_a_stacked,
            lora_b_stacked=lora_context.w2_lora_b_stacked,
            adapter_enabled=lora_context.adapter_enabled,
            fully_sharded=True,
            partial_expand=partial_expand,
            offset=w2_offset,
            group_list=group_list,
            group_list_type=group_list_type,
            combined_idx=combined_idx,
            phase="shrink",
        )
        ev2 = torch.npu.Event()
        ev2.record(aux)

    quantized_activated, activated_scale = DeviceOperator.npu_dynamic_quant(
        hidden_states=activated,
        dynamic_scale=None,
        act_quant_type=torch.int8,
        use_mxfp_quant=False,
    )
    before_gmm2_evt = main.record_event()
    down_out = DeviceOperator.npu_grouped_matmul_gmm2(
        hidden_states=quantized_activated,
        weight=w2,
        weight_scale=[w2_scale],
        per_token_scale=activated_scale,
        group_list=group_list,
        group_list_type=group_list_type,
        input_dtype=input_dtype,
        act_quant_type=torch.int8,
        weight_quant_type=None,
        scale_type=None,
        per_token_scale_type=None,
        use_bf16=input_dtype == torch.bfloat16,
        use_mxfp_quant=False,
        bias=None,
        fallback_output_dtype=w2_scale.dtype,
        mxfp_quant_dtype=None,
    )
    main.wait_event(ev2)
    wrapper.add_lora_fused_moe(
        y=down_out,
        x=activated,  # unused by phase="expand"
        lora_a_stacked=lora_context.w2_lora_a_stacked,
        lora_b_stacked=lora_context.w2_lora_b_stacked,
        adapter_enabled=lora_context.adapter_enabled,
        fully_sharded=True,
        partial_expand=partial_expand,
        offset=w2_offset,
        group_list=group_list,
        group_list_type=group_list_type,
        combined_idx=combined_idx,
        phase="expand",
        shrink_buffers=buf2,
    )
    return down_out, before_gmm2_evt


__all__ = [
    "quant_apply_mlp_with_moe_lora",
    "register_quant_moe_lora_impl",
    "validate_quant_moe_lora_activation_input",
]
