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
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.fused_moe import (
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    _recover_moe_lora_routing_from_slots,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
    reset_lora_indices,
)
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import npu_stream_switch

QuantMoELoRAApply = Callable[[MoEMlpComputeInput], tuple[torch.Tensor, torch.npu.Event | None]]
QuantMoELoRAActivationValidator = Callable[[torch.Tensor, torch.Tensor | None], None]


@dataclass(frozen=True)
class QuantMoELoRAImpl:
    apply: QuantMoELoRAApply
    validate_activation_input: QuantMoELoRAActivationValidator | None


@dataclass(frozen=True)
class _CompositeLoraGMMRouting:
    group_ids: torch.Tensor
    group_list: torch.Tensor
    enabled: torch.Tensor


@dataclass(frozen=True)
class _SingleLoraGMMRouting:
    enabled: torch.Tensor


_QUANT_MOE_LORA_IMPLS: dict[QuantType, QuantMoELoRAImpl] = {}

MOE_LORA_GMM_MIN_ROWS_PER_GROUP = 8

# Temporary dual-stream benchmark isolation: force the single-adapter
# expert-grouped GMM fast path off at both the AllGather token-dispatcher and
# the MLP compute call sites so A/B runs measure dual-stream overlap on the
# regular LoRA path only. The eligibility helpers keep their original
# semantics so the fast-path unit tests stay valid.
MOE_LORA_SINGLE_GMM_FAST_PATH_ENABLED = False


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


def _validate_dynamic_int8_activations(
    hidden_states: torch.Tensor,
    dynamic_scale: torch.Tensor | None,
) -> None:
    if dynamic_scale is not None or hidden_states.dtype == torch.int8:
        raise NotImplementedError("Dynamic INT8 MoE LoRA requires unquantized activations before expert routing.")


def _can_use_ep_moe_lora_aux_stream(
    lora_context,
    comm_type: MoECommType,
    *,
    is_decode_only: bool,
) -> bool:
    """Return whether this EP invocation can overlap base GMM and LoRA."""
    if not is_decode_only:
        return False
    if comm_type not in {MoECommType.ALLGATHER, MoECommType.ALLTOALL}:
        return False
    if not getattr(lora_context, "use_ep", False) or getattr(lora_context, "fully_sharded", False):
        return False
    aux_stream = getattr(lora_context, "aux_stream", None)
    events = getattr(lora_context, "events", None)
    return aux_stream is not None and events is not None and len(events) >= 4


def _execute_moe_lora_in_parallel(
    base_fn: Callable[[], torch.Tensor],
    lora_fn: Callable[[], None],
    start_event: torch.npu.Event,
    done_event: torch.npu.Event,
    aux_stream: torch.npu.Stream,
    aux_prepare_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Fork LoRA onto an auxiliary stream and join before accumulation.

    When provided, ``aux_prepare_fn`` is enqueued on the auxiliary stream
    before the base callback is submitted. The LoRA callback remains ordered
    after preparation on that stream without making the base wait for it.
    """
    # ACLGraph capture can temporarily switch away from the process default
    # stream. Query the runtime here so the fork/join is recorded on the
    # actual caller stream instead of a cached default-stream handle.
    main_stream = torch.npu.current_stream()
    start_event.record(main_stream)
    if aux_prepare_fn is not None:
        with npu_stream_switch(aux_stream):
            aux_stream.wait_event(start_event)
            aux_prepare_fn()
    base_result = base_fn()
    with npu_stream_switch(aux_stream):
        if aux_prepare_fn is None:
            aux_stream.wait_event(start_event)
        lora_fn()
        done_event.record(aux_stream)
    main_stream.wait_event(done_event)
    return base_result


def _lora_output_size(lora_b_stacked: tuple[torch.Tensor, ...]) -> int:
    return sum(weight.shape[-2] for weight in lora_b_stacked)


def _new_lora_delta_workspace(
    inputs: torch.Tensor,
    w13_lora_b_stacked: tuple[torch.Tensor, ...],
    w2_lora_b_stacked: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, int, int]:
    """Allocate one delta buffer reused after the W13 stream join for W2."""
    w13_output_size = _lora_output_size(w13_lora_b_stacked)
    w2_output_size = _lora_output_size(w2_lora_b_stacked)
    workspace = inputs.new_empty((*inputs.shape[:-1], max(w13_output_size, w2_output_size)))
    return workspace, w13_output_size, w2_output_size


def _can_prepare_single_lora_gmm(
    lora_context,
    *,
    num_routed_rows: int,
    num_experts: int,
    group_list_type: int,
    expert_map: torch.Tensor | None = None,
) -> bool:
    """Return whether AllGather should route single-LoRA metadata."""
    use_ep = getattr(lora_context, "use_ep", False)
    if use_ep and (expert_map is None or getattr(lora_context, "fully_sharded", False)):
        return False
    punica_wrapper = getattr(lora_context, "punica_wrapper", None)
    if punica_wrapper is None:
        return False
    if (
        getattr(punica_wrapper, "num_active_moe_loras", 0) != 1
        or getattr(punica_wrapper, "active_moe_lora_slot", None) is None
        or getattr(lora_context, "single_lora_cache_slot", None) != punica_wrapper.active_moe_lora_slot
        or group_list_type != 1
    ):
        return False

    # The single-adapter cache is created from the validated LoRA
    # configuration. Avoid rechecking its existence, dtype, and rank layout in
    # this per-forward eligibility path.
    weights = (
        *lora_context.w13_lora_a_packed,
        *lora_context.w13_lora_b_packed,
        *lora_context.w2_lora_a_packed,
        *lora_context.w2_lora_b_packed,
    )
    # EP keeps a full, statically shaped dispatched buffer while each rank's
    # GMM group list only covers its local experts. Use the global expert-map
    # size for the minimum-work heuristic so EP does not take this fast path at
    # much smaller batches than the equivalent non-EP execution.
    min_rows_num_experts = max(expert_map.numel(), num_experts) if use_ep else num_experts
    return not (
        any(weight.shape[0] != num_experts for weight in weights)
        or num_routed_rows < MOE_LORA_GMM_MIN_ROWS_PER_GROUP * min_rows_num_experts
    )


def _can_use_single_lora_gmm(
    lora_context,
    *,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    expert_map: torch.Tensor | None = None,
    routed_lora_slots: torch.Tensor | None = None,
) -> bool:
    """Return whether one active adapter can use expert-grouped GMM."""
    if routed_lora_slots is None or routed_lora_slots.numel() != hidden_states.shape[0]:
        return False
    return _can_prepare_single_lora_gmm(
        lora_context,
        num_routed_rows=hidden_states.shape[0],
        num_experts=group_list.numel(),
        group_list_type=group_list_type,
        expert_map=expert_map,
    )


def _can_prepare_composite_lora_gmm(
    lora_context,
    *,
    hidden_dtype: torch.dtype,
    num_routed_rows: int,
    num_experts: int,
    group_list_type: int,
    expert_map: torch.Tensor | None = None,
) -> bool:
    """Return whether AllGather should route composite-LoRA metadata."""
    use_ep = getattr(lora_context, "use_ep", False)
    if use_ep and (expert_map is None or getattr(lora_context, "fully_sharded", False)):
        return False
    punica_wrapper = getattr(lora_context, "punica_wrapper", None)
    if (
        punica_wrapper is None
        or getattr(punica_wrapper, "no_lora", True)
        or getattr(punica_wrapper, "num_active_moe_loras", 0) < 2
        or group_list_type != 1
        or hidden_dtype != torch.bfloat16
    ):
        return False

    w13_a = lora_context.w13_lora_a_stacked
    w13_b = lora_context.w13_lora_b_stacked
    w2_a = lora_context.w2_lora_a_stacked
    w2_b = lora_context.w2_lora_b_stacked
    weights = (*w13_a, *w13_b, *w2_a, *w2_b)
    max_loras = lora_context.max_loras
    min_rows_num_experts = max(expert_map.numel(), num_experts) if use_ep else num_experts
    num_composite_groups = max_loras * min_rows_num_experts
    return not (
        not weights
        or any(weight.shape[0] != max_loras for weight in weights)
        or any(weight.shape[1] != num_experts for weight in weights)
        or any(weight.dtype != hidden_dtype for weight in weights)
        or num_routed_rows < MOE_LORA_GMM_MIN_ROWS_PER_GROUP * num_composite_groups
    )


def _can_use_composite_lora_gmm(
    lora_context,
    *,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    expert_map: torch.Tensor | None = None,
    routed_lora_slots: torch.Tensor | None = None,
) -> bool:
    """Return whether mixed requests can use ``(slot, expert)`` GMM."""
    if routed_lora_slots is None or routed_lora_slots.numel() != hidden_states.shape[0]:
        return False
    return _can_prepare_composite_lora_gmm(
        lora_context,
        hidden_dtype=hidden_states.dtype,
        num_routed_rows=hidden_states.shape[0],
        num_experts=group_list.numel(),
        group_list_type=group_list_type,
        expert_map=expert_map,
    )


def _build_single_lora_gmm_routing(
    *,
    routed_lora_slots: torch.Tensor,
) -> _SingleLoraGMMRouting:
    """Build graph-safe single-adapter routing from init-routing metadata."""
    return _SingleLoraGMMRouting(enabled=routed_lora_slots >= 0)


def _build_composite_lora_gmm_routing(
    lora_context,
    *,
    routed_lora_slots: torch.Tensor,
    group_list: torch.Tensor,
) -> _CompositeLoraGMMRouting:
    """Build graph-safe ``(LoRA slot, local expert)`` routing."""
    num_experts = group_list.numel()
    max_loras = lora_context.w13_lora_a_stacked[0].shape[0]
    num_groups = max_loras * num_experts
    row_ids = torch.arange(
        routed_lora_slots.shape[0],
        dtype=group_list.dtype,
        device=routed_lora_slots.device,
    )
    valid_rows = row_ids < group_list.sum()
    expert_ids = torch.searchsorted(
        group_list.cumsum(0),
        row_ids,
        right=True,
    ).clamp_(max=num_experts - 1)

    safe_lora_slots = routed_lora_slots.clamp(min=0, max=max_loras - 1)
    adapter_enabled = lora_context.adapter_enabled
    enabled = valid_rows & (routed_lora_slots >= 0) & adapter_enabled[safe_lora_slots].bool()

    # Base, inactive-adapter, and non-local EP rows use a sentinel group so
    # token-permute moves them after every GMM group. Excluding them from the
    # group counts avoids both unnecessary LoRA matmuls and unloaded weights.
    active_group_ids = safe_lora_slots * num_experts + expert_ids
    group_ids = torch.where(enabled, active_group_ids, num_groups).to(torch.int32)
    composite_group_list = torch.zeros(
        num_groups,
        dtype=torch.int64,
        device=routed_lora_slots.device,
    )
    composite_group_list.scatter_add_(
        0,
        active_group_ids,
        enabled.to(torch.int64),
    )
    return _CompositeLoraGMMRouting(
        group_ids=group_ids,
        group_list=composite_group_list,
        enabled=enabled,
    )


def _grouped_lora_matmul(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    """Apply one BF16 expert-grouped LoRA projection.

    LoRA factors are stored for BGMV as ``[E, N, K]``.  GMM consumes
    ``[E, K, N]``; the transpose is a view, matching the existing unquantized
    MoE GMM path and avoiding a full weight copy on every forward.
    """
    return torch_npu.npu_grouped_matmul(
        x=[inputs],
        weight=[weight.transpose(-1, -2)],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=1,
    )[0]


def _communicate_fully_sharded_lora(
    shrink: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    fully_sharded: bool,
) -> torch.Tensor:
    """Assemble the LoRA rank dimension before applying the B projection."""
    if fully_sharded:
        local_rank = lora_a.shape[-2]
        full_rank = lora_b.shape[-1]
        if local_rank == full_rank:
            shrink = tensor_model_parallel_all_reduce(shrink)
        else:
            shrink = tensor_model_parallel_all_gather(shrink)

    if shrink.shape[-1] != lora_b.shape[-1]:
        raise ValueError(
            "MoE LoRA rank mismatch after TP communication: "
            f"A projection has rank {shrink.shape[-1]}, "
            f"but LoRA B expects rank {lora_b.shape[-1]}."
        )
    return shrink


def _add_single_lora_gmm(
    output: torch.Tensor,
    inputs: torch.Tensor,
    lora_a_stacked: tuple[torch.Tensor, ...],
    lora_b_stacked: tuple[torch.Tensor, ...],
    *,
    routing: _SingleLoraGMMRouting,
    group_list: torch.Tensor,
    fully_sharded: bool = False,
    output_offset: int = 0,
) -> None:
    """Add one adapter with two grouped matmuls per output slice.

    Packed weights have stable addresses and contain the host-known active
    adapter. The optional base-token mask remains a graph tensor.
    """
    masked_inputs = inputs * routing.enabled.unsqueeze(-1).to(inputs.dtype)
    current_output_offset = output_offset
    for lora_a, lora_b in zip(lora_a_stacked, lora_b_stacked, strict=True):
        shrink = _grouped_lora_matmul(
            masked_inputs,
            lora_a,
            group_list,
        )
        shrink = _communicate_fully_sharded_lora(
            shrink,
            lora_a,
            lora_b,
            fully_sharded=fully_sharded,
        )
        delta = _grouped_lora_matmul(
            shrink,
            lora_b,
            group_list,
        )
        output_size = delta.shape[-1]
        output.narrow(-1, current_output_offset, output_size).add_(delta)
        current_output_offset += output_size


def _add_composite_lora_gmm(
    output: torch.Tensor,
    inputs: torch.Tensor,
    lora_a_stacked: tuple[torch.Tensor, ...],
    lora_b_stacked: tuple[torch.Tensor, ...],
    *,
    routing: _CompositeLoraGMMRouting,
    fully_sharded: bool = False,
    output_offset: int = 0,
) -> None:
    """Add LoRA deltas with MoE-native token permutation."""
    grouped_inputs, reverse_mapping = torch_npu.npu_moe_token_permute(
        tokens=inputs,
        indices=routing.group_ids,
        num_out_tokens=inputs.shape[0],
    )

    current_output_offset = output_offset
    for lora_a, lora_b in zip(lora_a_stacked, lora_b_stacked, strict=True):
        lora_a_flat = lora_a.flatten(0, 1)
        lora_b_flat = lora_b.flatten(0, 1)
        shrink = _grouped_lora_matmul(grouped_inputs, lora_a_flat, routing.group_list)
        shrink = _communicate_fully_sharded_lora(
            shrink,
            lora_a_flat,
            lora_b_flat,
            fully_sharded=fully_sharded,
        )
        delta = _grouped_lora_matmul(shrink, lora_b_flat, routing.group_list)
        delta = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=delta,
            sorted_indices=reverse_mapping,
        )
        delta.masked_fill_(~routing.enabled.unsqueeze(-1), 0)
        output_size = delta.shape[-1]
        output.narrow(-1, current_output_offset, output_size).add_(delta)
        current_output_offset += output_size


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
            "Ascend quantized MoE LoRA currently supports AllGather TP/EP and AlltoAll EP paths; "
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

    use_single_lora_gmm = False
    use_composite_lora_gmm = False
    if comm_type == MoECommType.ALLGATHER:
        use_single_lora_gmm = MOE_LORA_SINGLE_GMM_FAST_PATH_ENABLED and _can_use_single_lora_gmm(
            lora_context,
            hidden_states=hidden_states,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            expert_map=mlp_compute_input.expert_map,
            routed_lora_slots=mlp_compute_input.routed_lora_slots,
        )
        if not use_single_lora_gmm:
            use_composite_lora_gmm = _can_use_composite_lora_gmm(
                lora_context,
                hidden_states=hidden_states,
                group_list=mlp_compute_input.group_list,
                group_list_type=mlp_compute_input.group_list_type,
                expert_map=mlp_compute_input.expert_map,
                routed_lora_slots=mlp_compute_input.routed_lora_slots,
            )

    use_aux_stream = _can_use_ep_moe_lora_aux_stream(
        lora_context,
        comm_type,
        is_decode_only=_EXTRA_CTX.is_decode_only is True,
    )
    defer_allgather_lora_routing = use_aux_stream and comm_type == MoECommType.ALLGATHER
    lora_routing = None
    single_lora_routing = None
    composite_lora_routing = None

    def prepare_lora_routing() -> None:
        nonlocal lora_routing, single_lora_routing, composite_lora_routing
        if use_single_lora_gmm:
            assert mlp_compute_input.routed_lora_slots is not None
            single_lora_routing = _build_single_lora_gmm_routing(
                routed_lora_slots=mlp_compute_input.routed_lora_slots,
            )
        elif use_composite_lora_gmm:
            assert mlp_compute_input.routed_lora_slots is not None
            composite_lora_routing = _build_composite_lora_gmm_routing(
                lora_context,
                routed_lora_slots=mlp_compute_input.routed_lora_slots,
                group_list=mlp_compute_input.group_list,
            )
        elif (
            comm_type == MoECommType.ALLGATHER
            and mlp_compute_input.routed_lora_slots is not None
            and mlp_compute_input.routed_lora_slots.numel() == hidden_states.shape[0]
        ):
            # The dispatcher routed the per-token LoRA slot through the
            # init-routing scale sideband (decode-only dual-stream regular
            # path, or a fast GMM path whose compute-side eligibility did not
            # hold). Rebuild routing from the expert-major sideband with
            # static-shape ops instead of the scatter-based recovery.
            lora_routing = _recover_moe_lora_routing_from_slots(
                mlp_compute_input.routed_lora_slots,
                mlp_compute_input.group_list,
            )
        elif comm_type == MoECommType.ALLGATHER:
            lora_routing = _recover_moe_lora_routing_allgather(
                lora_context,
                mlp_compute_input.expanded_row_idx,
                mlp_compute_input.topk_ids,
                expert_map=mlp_compute_input.expert_map,
            )
        else:
            lora_routing = _recover_moe_lora_routing_all2all(
                lora_context,
                group_list=mlp_compute_input.group_list,
            )

    # AlltoAll routing consumes exchanged indices and keeps its existing
    # communication ordering. AllGather routing is LoRA-only, so pure-decode
    # dual-stream execution can construct it on the auxiliary stream while the
    # main stream runs the base W13 GMM.
    if not defer_allgather_lora_routing:
        prepare_lora_routing()

    def apply_w13_lora(output: torch.Tensor) -> None:
        if use_single_lora_gmm:
            assert single_lora_routing is not None
            _add_single_lora_gmm(
                output,
                hidden_states,
                lora_context.w13_lora_a_packed,
                lora_context.w13_lora_b_packed,
                routing=single_lora_routing,
                group_list=mlp_compute_input.group_list,
                fully_sharded=lora_context.fully_sharded,
            )
        elif use_composite_lora_gmm:
            assert composite_lora_routing is not None
            _add_composite_lora_gmm(
                output,
                hidden_states,
                lora_context.w13_lora_a_stacked,
                lora_context.w13_lora_b_stacked,
                routing=composite_lora_routing,
                fully_sharded=lora_context.fully_sharded,
            )
        else:
            assert lora_routing is not None
            moe_lora_apply_w13(
                lora_context,
                gate_up_out=output,
                hidden_states=hidden_states,
                lora_routing=lora_routing,
            )

    input_dtype = hidden_states.dtype

    def quantize_w13_input() -> tuple[torch.Tensor, torch.Tensor]:
        return DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=torch.int8,
            use_mxfp_quant=False,
        )

    def base_w13_gmm_fn(
        quantized_input: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch_npu.npu_grouped_matmul(
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

    def base_w13_fn() -> torch.Tensor:
        quantized_input, input_scale = quantize_w13_input()
        return base_w13_gmm_fn(quantized_input, input_scale)

    lora_delta_workspace = None
    w2_output_size = 0
    if use_aux_stream:
        lora_delta_workspace, w13_output_size, w2_output_size = _new_lora_delta_workspace(
            hidden_states,
            lora_context.w13_lora_b_stacked,
            lora_context.w2_lora_b_stacked,
        )
        lora_delta_w13 = lora_delta_workspace[..., :w13_output_size]

        def lora_w13_fn() -> None:
            lora_delta_w13.zero_()
            apply_w13_lora(lora_delta_w13)

        w13_base_fn = base_w13_fn
        w13_aux_prepare_fn: Callable[[], None] | None = None
        if defer_allgather_lora_routing:
            quantized_input, input_scale = quantize_w13_input()

            def quantized_base_w13_fn() -> torch.Tensor:
                return base_w13_gmm_fn(quantized_input, input_scale)

            w13_base_fn = quantized_base_w13_fn
            w13_aux_prepare_fn = prepare_lora_routing

        gate_up_out = _execute_moe_lora_in_parallel(
            w13_base_fn,
            lora_w13_fn,
            lora_context.events[0],
            lora_context.events[1],
            lora_context.aux_stream,
            aux_prepare_fn=w13_aux_prepare_fn,
        )
        gate_up_out.add_(lora_delta_w13)
    else:
        gate_up_out = base_w13_fn()
        apply_w13_lora(gate_up_out)

    activated = _apply_moe_activation(
        gate_up_out,
        mlp_compute_input.activation,
        mlp_compute_input.swiglu_limit,
        mlp_compute_input.swiglu_alpha,
        mlp_compute_input.swiglu_beta,
    )
    if mlp_compute_input.topk_scales is not None:
        activated *= mlp_compute_input.topk_scales

    def apply_w2_lora(output: torch.Tensor) -> None:
        output_offset = 0
        if lora_context.fully_sharded:
            output_offset = lora_context.w2_lora_b_stacked[0].shape[-2] * lora_context.tp_rank
        if use_single_lora_gmm:
            assert single_lora_routing is not None
            # activated already includes topk_scales, so the W2 LoRA delta
            # must not multiply routed weights a second time.
            _add_single_lora_gmm(
                output,
                activated,
                lora_context.w2_lora_a_packed,
                lora_context.w2_lora_b_packed,
                routing=single_lora_routing,
                group_list=mlp_compute_input.group_list,
                fully_sharded=lora_context.fully_sharded,
                output_offset=output_offset,
            )
            reset_lora_indices(lora_context)
        elif use_composite_lora_gmm:
            assert composite_lora_routing is not None
            _add_composite_lora_gmm(
                output,
                activated,
                lora_context.w2_lora_a_stacked,
                lora_context.w2_lora_b_stacked,
                routing=composite_lora_routing,
                fully_sharded=lora_context.fully_sharded,
                output_offset=output_offset,
            )
            reset_lora_indices(lora_context)
        else:
            assert lora_routing is not None
            moe_lora_apply_w2(
                lora_context,
                down_out=output,
                silu_out=activated,
                lora_routing=lora_routing,
            )

    def base_w2_fn() -> torch.Tensor:
        quantized_activated, activated_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=activated,
            dynamic_scale=None,
            act_quant_type=torch.int8,
            use_mxfp_quant=False,
        )
        return DeviceOperator.npu_grouped_matmul_gmm2(
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

    if use_aux_stream:
        assert lora_delta_workspace is not None
        lora_delta_w2 = lora_delta_workspace[..., :w2_output_size]

        def lora_w2_fn() -> None:
            lora_delta_w2.zero_()
            apply_w2_lora(lora_delta_w2)

        down_out = _execute_moe_lora_in_parallel(
            base_w2_fn,
            lora_w2_fn,
            lora_context.events[2],
            lora_context.events[3],
            lora_context.aux_stream,
        )
        down_out.add_(lora_delta_w2)
    else:
        down_out = base_w2_fn()
        apply_w2_lora(down_out)
    return down_out, None


__all__ = [
    "quant_apply_mlp_with_moe_lora",
    "register_quant_moe_lora_impl",
    "validate_quant_moe_lora_activation_input",
]
