#
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
#
"""Ascend MoE LoRA wrapper and routed-token mapping helpers.

The wrapper reuses upstream weight allocation, loading, and TP/EP slicing but
publishes the resulting LoRA context through Ascend's MoERunner pipeline rather
than the GPU modular kernel. Unquantized LoRA keeps the existing AllGather and
AlltoAll implementations. Quantized backends inject deltas at their floating
point GMM boundaries; the first implementation supports W8A8_DYNAMIC with
AllGather TP/EP and AlltoAll EP execution.

Shared experts remain ordinary dense LoRA layers. This module preserves their
module hierarchy and selects a compatible NPU dense expand implementation when
the MoE wrapper is mapped.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm import envs
from vllm.logger import logger
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all
from vllm_ascend.quantization.quant_type import QuantType

_MOE_LORA_INDEX_FIELDS = (
    "split_lora_indices",
    "permuted_lora_indices",
    "exchanged_lora_indices",
    "allgather_lora_indices",
)

# The mixed-adapter composite GMM path still targets the production DeepSeek
# configuration. The single-adapter path derives these dimensions from its
# packed weights instead.
MOE_LORA_GMM_MAX_LORAS = 3
MOE_LORA_GMM_RANK = 16
MOE_LORA_GMM_TOP_K = 6


def has_lora(lora_context) -> bool:
    """Return whether this rank must execute the LoRA-aware MoE path."""
    return lora_context is not None and (
        getattr(lora_context, "allgather_lora_indices", None) is not None or not lora_context.punica_wrapper.no_lora
    )


def reset_lora_indices(lora_context) -> None:
    for field in _MOE_LORA_INDEX_FIELDS:
        if hasattr(lora_context, field):
            delattr(lora_context, field)


def get_allgather_lora_indices(lora_context) -> torch.Tensor:
    """Return token-to-LoRA indices aligned with AllGather input rows."""
    gathered_indices = getattr(lora_context, "allgather_lora_indices", None)
    if gathered_indices is not None:
        return gathered_indices
    return lora_context.punica_wrapper.token_lora_indices


def prepare_lora_indices(
    lora_context,
    *,
    num_tokens: int,
    pad_size: int,
    tp_size: int,
    tp_rank: int,
) -> None:
    """Truncate, pad, and TP-split the per-token LoRA index tensor,
    storing the result in ``lora_context.split_lora_indices``.

    Caller must ensure ``lora_context`` is not ``None``.
    """
    token_indices = lora_context.punica_wrapper.token_lora_indices
    token_indices = token_indices[:num_tokens]
    if pad_size > 0:
        token_indices = torch.nn.functional.pad(token_indices, (0, pad_size), value=-1)
    if tp_size > 1:
        lora_context.split_lora_indices = torch.tensor_split(token_indices, tp_size, dim=0)[tp_rank]
    else:
        # use ep for dp without tp.
        lora_context.split_lora_indices = token_indices


def preprocess_lora_indices(
    lora_context,
    *,
    topk_ids: torch.Tensor,
    reversed_permutation_mapping: torch.Tensor,
) -> None:
    """Expand and permute LoRA token indices for the AlltoAll dispatch path.

    Reads ``lora_context.split_lora_indices``, broadcasts each entry across
    ``topk``, applies the token permutation to align with the dispatched
    hidden states, and stores the result in
    ``lora_context.permuted_lora_indices``.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``split_lora_indices`` has been populated.
    """
    split_indices = getattr(lora_context, "split_lora_indices", None)
    if split_indices is None:
        return
    expanded = split_indices.unsqueeze(-1).expand(-1, topk_ids.shape[1]).reshape(-1)
    permutation = torch.argsort(reversed_permutation_mapping.reshape(-1).long())
    lora_context.permuted_lora_indices = expanded[permutation]


def postprocess_lora_indices(
    lora_context,
    *,
    reversed_permutation_mapping: torch.Tensor,
) -> None:
    """Re-permute exchanged LoRA indices to align with the global token
    ordering after ``npu_moe_token_permute`` in the AlltoAll dispatch
    postprocess.

    Reads ``lora_context.exchanged_lora_indices``, applies the
    global permutation, and writes the result back.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``exchanged_lora_indices`` has been populated.
    """
    exchanged = getattr(lora_context, "exchanged_lora_indices", None)
    if exchanged is None:
        return
    permutation = torch.argsort(reversed_permutation_mapping.reshape(-1).long())
    lora_context.exchanged_lora_indices = exchanged[permutation]


def all2all_lora_indices(
    lora_context,
    *,
    output_splits,
    input_splits,
    ep_group,
) -> None:
    """Exchange permuted LoRA indices across EP ranks via all_to_all.

    Reads ``lora_context.permuted_lora_indices``, performs the all_to_all
    exchange with the given splits and group, and stores the result in
    ``lora_context.exchanged_lora_indices``.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``permuted_lora_indices`` has been populated.
    """
    permuted = getattr(lora_context, "permuted_lora_indices", None)
    if permuted is None:
        return
    lora_dtype = permuted.dtype
    _, exchanged, handle = async_all_to_all(permuted, output_splits, input_splits, ep_group)
    handle.wait()
    lora_context.exchanged_lora_indices = exchanged.to(lora_dtype)


def sync_lora_context(quant_method, lora_context):
    """Push ``lora_context`` onto MoE communication singletons, or clear
    them when ``lora_context`` is ``None``.

    Encapsulates the ``hasattr``/``set_lora_context`` pattern shared by
    setup and teardown so callers just pass the target value.
    """
    if hasattr(_EXTRA_CTX.moe_comm_method, "set_lora_context"):
        _EXTRA_CTX.moe_comm_method.set_lora_context(lora_context)
    if hasattr(quant_method, "set_lora_context"):
        quant_method.set_lora_context(lora_context)


def _assert_ascend_moe_lora_supported(base_layer: nn.Module) -> None:
    if getattr(base_layer, "dynamic_eplb", False):
        raise AssertionError(
            "Ascend MoE LoRA is incompatible with dynamic EPLB "
            "(expert migration would break the per-expert LoRA layout)."
        )
    if int(envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2) != 0:
        raise AssertionError(
            "Ascend MoE LoRA cannot patch FusedMC2 path "
            "(dispatch_ffn_combine/mega_moe is a single fused C++ op). "
            "Set VLLM_ASCEND_ENABLE_FUSED_MC2=0."
        )
    if getattr(base_layer, "_shared_experts", None) is not None:
        logger.warning_once(
            "Ascend MoE LoRA: shared_experts detected. Routed-expert LoRA "
            "uses the MoE path; shared-expert LoRA uses dense wrappers with "
            "the compatible NPU expand-slice implementation."
        )


def _recover_moe_lora_routing_allgather(
    lora_context,
    expanded_row_idx,
    topk_ids,
    expert_map: torch.Tensor | None = None,
):
    """Recover per-permuted-row (expert_id, lora_slot) for the dispatched rows.

    npu_moe_init_routing semantics (verified empirically): ``expanded_row_idx``
    is indexed by the ORIGINAL flat (token, k) position and gives where that
    pair landed in the expert-sorted array -- not the reverse. Without EP, the
    full mapping is a permutation and can be inverted with ``argsort``. With
    EP, only destinations for this rank's active expert range are valid, so
    local metadata is scattered into those destinations instead. Both paths
    keep static shapes and avoid ``.item()`` or other host synchronization.
    """
    top_k = lora_context.top_k
    expanded = expanded_row_idx.to(torch.long).abs()
    flat_expert_ids = topk_ids.reshape(-1).to(torch.long)
    token_lora_indices = get_allgather_lora_indices(lora_context)

    if expert_map is not None:
        # With active_expert_range, init-routing only guarantees that the
        # local rows at the start of expanded_x are valid; mapping entries for
        # non-local rows may be repeated or otherwise invalid. Scatter the
        # original pair metadata into its destination instead of inverting the
        # full mapping. Non-local pairs contribute zero, so collisions cannot
        # overwrite a valid local row.
        local_expert_ids = expert_map[flat_expert_ids].to(torch.long)
        is_local = local_expert_ids >= 0
        destination = expanded.clamp_(max=max(flat_expert_ids.numel() - 1, 0))

        encoded_expert_ids = torch.where(
            is_local,
            local_expert_ids + 1,
            torch.zeros_like(local_expert_ids),
        )
        expert_per_row = torch.zeros_like(encoded_expert_ids)
        expert_per_row.scatter_add_(0, destination, encoded_expert_ids)
        expert_per_row.sub_(1).clamp_min_(0)

        lora_per_pair = token_lora_indices[: topk_ids.shape[0]].unsqueeze(-1).expand_as(topk_ids).reshape(-1)
        encoded_lora_ids = torch.where(
            is_local & (lora_per_pair >= 0),
            lora_per_pair + 1,
            torch.zeros_like(lora_per_pair),
        )
        lora_per_row = torch.zeros_like(encoded_lora_ids)
        lora_per_row.scatter_add_(0, destination, encoded_lora_ids)
        lora_per_row.sub_(1)
        return expert_per_row, lora_per_row

    inv_perm = torch.argsort(expanded)
    expert_per_row = flat_expert_ids[inv_perm]

    # token_lora_indices is a 1D LongTensor sized to max_num_batched_tokens
    # (host-known constant). Clamping defensively to the last index is a no-op
    # in normal operation but keeps the gather graph-safe.
    orig_token = inv_perm // top_k
    orig_token = orig_token.clamp_(max=token_lora_indices.numel() - 1)
    lora_per_row = token_lora_indices[orig_token]
    return expert_per_row, lora_per_row


def _recover_moe_lora_routing_all2all(
    lora_context,
    group_list: torch.Tensor,
):
    """Recover per-row (expert_id, lora_id) for the AlltoAll dispatched tokens.

    In the AlltoAll + EP path, tokens have already been exchanged via
    all_to_all and sorted by local expert.  ``group_list`` tells us how
    many tokens belong to each local expert. The LoRA indices for those
    dispatched rows are stored on ``lora_context.exchanged_lora_indices``.

    Returns:
        expert_per_row: [num_dispatched_tokens] local expert id (0..E-1)
        lora_per_row:   [num_dispatched_tokens] lora adapter id (-1 = none)
    """
    num_local_experts = lora_context.local_num_experts
    exchanged_lora_indices = getattr(lora_context, "exchanged_lora_indices", None)
    if exchanged_lora_indices is None:
        raise AssertionError("AlltoAll MoE LoRA requires exchanged_lora_indices in lora_context.")

    lora_per_row = exchanged_lora_indices.reshape(-1).to(torch.long)
    if group_list.numel() != num_local_experts:
        raise AssertionError(
            "AlltoAll MoE LoRA routing metadata is misaligned: "
            f"expected {num_local_experts} local expert counts, but "
            f"received {group_list.numel()}."
        )

    # Mark every expert boundary, then recover the expert id for each row
    # with a cumulative sum. Multiple empty experts share a boundary and
    # therefore contribute multiple increments at the same position. Unlike
    # tensor-repeat expansion, every intermediate has a static
    # shape and is safe to capture in ACLGraph.
    group_counts = group_list.to(torch.long)
    expert_starts = torch.cumsum(group_counts, dim=0) - group_counts
    boundary_increments = torch.zeros(
        lora_per_row.shape[0] + 1,
        dtype=torch.long,
        device=group_list.device,
    )
    boundary_increments.scatter_add_(
        0,
        expert_starts.clamp(max=lora_per_row.shape[0]),
        torch.ones_like(expert_starts),
    )
    expert_per_row = torch.cumsum(boundary_increments[:-1], dim=0).sub_(1)

    return expert_per_row, lora_per_row


def moe_lora_apply_w13(lora_context, *, gate_up_out, hidden_states, lora_routing):
    """Add the w13 LoRA delta into ``gate_up_out`` (in place), before activation.

    Called from ``unquant_apply_mlp`` right after the base gate_up GMM.

    Args:
        lora_routing: (expert_per_row, lora_per_row) pre-computed by the
            caller via _recover_moe_lora_routing (AllGather) or
            _recover_moe_lora_routing_all2all (AlltoAll).
    """
    expert_per_row, lora_per_row = lora_routing
    # EP rank may receive 0 dispatched tokens when all tokens route to
    # experts on other ranks. Skip LoRA to avoid passing empty tensors
    # to add_lora_fused_moe (which can trigger NPU kernel crashes).
    if expert_per_row.numel() == 0:
        return
    lora_context.punica_wrapper.add_lora_fused_moe(
        y=gate_up_out,
        x=hidden_states,
        lora_a_stacked=lora_context.w13_lora_a_stacked,
        lora_b_stacked=lora_context.w13_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        fully_sharded=lora_context.fully_sharded,
        token_lora_mapping=lora_per_row,
    )


def moe_lora_apply_w2(lora_context, *, down_out, silu_out, lora_routing):
    """Add the w2 LoRA delta into ``down_out`` (in place), after the down GMM.

    Reuses the per-row routing computed by ``moe_lora_apply_w13``; ``silu_out``
    is the activation output that fed the base down GMM.
    """
    expert_per_row, lora_per_row = lora_routing
    # EP rank may receive 0 dispatched tokens; skip LoRA to avoid NPU
    # kernel crashes with empty tensors.
    if expert_per_row.numel() == 0:
        return
    offset = 0
    if lora_context.fully_sharded:
        shard_size = lora_context.w2_lora_b_stacked[0].shape[-2]
        offset = shard_size * lora_context.tp_rank
    lora_context.punica_wrapper.add_lora_fused_moe(
        y=down_out,
        x=silu_out,
        lora_a_stacked=lora_context.w2_lora_a_stacked,
        lora_b_stacked=lora_context.w2_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        fully_sharded=lora_context.fully_sharded,
        offset=offset,
        token_lora_mapping=lora_per_row,
    )
    # Clear per-forward intermediate indices now that the LoRA delta
    # for this layer has been fully applied — they are not needed for
    # the remaining combine/finalize stages.
    reset_lora_indices(lora_context)


class AscendFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Ascend-native MoE-LoRA wrapper.

    Reuses upstream weight allocation, set_lora, reset_lora, and slicing.
    Instead of the GPU modular-kernel injection, it publishes a per-layer
    ``MoELoRAContext`` onto the base layer (``_ascend_moe_lora_context``).
    The Ascend unquant MoE path threads that context through
    ``MoEFusedExpertsInput`` -> ``MoEMlpComputeInput`` and applies the LoRA
    delta natively inside ``unquant_apply_mlp`` (see
    ``moe_lora_apply_w13`` / ``moe_lora_apply_w2`` below) -- no runtime
    monkey-patch of ``comm._apply_mlp``.
    """

    def __init__(self, base_layer: nn.Module) -> None:
        # Skip FusedMoEWithLoRA.__init__: it immediately asserts Triton
        # internals and calls _inject_lora_into_fused_moe which is GPU-only.
        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        _assert_ascend_moe_lora_supported(base_layer)
        self.moe_config = base_layer.moe_config
        # Match upstream FusedMoEWithLoRA: EP collapses the MoE TP dimension
        # to one and shards experts across the original TP group.  Using the
        # global TP rank/size here would incorrectly TP-slice every local
        # expert's LoRA weights a second time.
        moe_parallel_config = self.moe_config.moe_parallel_config
        self.tp_size = moe_parallel_config.tp_size
        self.tp_rank = moe_parallel_config.tp_rank
        self.device = _get_lora_device(base_layer)
        self._enable_aux_cuda_stream = envs.VLLM_LORA_ENABLE_DUAL_STREAM
        # _build_lora_context is inherited from vLLM, whose GPU constructor
        # normally initializes these fields. Ascend deliberately skips it.
        self._lora_stream = None
        self._events = None
        self.enable_moe_shared_loras = False
        self._single_lora_packed_weights = None
        self._single_lora_slot_views = None
        self._single_lora_cache_slot = None
        self._ascend_lora_context = None
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        # Mirrors per-(lora_id) layout of `self.lora_a_stacked` (built in
        # `create_lora_weights`) so `create_dummy_lora`'s n_slices fallback
        # matches `lora_a_stacked` length under EP.
        self.n_slices = self.local_num_experts * (self._w13_slices + 1)
        # Preserve the model-manager-visible module path used to discover and
        # wrap shared_experts.{gate_up,down}_proj as ordinary dense LoRA.
        shared_experts = getattr(base_layer, "_shared_experts", None)
        if shared_experts is not None:
            self._shared_experts = shared_experts

    def _build_lora_context(self):
        lora_context = super()._build_lora_context()
        lora_context.use_ep = self.use_ep
        packed_weights = self._single_lora_packed_weights
        if packed_weights is None:
            packed_weights = (None, None, None, None)
        (
            lora_context.w13_lora_a_packed,
            lora_context.w13_lora_b_packed,
            lora_context.w2_lora_a_packed,
            lora_context.w2_lora_b_packed,
        ) = packed_weights
        lora_context.single_lora_cache_slot = self._single_lora_cache_slot
        return lora_context

    def create_lora_weights(self, max_loras, lora_config, model_config=None) -> None:
        """Allocate upstream weights and an optional fixed-address GMM cache."""
        super().create_lora_weights(max_loras, lora_config, model_config)
        self._single_lora_packed_weights = None
        self._single_lora_slot_views = None
        self._single_lora_cache_slot = None

        weights = (
            self.w13_lora_a_stacked,
            self.w13_lora_b_stacked,
            self.w2_lora_a_stacked,
            self.w2_lora_b_stacked,
        )
        supports_packed_cache = (
            getattr(self.base_layer, "quant_type", None) == QuantType.W8A8
            and not self.enable_moe_shared_loras
            and all(weight.shape[0] == self.max_loras for stacked in weights for weight in stacked)
            and all(weight.shape[1] == self.local_num_experts for stacked in weights for weight in stacked)
        )
        if not supports_packed_cache:
            return

        # GMM always reads these one-adapter buffers at stable addresses. Slot
        # views are created once so an adapter switch needs only device copies,
        # not a runtime index/index_select operation over full weight banks.
        self._single_lora_packed_weights = tuple(
            tuple(weight.new_empty(weight.shape[1:]) for weight in stacked) for stacked in weights
        )
        self._single_lora_slot_views = tuple(tuple(weight.unbind(0) for weight in stacked) for stacked in weights)

    def refresh_single_lora_cache(self, slot: int | None) -> None:
        """Refresh fixed-address packed weights when the active slot changes."""
        if slot is None:
            self._single_lora_cache_slot = None
            if self._ascend_lora_context is not None:
                self._ascend_lora_context.single_lora_cache_slot = None
            return
        if self._single_lora_packed_weights is None or self._single_lora_slot_views is None:
            return
        if slot == self._single_lora_cache_slot:
            return
        if not 0 <= slot < self.max_loras:
            raise ValueError(f"LoRA slot {slot} is outside the packed cache range [0, {self.max_loras}).")

        for packed_stack, slot_view_stack in zip(
            self._single_lora_packed_weights,
            self._single_lora_slot_views,
            strict=True,
        ):
            for packed, slot_views in zip(packed_stack, slot_view_stack, strict=True):
                packed.copy_(slot_views[slot], non_blocking=True)
        self._single_lora_cache_slot = slot
        if self._ascend_lora_context is not None:
            self._ascend_lora_context.single_lora_cache_slot = slot

    def reset_lora(self, index: int):
        refresh_cache = index == self._single_lora_cache_slot or index == getattr(
            getattr(self, "punica_wrapper", None),
            "active_moe_lora_slot",
            None,
        )
        super().reset_lora(index)
        if refresh_cache:
            self.refresh_single_lora_cache(None)
            self.refresh_single_lora_cache(index)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        refresh_cache = index == self._single_lora_cache_slot or index == getattr(
            getattr(self, "punica_wrapper", None),
            "active_moe_lora_slot",
            None,
        )
        super().set_lora(index, lora_a, lora_b)
        if refresh_cache:
            # ``super().set_lora`` calls our reset method before loading the
            # new factors. Force one final copy after all source views update.
            self.refresh_single_lora_cache(None)
            self.refresh_single_lora_cache(index)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    def set_mapping(self, punica_wrapper):
        # Upstream FusedMoEWithLoRA.set_mapping (vllm v0.22.0+) chains into
        # ``self._moe_kernel.fused_experts.set_lora_context(...)``, but
        # ``_moe_kernel`` is only set by the GPU modular-kernel path that we
        # deliberately skip in __init__. We instead build the per-layer
        # MoELoRAContext (now that punica_wrapper is available) and publish it
        # on the module that ``AscendUnquantizedFusedMoEMethod.apply`` reads via
        # ``getattr(layer, "_ascend_moe_lora_context", None)``
        # Build the per-layer MoELoRAContext once punica_wrapper is available and
        # publish it through the Ascend MoE runner. The runner stores it on
        # routed_experts; batch-local LoRA indices are refreshed before each forward.
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)
        self._ascend_lora_context = self._build_lora_context()
        punica_wrapper.register_single_lora_moe_layer(self)
        self.base_layer.set_lora_context(self._ascend_lora_context)


class AscendFusedMoE3DWithLoRA(AscendFusedMoEWithLoRA, FusedMoE3DWithLoRA):
    """For checkpoints that already fuse w1+w3 into a 3D weight (single slice)."""

    def __init__(self, base_layer: nn.Module) -> None:
        AscendFusedMoEWithLoRA.__init__(self, base_layer)
        # Override: 3D MoE LoRA uses a single w13 slice.
        self._w13_slices = 1


# ----------------------------------------------------------------------
# Upstream compatibility shim: vllm/lora/model_manager.py:create_dummy_lora
# branches on `module.__class__.__name__ == "FusedMoEWithLoRA"` (and the
# 3D variant). Without this override, our subclasses would skip the
# pack_moe path and hit the generic pack() fallback, which produces a
# flat list of N_experts * 3 sub-LoRAs -- `set_lora` then fails with
# "too many values to unpack (expected 3)".
#
# Overriding only __name__ keeps the actual class object distinct (so
# isinstance / type identity / debugging are unaffected) but lets the
# upstream string compare hit our objects.
# ----------------------------------------------------------------------
AscendFusedMoEWithLoRA.__name__ = "FusedMoEWithLoRA"
AscendFusedMoE3DWithLoRA.__name__ = "FusedMoE3DWithLoRA"
