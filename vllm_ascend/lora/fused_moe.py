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
AllGather execution in both TP-only and expert-parallel deployments (local
expert mapping), plus AlltoAll EP execution.

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

_MOE_LORA_INDEX_FIELDS = (
    "split_lora_indices",
    "permuted_lora_indices",
    "exchanged_lora_indices",
    "combined_lora_idx",
)


def has_lora(lora_context) -> bool:
    """Return whether the current batch contains at least one LoRA token."""
    return lora_context is not None and not lora_context.punica_wrapper.no_lora


def reset_lora_indices(lora_context) -> None:
    for field in _MOE_LORA_INDEX_FIELDS:
        if hasattr(lora_context, field):
            delattr(lora_context, field)


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


def _gather_by_row_permutation(values: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """
    Sorting int32/int64 keys falls back to AiCpu on Ascend,
    so we cast it in fp32. Result is identical.
    """
    return values[torch.argsort(mapping.reshape(-1).to(torch.float32))]


def preprocess_lora_indices(
    lora_context,
    *,
    topk_ids: torch.Tensor,
    reversed_permutation_mapping: torch.Tensor,
) -> None:
    """Expand and permute LoRA token indices for the AlltoAll dispatch path.

    Reads ``lora_context.split_lora_indices``, repeats each entry by
    ``topk``, applies the token permutation to align with the dispatched
    hidden states, and stores the result in
    ``lora_context.permuted_lora_indices``.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``split_lora_indices`` has been populated.
    """
    split_indices = getattr(lora_context, "split_lora_indices", None)
    if split_indices is None:
        return
    expanded = split_indices.repeat_interleave(topk_ids.shape[1])
    lora_context.permuted_lora_indices = _gather_by_row_permutation(
        expanded, reversed_permutation_mapping
    )


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
    lora_context.exchanged_lora_indices = _gather_by_row_permutation(
        exchanged, reversed_permutation_mapping
    )


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


def _recover_moe_lora_routing_allgather(lora_context, expanded_row_idx, topk_ids):
    """Recover per-permuted-row (expert_id, lora_slot) for the dispatched rows.

    npu_moe_init_routing semantics (verified empirically): ``expanded_row_idx``
    is indexed by the ORIGINAL flat (token, k) position and gives where that
    pair landed in the expert-sorted array -- not the reverse. So recovering
    "which (token, k) pair does sorted row i hold" needs the inverse permutation
    of ``expanded``, not a direct gather by it. ``argsort`` output shape ==
    input shape (value-independent), so this stays graph-capturable -- no
    ``.item()``/data-dependent host sync.

    Under expert parallelism the dispatcher restricts routing to the local
    expert range (``active_expert_range``). Per the npu_moe_init_routing_v2
    contract (row_idx_type=0): active pairs keep their compacted destination
    row in ``[0, available)`` while inactive pairs are filled with -1, and
    the tail rows of the static (num_tokens * top_k)-row tensors are
    undefined. Recovery therefore (a) pushes inactive pairs past the valid
    destination range before ``argsort`` so the first entries of the inverse
    still map dispatched row -> original pair, and (b) converts global expert
    ids to this rank's local ids, disabling LoRA on rows whose pair routed
    to another rank (including the undefined tail) via the -1 slot sentinel.
    """
    top_k = lora_context.top_k
    dest = expanded_row_idx.reshape(-1).to(torch.long)
    num_pairs = dest.numel()
    active = dest >= 0
    # Keys keep active destinations in [0, available) and push inactive
    # pairs beyond num_pairs (distinct keys -> deterministic argsort), so
    # inv maps dispatched rows to their original pairs and every undefined
    # tail row lands on an inactive pair that is masked out below.
    keys = torch.where(
        active, dest, num_pairs + torch.arange(num_pairs, device=dest.device)
    )
    # again sorting issue with fallback to AiCPU
    inv = torch.argsort(keys.to(torch.float32))
    expert_per_row = topk_ids.reshape(-1)[inv].to(torch.long)

    # token_lora_indices is a 1D LongTensor sized to max_num_batched_tokens
    # (host-known constant). When prepare split the batch across TP ranks
    # (flash-comm off), the dispatched rows come from this rank's shard and
    # split_lora_indices carries the matching (padded) shard slots; with
    # flash-comm on (the SP/EP AllGather deployment) prepare keeps the full
    # batch and the global indices apply. Prefer the split when present.
    lora_indices = getattr(lora_context, "split_lora_indices", None)
    if lora_indices is None:
        lora_indices = lora_context.punica_wrapper.token_lora_indices
    orig_token = inv // top_k
    orig_token = orig_token.clamp_(max=lora_indices.numel() - 1)
    lora_per_row = lora_indices[orig_token]

    if getattr(lora_context, "use_ep", False):
        # Same contiguous-sharding formula as the AllGather dispatcher's
        # active_expert_range: first = ep_rank * num_local_experts.
        from vllm.distributed.parallel_state import get_ep_group

        num_local_experts = lora_context.local_num_experts
        first_expert_idx = get_ep_group().rank_in_group * num_local_experts
        local_expert = expert_per_row - first_expert_idx
        local_row = (local_expert >= 0) & (local_expert < num_local_experts)
        # Rows outside the local range (other ranks' pairs and the undefined
        # tail) must not index the EP-sharded LoRA stacks: clamp the expert
        # id for safety and disable the adapter slot so add_lora_fused_moe's
        # combined gather index becomes -1 (bgmv skips those rows entirely).
        expert_per_row = local_expert.clamp_(0, num_local_experts - 1)
        lora_per_row = torch.where(
            local_row, lora_per_row, torch.full_like(lora_per_row, -1)
        )
    return expert_per_row, lora_per_row


def _build_combined_lora_idx_allgather(lora_context, expanded_row_idx, topk_ids):
    """Build the per-row combined gather index (AG path).

    aten builds the unique fp32 sort keys and argsorts them; the in-tree
    kernel (torch.ops._C_ascend.build_combined_lora_idx,
    csrc/kernels/combined_lora_idx.cpp) then folds the per-pair
    slot/expert/adapter lookup into a single gather pass.
    Bit-identical to _build_combined_lora_idx_allgather_torch, which is kept
    as the reference implementation for the equality tests.
    """
    # token->slot mapping: prefer the TP-split shard when present
    lora_indices = getattr(lora_context, "split_lora_indices", None)
    if lora_indices is None:
        lora_indices = lora_context.punica_wrapper.token_lora_indices
    first_expert_idx = 0
    if getattr(lora_context, "use_ep", False):
        # Same contiguous-sharding formula as the AllGather dispatcher's
        # active_expert_range: first = ep_rank * num_local_experts.
        from vllm.distributed.parallel_state import get_ep_group

        first_expert_idx = get_ep_group().rank_in_group * lora_context.local_num_experts
    num_experts = lora_context.w13_lora_a_stacked[0].shape[1]
    return torch.ops._C_ascend.build_combined_lora_idx(
        expanded_row_idx,
        topk_ids,
        lora_indices,
        lora_context.adapter_enabled,
        first_expert_idx,
        num_experts,
        lora_context.top_k,
    )


def _build_combined_lora_idx_allgather_torch(lora_context, expanded_row_idx, topk_ids):
    """Reference torch implementation of the combined gather index (AG path).

    Produces the combined value per (token, k) pair and applies ONE final
    gather by the argsort inverse, instead of recovering two per-row tensors
    and recombining them.

    Row semantics: rows whose pair routed to another rank (or the undefined
    tail) land on inactive pairs and receive the -1 sentinel.
    """
    top_k = lora_context.top_k
    dest = expanded_row_idx.reshape(-1).to(torch.long)
    num_pairs = dest.numel()
    active = dest >= 0
    # Identical key construction to _recover_moe_lora_routing_allgather, so
    # `inv` is bit-identical (see its fp32-sort rationale).
    keys = torch.where(active, dest, num_pairs + torch.arange(num_pairs, device=dest.device))
    inv = torch.argsort(keys.to(torch.float32))

    # Per-pair LoRA slot: pair p belongs to token p // top_k.
    lora_indices = getattr(lora_context, "split_lora_indices", None)
    if lora_indices is None:
        lora_indices = lora_context.punica_wrapper.token_lora_indices
    num_tokens = min(num_pairs // top_k, lora_indices.numel())
    # NOTE: expand+reshape, NOT repeat_interleave -- NPU's repeat_interleave
    # degrades catastrophically at prefill sizes (~14ms at 98k pairs vs
    # ~36us for this copy). Bitwise-equal output.
    slot_pairs = (
        lora_indices[:num_tokens]
        .unsqueeze(1)
        .expand(num_tokens, top_k)
        .reshape(-1)
    )
    if slot_pairs.numel() < num_pairs:  # shapes are static per captured graph
        slot_pairs = torch.nn.functional.pad(
            slot_pairs, (0, num_pairs - slot_pairs.numel()), value=-1
        )

    # Per-pair local expert id (identity without EP).
    if getattr(lora_context, "use_ep", False):
        from vllm.distributed.parallel_state import get_ep_group

        num_local_experts = lora_context.local_num_experts
        first_expert_idx = get_ep_group().rank_in_group * num_local_experts
        expert_pairs = topk_ids.reshape(-1).to(torch.long) - first_expert_idx
    else:
        expert_pairs = topk_ids.reshape(-1).to(torch.long)

    num_experts = lora_context.w13_lora_a_stacked[0].shape[1]
    slot_safe = slot_pairs.clamp(min=0)
    enabled = (
        active
        & (slot_pairs >= 0)
        & lora_context.adapter_enabled[slot_safe].bool()
    )
    combined_pairs = torch.where(
        enabled, slot_safe * num_experts + expert_pairs, -1
    )
    return combined_pairs[inv].contiguous()


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
    exchanged_lora_indices = getattr(lora_context, "exchanged_lora_indices", None)
    if exchanged_lora_indices is None:
        raise AssertionError(
            "AlltoAll MoE LoRA requires exchanged_lora_indices in lora_context."
        )

    # Build per-token expert IDs via cumsum+searchsorted: equivalent to the
    # previous repeat_interleave(arange(E), group_list) but the search runs
    # on the vector cores (repeat_interleave with tensor repeats takes a slower
    # path). Output length stays sum(group_list)-dependent as before
    group_ends = torch.cumsum(group_list.to(torch.int64), dim=0)
    num_rows = int(group_ends[-1])
    expert_per_row = torch.searchsorted(
        group_ends, torch.arange(num_rows, device=group_list.device), right=True
    )

    lora_per_row = exchanged_lora_indices.reshape(-1).to(torch.long)
    if expert_per_row.numel() != lora_per_row.numel():
        raise AssertionError(
            "AlltoAll MoE LoRA routing metadata is misaligned: "
            f"group_list describes {expert_per_row.numel()} rows, but "
            f"received {lora_per_row.numel()} LoRA indices."
        )

    return expert_per_row, lora_per_row


def moe_lora_apply_w13(
    lora_context,
    *,
    gate_up_out,
    hidden_states,
    lora_routing,
    group_list=None,
    group_list_type=1,
):
    """Apply the w13 LoRA delta, before activation.

    Called from ``unquant_apply_mlp`` right after the base gate_up GMM.

    Args:
        lora_routing: (expert_per_row, lora_per_row) pre-computed by the
            caller via _recover_moe_lora_routing (AllGather) or
            _recover_moe_lora_routing_all2all (AlltoAll).
        group_list: base per-expert row counts, reused by the gmm path.

    Returns:
        The delta when it was not folded into ``gate_up_out``, else None.
    """
    # AG callers (quant_moe / unquant_apply_mlp) stash the single-pass
    # combined index built by _build_combined_lora_idx_allgather directly;
    # A2A callers pass the recovered routing pair and we build here
    combined_idx = getattr(lora_context, "combined_lora_idx", None)
    if combined_idx is not None and combined_idx.numel() != gate_up_out.shape[0]:
        combined_idx = None  # stale stash: fall through to the routing path
    if combined_idx is None:
        if lora_routing is None:
            raise AssertionError(
                "moe_lora_apply_w13 requires either a stashed combined index "
                "or a routing pair"
            )
        expert_per_row, lora_per_row = lora_routing
        # EP rank may receive 0 dispatched tokens when all tokens route to
        # experts on other ranks. Skip LoRA to avoid passing empty tensors
        # to add_lora_fused_moe (which can trigger NPU kernel crashes).
        if expert_per_row.numel() == 0:
            return None
        # The w13 and w2 applies of this layer share one routing, hence one
        # combined gather index: build it once here, stash it for w2, and
        # pass it through explicitly.
        from vllm_ascend.lora.punica_npu import build_combined_lora_idx

        combined_idx = build_combined_lora_idx(
            lora_per_row,
            expert_per_row,
            lora_context.adapter_enabled,
            lora_context.w13_lora_a_stacked[0].shape[1],
        )
        lora_context.combined_lora_idx = combined_idx
        return lora_context.punica_wrapper.add_lora_fused_moe(
            y=gate_up_out,
            x=hidden_states,
            lora_a_stacked=lora_context.w13_lora_a_stacked,
            lora_b_stacked=lora_context.w13_lora_b_stacked,
            expert_ids=expert_per_row,
            adapter_enabled=lora_context.adapter_enabled,
            fully_sharded=lora_context.fully_sharded,
            token_lora_mapping=lora_per_row,
            group_list=group_list,
            combined_idx=combined_idx,
        )
    if combined_idx.numel() == 0:
        return None
    return lora_context.punica_wrapper.add_lora_fused_moe(
        y=gate_up_out,
        x=hidden_states,
        lora_a_stacked=lora_context.w13_lora_a_stacked,
        lora_b_stacked=lora_context.w13_lora_b_stacked,
        expert_ids=None,
        adapter_enabled=lora_context.adapter_enabled,
        fully_sharded=lora_context.fully_sharded,
        group_list=group_list,
        group_list_type=group_list_type,
        combined_idx=combined_idx,
    )


def moe_lora_apply_w2(
    lora_context,
    *,
    down_out,
    silu_out,
    lora_routing,
    group_list=None,
    group_list_type=1,
):
    """Add the w2 LoRA delta into ``down_out`` (in place), after the down GMM.

    Reuses the per-row routing computed by ``moe_lora_apply_w13``; ``silu_out``
    is the activation output that fed the base down GMM.
    """
    # AG path: the stash holds the single-pass combined index (built by the
    # caller or by w13); A2A path: recover the routing pair and build here.
    combined_idx = getattr(lora_context, "combined_lora_idx", None)
    if combined_idx is not None and combined_idx.numel() != down_out.shape[0]:
        combined_idx = None  # stale stash: fall through to the routing path
    expert_per_row = token_lora_mapping = None
    if combined_idx is None:
        if lora_routing is None:
            raise AssertionError(
                "moe_lora_apply_w2 requires either a stashed combined index "
                "or a routing pair"
            )
        expert_per_row, token_lora_mapping = lora_routing
        # EP rank may receive 0 dispatched tokens; skip LoRA to avoid NPU
        # kernel crashes with empty tensors.
        if expert_per_row.numel() == 0:
            return None
    offset = 0
    if lora_context.fully_sharded:
        shard_size = lora_context.w2_lora_b_stacked[0].shape[-2]
        offset = shard_size * lora_context.tp_rank
    # S-LoRA fold: with input-sharded (full-rank) A under TP-only, this
    # rank's partial shrink expands into its own output slice and the MoE
    # runner's final TP all-reduce completes the delta, so the dedicated
    # w2 all-reduce can be skipped. Never under EP: the FS sharding group
    # and the combine collectives do not match there (and FS+EP is blocked
    # upstream anyway).
    partial_expand = (
        lora_context.fully_sharded
        and getattr(lora_context, "tp_size", 1) > 1
        and not getattr(lora_context, "use_ep", False)
    )
    delta = lora_context.punica_wrapper.add_lora_fused_moe(
        y=down_out,
        x=silu_out,
        lora_a_stacked=lora_context.w2_lora_a_stacked,
        lora_b_stacked=lora_context.w2_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        fully_sharded=lora_context.fully_sharded,
        partial_expand=partial_expand,
        offset=offset,
        token_lora_mapping=token_lora_mapping,
        group_list=group_list,
        group_list_type=group_list_type,
        combined_idx=combined_idx,
    )
    if delta is not None:
        down_out += delta
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
        return lora_context

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
        self.base_layer.set_lora_context(self._build_lora_context())


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
