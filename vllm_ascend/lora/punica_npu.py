# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
import os
import torch
import torch_npu
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.logger import logger

from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# in-tree fused kernel for decode
ENABLE_ADD_LORA_KERNEL = True



def build_combined_lora_idx(
    token_lora_mapping: torch.Tensor,
    expert_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Per-row gather index into the flattened [max_loras * num_experts, ...]
    LoRA weight stacks: ``combined_idx[row] = lora_id[row] * num_experts +
    expert_id[row]``, or -1 when the row has no active adapter. Identical to
    the block historically inlined in add_lora_fused_moe; extracted so the
    w13 and w2 applies of one MoE layer can share a single build.
    """
    expert_idx = expert_ids.view(-1).to(torch.long)
    lora_idx_safe = token_lora_mapping.clamp(min=0)
    enabled = (token_lora_mapping >= 0) & adapter_enabled[lora_idx_safe].bool()
    return torch.where(
        enabled,
        lora_idx_safe * num_experts + expert_idx,
        torch.full_like(token_lora_mapping, -1),
    ).contiguous()


class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int, device: torch.device | str, **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        refresh_all_lora_classes()
        self.lora_config = kwargs.get("lora_config")
        if get_ascend_device_type() == AscendDeviceType._310P or (
            self.lora_config is not None and self.lora_config.max_lora_rank >= 128
        ):
            from vllm.lora.ops.torch_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        else:
            from vllm_ascend.lora.lora_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_shrink = bgmv_shrink
        self.sgmv_expand = sgmv_expand
        self.sgmv_expand_slice = sgmv_expand_slice
        self.sgmv_shrink = sgmv_shrink

        self._use_gmm_shrink_cpu = torch.tensor(False, dtype=torch.bool)
        self._use_gmm_expand_cpu = torch.tensor(False, dtype=torch.bool)
        self._no_lora_cpu = torch.tensor(True, dtype=torch.bool)
        self._prefill_meta_ready = False
        self._use_moe_gmm_cpu = torch.tensor(False, dtype=torch.bool)

        # switch for fused addlora kernel
        self._use_add_lora_cpu = torch.tensor(ENABLE_ADD_LORA_KERNEL, dtype=torch.bool)
        # In the previous punica_npu code there were no switch between prefill/decode,
        # everything was processed via SGMV kernels. GroupGEMM on prefill is generally
        # much faster, but kernel launch takes way to much time for using it for decode,
        # so we temporarely use the serving tokens number switch to distinguish
        # prefill GroupGEMM and decode BGMV/fused addlora
        self.gmm_threshold = (
            int(os.environ["LORA_GMM_THRESHOLD"])
            if "LORA_GMM_THRESHOLD" in os.environ
            else max_batches
        )

        from vllm.config import get_current_vllm_config

        # Speculative decoding multiplies the decode token count by
        # (1 + num_speculative_tokens); the threshold must track this or
        # spec-decode batches exceed it and trip the graph-capture assert.
        _spec_config = get_current_vllm_config().speculative_config
        if _spec_config is not None and _spec_config.num_speculative_tokens:
            self.gmm_threshold *= 1 + _spec_config.num_speculative_tokens

        max_capture = get_current_vllm_config().compilation_config.max_cudagraph_capture_size
        assert max_capture is None or self.gmm_threshold >= max_capture, (
            f"LORA_GMM_THRESHOLD ({self.gmm_threshold}) must be >= "
            f"max_cudagraph_capture_size ({max_capture}); otherwise a batch that takes "
            "the gmm branch could also be cudagraph-captured, baking the host lora_id "
            "weight address into the replayed graph."
        )
        logger.warning(
            "LoRA gmm kernels enabled for batches with > %d tokens (LORA_GMM_THRESHOLD); "
            "batches at or below this size use the bgmv path.",
            self.gmm_threshold,
        )

    def update_metadata(
        self,
        mapping,
        lora_index_to_id,
        max_loras,
        vocab_size,
        **kwargs,
    ) -> None:
        # TODO: when all prefill requests will be served with GMM,
        # switch to self._update_base_metadata
        super().update_metadata(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            **kwargs,
        )

        token_num = len(mapping.index_mapping)
        gmm_enabled = token_num > self.gmm_threshold

        if gmm_enabled:
            # TODO: two metadata calls issue
            self._update_prefill_metadata(self.token_lora_indices)
            self.no_lora = bool(self.no_lora)
        else:
            self.no_lora = not any(mapping.index_mapping)

        self._use_gmm_shrink_cpu.fill_(gmm_enabled)
        self._use_gmm_expand_cpu.fill_(gmm_enabled)
        self._no_lora_cpu.fill_(self.no_lora)

        self._use_moe_gmm_cpu.fill_(gmm_enabled)

        self.is_prefill = bool(getattr(mapping, "is_prefill", True))
        self._prefill_meta_ready = gmm_enabled

    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.
        Prefill/decode kernel is choosen within native op.
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        y_views = [y[i].view(-1, y[i].shape[-1]) for i in range(len(lora_a_stacked))]
        _, seq_len, lora_indices, _, _, _ = self.prefill_metadata
        torch.ops._C_ascend.add_lora_shrink(
            y_views, x, list(lora_a_stacked),
            lora_indices, seq_len, self.token_lora_indices,
            scale, self._use_gmm_shrink_cpu, self._no_lora_cpu,
        )

    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (Tuple[int, ...]): Every slice's size
            offset_start (int): The starting position of y, defaults to 0
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        _, seq_len, lora_indices, _, _, _ = self.prefill_metadata
        torch.ops._C_ascend.add_lora_expand(
            y, [x[i] for i in range(len(lora_b_stacked))], list(lora_b_stacked),
            lora_indices, seq_len, self.token_lora_indices,
            list(output_slices), offset_start, add_inputs,
            self._use_gmm_expand_cpu, self._no_lora_cpu,
        )
        y = y.view_as(y_org)

    def add_lora_embedding(
        self, y: torch.Tensor, x: torch.Tensor, lora_b_stacked: torch.Tensor, add_inputs: bool = True, **kwargs
    ) -> None:
        """
        Applies lora specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """
        if self.no_lora:
            return
        x = x.to(torch.float32)
        if self._prefill_meta_ready:
            self.sgmv_expand(x, lora_b_stacked, y, *self.prefill_metadata, add_inputs)
        else:
            self.bgmv_expand(x, lora_b_stacked, y, self.token_lora_indices, add_inputs)

    def add_lora(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs: bool = True,
        **kwargs,
    ) -> None:
        """
        Performs the full `y += (x @ lora_a) @ lora_b * scale` LoRA apply in a
        single opaque op. Inside the op: in-tree fused kernel (decode and
        small prefill, option-gated) or gmm; decode fallback -> bgmv
        shrink+expand pair. add_inputs=False is emulated by zeroing the
        target slice first (the fused kernel always accumulates).
        """
        y2d = y.view(-1, y.shape[-1])
        x2d = x.view(-1, x.shape[-1])
        _, seq_len, lora_indices, _, _, _ = self.prefill_metadata
        torch.ops._C_ascend.add_lora(
            y2d, x2d, list(lora_a_stacked), list(lora_b_stacked),
            lora_indices, seq_len, self.token_lora_indices,
            list(output_slices), offset_start, scale, add_inputs,
            self._use_gmm_shrink_cpu, self._no_lora_cpu, self._use_add_lora_cpu,
        )

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        add_inputs: bool = True,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0) @ lora_a_stacked[
                    indices[i], layer_idx, :, :] @ lora_b_stacked[
                    indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        if buffer is None:
            if ENABLE_ADD_LORA_KERNEL:
                # Fused single-op path: in-tree fused kernel / gmm / bgmv
                # branch inside the opaque op (the C++ side falls back to
                # gmm/bgmv internally for ineligible shapes).
                self.add_lora(
                    y, x, lora_a_stacked, lora_b_stacked, scale, output_slices,
                    add_inputs=add_inputs, **kwargs
                )
                return

            # fused kernel disabled: original two-op path with fp32 buffers
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            buffer = tuple(
                torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device) for _ in range(len(output_slices))
            )
        self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs)

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        *,
        topk_weights: torch.Tensor | None = None,
        sorted_token_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_post_padded: torch.Tensor | None = None,
        max_lora_rank: int = 0,
        top_k_num: int = 1,
        shrink_config=None,
        expand_config=None,
        adapter_enabled: torch.Tensor,
        mul_routed_weight: bool = False,
        fully_sharded: bool = False,
        partial_expand: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
        group_list: torch.Tensor | None = None,
        group_list_type: int = 1,
        combined_idx: torch.Tensor | None = None,
        overwrite: bool = False,
        phase: str = "full",
        shrink_buffers: list | None = None,
    ) -> list | torch.Tensor | None:
        """
        Ascend-native fused MoE LoRA (v2): static-shape per-row gather via the
        same bgmv_shrink/bgmv_expand AscendC kernels (csrc/kernels/bgmv_*.cpp)
        used by the dense Linear LoRA layers, instead of grouping rows by a
        data-dependent ``torch.unique`` over active LoRA ids. The previous
        ``torch.unique``/``nonzero`` version produced output whose *shape*
        depended on tensor values, which ACL Graph capture cannot record
        (it failed with an `aclnnUnique2` error as soon as `enforce_eager`
        was turned off) -- every tensor below has a shape that depends only
        on input shapes, never on values, so this stays graph-capturable.

        Rows are already one-token-per-row (top_k_num=1). Each row needs the
        LoRA slot for (lora_id, expert_id), so we fold both into a single
        gather index into a ``[max_loras * num_experts, ...]`` view of the
        existing per-(lora, expert) weight stacks:
            combined_idx[row] = lora_id[row] * num_experts + expert_id[row]
        or -1 when the row has no active adapter, mirroring the -1 sentinel
        ``PunicaWrapperBase.token_lora_indices`` already uses. bgmv_shrink/
        bgmv_expand skip any row whose index is negative (leaving the
        zero-initialized shrink buffer / unmodified ``y`` in place), so
        inactive rows get a zero delta for free -- no Python-level branching
        needed.

        overwrite=True writes the LoRA delta into ``y`` without accumulating
        (used by the aux-stream overlap path: ``y`` is then an isolated delta
        buffer the caller zeroes and folds into the GEMM output afterwards).
        Only supported on the decode fused/bgmv path.

        partial_expand: fully-sharded w2-style apply only (input-sharded A,
        full rank). Skips the dedicated shrink all-reduce and expands this
        rank's partial sum into its tp_rank output slice; the MoE runner's
        final TP all-reduce completes the delta
        (sum_r(z_r @ B_r^T) == z_full @ B_r^T after reduction). The caller
        must guarantee the final reduction covers the expanded tensor.

        phase: split execution of the two-op (group_list) branch so the aux
        stream can overlap the shrink with the base GEMM:
          "shrink"  -- run add_lora_shrink only and return the buffers (no
                       collective); the caller later passes them back with
          "expand"  -- run the cross-rank collective (unless folded) and
                       add_lora_expand into y.
        Only the group_list branch supports splitting (the fused single-op
        path cannot host the collective between shrink and expand anyway).
        """
        del sorted_token_ids, num_tokens_post_padded, max_lora_rank
        del shrink_config, expand_config
        assert top_k_num == 1, "Ascend MoE LoRA v1 expects pre-expanded rows (top_k_num=1)."
        if token_lora_mapping is None and combined_idx is None:
            # Only needed for the index build; the prebuilt-combined path
            # does not touch it (and bare test wrappers may not carry it).
            token_lora_mapping = self.token_lora_indices

        x2d = x.view(-1, x.shape[-1])
        y2d = y.view(-1, y.shape[-1])
        if combined_idx is None:
            if expert_ids is None or token_lora_mapping is None:
                raise AssertionError(
                    "add_lora_fused_moe requires either combined_idx or "
                    "(expert_ids, token_lora_mapping)"
                )
            # moe_lora_apply_w13/w2 share one routing per layer
            combined_idx = build_combined_lora_idx(
                token_lora_mapping,
                expert_ids,
                adapter_enabled,
                lora_a_stacked[0].shape[1],
            )

        if (
            not fully_sharded
            and ENABLE_ADD_LORA_KERNEL
            and not self._use_moe_gmm_cpu.item()
            and not (mul_routed_weight and topk_weights is not None)
        ):
            # use_gmm is False on this path, so the op's per-sequence gmm branch
            # (invalid for expert-sorted rows) is never taken; single-adapter prefill
            # keeps the grouped path below. fully_sharded is checked first: the
            # fused kernel cannot host the cross-rank collective that fully
            # sharded LoRA needs between shrink and expand.
            a_views = [a.view(-1, 1, a.shape[-2], a.shape[-1]) for a in lora_a_stacked]
            b_views = [b.view(-1, 1, b.shape[-2], b.shape[-1]) for b in lora_b_stacked]
            output_slices_fused = [b.shape[-2] for b in lora_b_stacked]
            torch.ops._C_ascend.add_lora(
                y2d, x2d, a_views, b_views,
                combined_idx, combined_idx, combined_idx,
                output_slices_fused, offset, 1.0, not overwrite,
                self._use_moe_gmm_cpu, self._no_lora_cpu, self._use_add_lora_cpu,
            )
            return

        if overwrite:
            raise ValueError(
                "overwrite=True (delta-buffer mode) is only supported on the "
                "decode fused/bgmv path (AllGather, non-fully-sharded, no "
                "routed-weight fold)"
            )

        if group_list is not None and not mul_routed_weight:
            n_slices = len(lora_a_stacked)
            max_loras = lora_a_stacked[0].shape[0]
            local_rank = lora_a_stacked[0].shape[-2]
            full_rank = lora_b_stacked[0].shape[-1]
            rank_sharded = local_rank != full_rank
            gmm_flag = self._use_moe_gmm_cpu
            # Under fully_sharded the shrink output must survive a cross-rank
            # collective (all_gather/all_reduce) between shrink and expand, so
            # the buffers are fp32; the non-sharded path keeps the historical
            # lora dtype. Both C++ shrink variants fully initialize the buffer
            # they target (gmm: copy_ of the masked result; bgmv: explicit
            # zero_() then scatter), so no zero fill is needed here.
            buf_dtype = torch.float32 if fully_sharded else lora_a_stacked[0].dtype
            if phase == "expand":
                if shrink_buffers is None:
                    raise ValueError('phase="expand" requires shrink_buffers from phase="shrink"')
                buffers = shrink_buffers
            else:
                buffers = [
                    torch.empty(
                        (x2d.shape[0], max_loras * local_rank),
                        dtype=buf_dtype,
                        device=x2d.device,
                    )
                    for _ in range(n_slices)
                ]
                buffers.append(
                    torch.empty(
                        (n_slices, x2d.shape[0], local_rank), dtype=torch.float32, device=x2d.device
                    )
                )
                torch.ops._C_ascend.add_lora_shrink(
                    buffers, x2d, list(lora_a_stacked), combined_idx, group_list, combined_idx,
                    1.0, gmm_flag, self._no_lora_cpu, True, group_list_type,
                )
                if phase == "shrink":
                    return buffers
            output_slices = [b.shape[-2] for b in lora_b_stacked]
            # partial_expand folds the cross-rank completion of a w2-style
            # apply into the MoE runner's final TP all-reduce: each rank
            # expands its own partial shrink (input-sharded A) into its
            # tp_rank output slice, and sum(rank partials) == full delta
            # after the final reduction. Only valid for the non-rank-sharded
            # (w2) case and never under EP, where the FS sharding group and
            # the combine collectives do not match.
            fold_partial = partial_expand and not rank_sharded
            if fully_sharded and not fold_partial and get_tensor_model_parallel_world_size() > 1:
                # The C++ shrink writes the per-slice [T, L*R_local] buffers on
                # the gmm (prefill) branch and the stacked [S, T, R_local] fp32
                # buffer on the bgmv (decode) branch; the collective must
                # target the one that was actually written. Rank shards are
                # contiguous by tp_rank (slice_lora_a), so an all_gather along
                # the last dim concatenates them in the correct order. The gmm
                # per-slice buffer carries L lora blocks per row, so it is
                # viewed [T, L, R_local] first to keep the blocks from
                # interleaving; the stacked bgmv buffer needs no such view.
                use_gmm = bool(gmm_flag.item())
                if use_gmm:
                    if rank_sharded:
                        # w13-style: A is rank-sharded; gather the rank axis.
                        T = x2d.shape[0]
                        for s in range(n_slices):
                            v = buffers[s].view(T, max_loras, local_rank)
                            buffers[s] = tensor_model_parallel_all_gather(
                                v, dim=-1
                            ).reshape(T, max_loras * full_rank)
                    else:
                        # w2-style: A is unsharded; shrink outputs are partial
                        # sums over this rank's input shard.
                        for s in range(n_slices):
                            buffers[s] = tensor_model_parallel_all_reduce(buffers[s])
                else:
                    stacked = buffers[n_slices]
                    if rank_sharded:
                        buffers[n_slices] = tensor_model_parallel_all_gather(
                            stacked, dim=-1
                        )
                    else:
                        buffers[n_slices] = tensor_model_parallel_all_reduce(stacked)
            # the gmm expand cannot accumulate into its own output, so it hands
            # the delta back and the caller folds it; every other branch folds
            # in place and returns None. fully-sharded w2 and partial_expand
            # target a tp_rank slice, so they keep folding.
            covers_y = offset == 0 and sum(output_slices) == y2d.shape[1]
            if covers_y and bool(gmm_flag.item()):
                return torch.ops._C_ascend.add_lora_expand_delta(
                    y2d, buffers, list(lora_b_stacked), group_list, group_list_type,
                )
            torch.ops._C_ascend.add_lora_expand(
                y2d, buffers, list(lora_b_stacked), combined_idx, group_list, combined_idx,
                output_slices, offset, True, gmm_flag, self._no_lora_cpu,
                True, group_list_type,
            )
            return None

        if phase != "full":
            raise ValueError(
                'phase="shrink"/"expand" is only supported on the two-op '
                "(group_list) branch"
            )
        cur_offset = offset
        for slice_idx in range(len(lora_a_stacked)):
            # lora_a_stacked[s]/lora_b_stacked[s]: [max_loras, num_experts, rank, *].
            # Flattening the leading two dims turns "gather by (lora, expert)"
            # into "the plain per-row gather" to reuse bgmv_shrink/bgmv_expand.
            a = lora_a_stacked[slice_idx]
            b = lora_b_stacked[slice_idx]
            local_rank = a.shape[-2]
            full_rank = b.shape[-1]
            out_size = b.shape[-2]
            a_flat = a.view(-1, local_rank, a.shape[-1])

            # bgmv_shrink writes fp32 (its Y_T); bgmv_expand reads fp32
            # (its X_T), so the shrink buffer is fp32.
            shrink_out = torch.zeros(
                (x2d.shape[0], local_rank),
                dtype=torch.float32,
                device=x2d.device,
            )

            self.bgmv_shrink(x2d, a_flat, shrink_out, combined_idx, 1.0)

            if fully_sharded:
                if local_rank == full_rank:
                    shrink_out = tensor_model_parallel_all_reduce(shrink_out)
                else:
                    shrink_out = tensor_model_parallel_all_gather(shrink_out)

            if shrink_out.shape[-1] != full_rank:
                raise ValueError(
                    "MoE LoRA rank mismatch after TP communication: "
                    f"A projection has rank {shrink_out.shape[-1]}, "
                    f"but LoRA B expects rank {full_rank}."
                )
            b_flat = b.view(-1, out_size, full_rank)

            delta = shrink_out
            if mul_routed_weight and topk_weights is not None:
                delta = shrink_out * topk_weights.view(-1, 1)

            self.bgmv_expand_slice(delta, b_flat, y2d, combined_idx, cur_offset, out_size, add_inputs=True)
            cur_offset += out_size

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)

        indices = self._sampler_indices
        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)

        y = y.view_as(y_org)
