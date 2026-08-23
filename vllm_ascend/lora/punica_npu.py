# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

MOE_LORA_FUSED_BGMV_SUPPORTED_RANKS = frozenset((8, 16, 32, 64))
MOE_LORA_FUSED_BGMV_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
MOE_LORA_FUSED_BGMV_MIN_DIM = 1
MOE_LORA_FUSED_BGMV_MAX_DIM = 16384


# The platforms that are compatible with the PyTorch-native implementation can
# inherit this class
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
        self._bgmv_uses_torch_ops = get_ascend_device_type() == AscendDeviceType._310P or (
            self.lora_config is not None and self.lora_config.max_lora_rank >= 128
        )
        if self._bgmv_uses_torch_ops:
            moe_lora_bgmv_fused = None
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
                moe_lora_bgmv_fused,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_shrink = bgmv_shrink
        self.moe_lora_bgmv_fused = moe_lora_bgmv_fused
        self.sgmv_expand = sgmv_expand
        self.sgmv_expand_slice = sgmv_expand_slice
        self.sgmv_shrink = sgmv_shrink

    def update_metadata(
        self,
        mapping,
        lora_index_to_id,
        max_loras,
        vocab_size,
        **kwargs,
    ) -> None:
        super().update_metadata(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            **kwargs,
        )
        # PunicaWrapperBase computes this only for prefill. Decode must also
        # choose between the active-LoRA and base-only quantized MoE paths.
        self.no_lora = not any(lora_id > 0 for lora_id in mapping.index_mapping)

    def _shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def _shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        self.bgmv_shrink(x, w_t_all, y, self._get_token_lora_indices(x), scale)

    def _expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            add_inputs,
        )

    def _expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        self.bgmv_expand(x, w_t_all, y, self._get_token_lora_indices(x), add_inputs)

    def _expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand_slice(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        self.bgmv_expand_slice(
            x,
            w_t_all,
            y,
            self._get_token_lora_indices(x),
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _get_token_lora_indices(self, x: torch.Tensor) -> torch.Tensor:
        return torch.narrow(self._token_lora_indices, 0, 0, x.size(0))

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool = True,
    ):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all`
        computation, which is suitable for the
        GEMM of lora'b.
        """

        expand_slice_fun: Callable = self._expand_slice_prefill if self.is_prefill else self._expand_slice_decode
        expand_slice_fun(y, x, w_t_all, y_offset, y_slice_size, add_inputs)

    def _apply_shrink(self, y: torch.Tensor, x: torch.Tensor, w_t_all: torch.Tensor, scale: float):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        shrink_fun: Callable = self._shrink_prefill if self.is_prefill else self._shrink_decode
        shrink_fun(y, x, w_t_all, scale)
        y = y.view_as(y_org)

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
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.

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
        # TODO fuse these kernels
        for slice_idx in range(len(lora_a_stacked)):
            self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx], scale)

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
        offset_left = offset_start
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_embedding(
        self, y: torch.Tensor, x: torch.Tensor, lora_b_stacked: torch.Tensor, add_inputs: bool = True, **kwargs
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        # Embedding layer only need expand op
        expand_fun: Callable = self._expand_prefill if self.is_prefill else self._expand_decode
        x = x.to(torch.float32)
        expand_fun(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
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
                    ).squeeze(0)+lora_bias_stacked[i]

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
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
        combined_indices: torch.Tensor | None = None,
    ) -> None:
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
        """
        del sorted_token_ids, num_tokens_post_padded, max_lora_rank
        del shrink_config, expand_config
        assert top_k_num == 1, "Ascend MoE LoRA v1 expects pre-expanded rows (top_k_num=1)."
        x2d = x.view(-1, x.shape[-1])
        y2d = y.view(-1, y.shape[-1])
        if not lora_a_stacked or len(lora_a_stacked) != len(lora_b_stacked):
            raise ValueError("MoE LoRA requires the same nonzero number of A and B slices.")
        if any(a.ndim != 4 or b.ndim != 4 for a, b in zip(lora_a_stacked, lora_b_stacked)):
            raise ValueError("MoE LoRA A and B weights must be 4D tensors.")
        weight_group_shape = lora_a_stacked[0].shape[:-2]
        if any(size <= 0 for size in weight_group_shape):
            raise ValueError("MoE LoRA weight group dimensions must be positive.")
        num_weight_groups = weight_group_shape[0] * weight_group_shape[1]
        num_experts = lora_a_stacked[0].shape[1]
        if combined_indices is not None:
            if expert_ids is not None or token_lora_mapping is not None:
                raise ValueError("combined_indices cannot be mixed with unfused routing metadata.")
            if combined_indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("combined_indices must be int32 or int64.")
            combined_idx = combined_indices.view(-1).contiguous()
        else:
            if expert_ids is None:
                raise ValueError("expert_ids are required when combined_indices are not provided.")
            if token_lora_mapping is None:
                token_lora_mapping = self.token_lora_indices
            if expert_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("MoE LoRA expert_ids must be int32 or int64.")
            if token_lora_mapping.dtype not in (torch.int32, torch.int64):
                raise TypeError("MoE LoRA token_lora_mapping must be int32 or int64.")
            if expert_ids.device != x2d.device or token_lora_mapping.device != x2d.device:
                raise ValueError("MoE LoRA routing metadata must be on the same device as x.")
            if adapter_enabled.device != x2d.device:
                raise ValueError("MoE LoRA adapter_enabled must be on the same device as x.")
            if adapter_enabled.ndim != 1 or adapter_enabled.numel() == 0:
                raise ValueError("MoE LoRA adapter_enabled must be a nonempty 1D tensor.")
            expert_idx = expert_ids.reshape(-1).to(torch.long)
            lora_idx = token_lora_mapping.reshape(-1)
            if expert_idx.numel() != x2d.shape[0] or lora_idx.numel() != x2d.shape[0]:
                raise ValueError("MoE LoRA routing metadata size must match the number of rows.")
            valid_lora = (lora_idx >= 0) & (lora_idx < adapter_enabled.numel())
            valid_expert = (expert_idx >= 0) & (expert_idx < num_experts)
            lora_idx_safe = lora_idx.clamp(min=0, max=adapter_enabled.numel() - 1)
            enabled = valid_lora & valid_expert & adapter_enabled[lora_idx_safe].bool()
            combined_idx = torch.where(
                enabled,
                lora_idx_safe * num_experts + expert_idx,
                torch.full_like(lora_idx, -1),
            ).contiguous()
        if combined_idx.numel() != x2d.shape[0]:
            raise ValueError(
                "MoE LoRA routing size mismatch: "
                f"got {combined_idx.numel()} indices for {x2d.shape[0]} rows."
            )
        bgmv_uses_torch_ops = getattr(self, "_bgmv_uses_torch_ops", False)
        topk_weights_fp32 = None
        if mul_routed_weight and topk_weights is None:
            raise ValueError("topk_weights are required when mul_routed_weight is enabled.")
        if mul_routed_weight:
            if topk_weights.numel() != x2d.shape[0]:
                raise ValueError(
                    "MoE LoRA topk_weights size must match the number of rows."
                )
            if topk_weights.device != x2d.device:
                raise ValueError("MoE LoRA topk_weights must be on the same device as x.")
            if not topk_weights.is_floating_point():
                raise TypeError("MoE LoRA topk_weights must have a floating-point dtype.")
        if x2d.shape[0] != y2d.shape[0]:
            raise ValueError("MoE LoRA x and y row counts must match.")
        if combined_idx.device != x2d.device:
            raise ValueError("MoE LoRA routing indices must be on the same device as x.")
        if not bgmv_uses_torch_ops:
            if x2d.dtype not in MOE_LORA_FUSED_BGMV_SUPPORTED_DTYPES or y2d.dtype != x2d.dtype:
                raise TypeError("MoE LoRA x and y must have the same float16 or bfloat16 dtype.")
            if not x2d.is_contiguous() or not y2d.is_contiguous():
                raise ValueError("Native MoE LoRA x and y must be contiguous.")

        slice_specs = []
        tp_world_size = None
        final_offset = offset
        for slice_idx, (a, b) in enumerate(zip(lora_a_stacked, lora_b_stacked)):
            local_rank = a.shape[-2]
            full_rank = b.shape[-1]
            out_size = b.shape[-2]
            if not bgmv_uses_torch_ops and (a.dtype != x2d.dtype or b.dtype != x2d.dtype):
                raise TypeError(
                    "MoE LoRA x, y, lora_a and lora_b must have the same "
                    f"float16 or bfloat16 dtype; slice {slice_idx} does not."
                )
            if full_rank <= 0:
                raise ValueError(f"MoE LoRA B rank must be positive; slice {slice_idx} has rank {full_rank}.")
            if not bgmv_uses_torch_ops and full_rank not in MOE_LORA_FUSED_BGMV_SUPPORTED_RANKS:
                raise ValueError(
                    "MoE LoRA B rank must be one of 8, 16, 32 or 64; "
                    f"slice {slice_idx} has rank {full_rank}."
                )
            if local_rank <= 0:
                raise ValueError(f"MoE LoRA A rank must be positive; slice {slice_idx} has rank {local_rank}.")
            if fully_sharded and (local_rank > full_rank or full_rank % local_rank != 0):
                raise ValueError(
                    "MoE LoRA fully_sharded rank mismatch: "
                    f"A rank {local_rank} must not exceed and must divide B rank {full_rank}."
                )
            if fully_sharded and local_rank < full_rank:
                if tp_world_size is None:
                    tp_world_size = get_tensor_model_parallel_world_size()
                if local_rank * tp_world_size != full_rank:
                    raise ValueError(
                        "MoE LoRA fully_sharded rank mismatch: "
                        f"A rank {local_rank} across TP world size {tp_world_size} "
                        f"does not reconstruct B rank {full_rank}."
                    )
            if not fully_sharded and local_rank != full_rank:
                raise ValueError(
                    "MoE LoRA rank mismatch without fully_sharded: "
                    f"A projection has rank {local_rank}, but LoRA B expects rank {full_rank}."
                )
            if a.shape[-1] != x2d.shape[1]:
                raise ValueError(
                    f"MoE LoRA A input dimension for slice {slice_idx} must match x."
                )
            if a.shape[:-2] != b.shape[:-2]:
                raise ValueError(
                    f"MoE LoRA A and B weight groups for slice {slice_idx} must match."
                )
            if a.shape[:-2] != weight_group_shape:
                raise ValueError("MoE LoRA weight groups must match across all slices.")
            if a.shape[1] != num_experts:
                raise ValueError(
                    f"MoE LoRA expert count for slice {slice_idx} must match the first slice."
                )
            if out_size <= 0 or final_offset < 0 or out_size > y2d.shape[1] - final_offset:
                raise ValueError(f"MoE LoRA output slice {slice_idx} is out of range.")
            if a.device != x2d.device or b.device != x2d.device or y2d.device != x2d.device:
                raise ValueError(f"MoE LoRA tensors for slice {slice_idx} must be on the same device.")
            if not bgmv_uses_torch_ops and (not a.is_contiguous() or not b.is_contiguous()):
                raise ValueError(f"Native MoE LoRA weights for slice {slice_idx} must be contiguous.")

            if bgmv_uses_torch_ops:
                a_flat = a.reshape(-1, local_rank, a.shape[-1])
                b_flat = b.reshape(-1, out_size, full_rank)
            else:
                a_flat = a.view(-1, local_rank, a.shape[-1])
                b_flat = b.view(-1, out_size, full_rank)

            fused_dims_supported = (
                MOE_LORA_FUSED_BGMV_MIN_DIM <= x2d.shape[1] <= MOE_LORA_FUSED_BGMV_MAX_DIM
                and MOE_LORA_FUSED_BGMV_MIN_DIM <= out_size <= MOE_LORA_FUSED_BGMV_MAX_DIM
            )
            use_fused_bgmv = (
                not bgmv_uses_torch_ops
                and getattr(self, "moe_lora_bgmv_fused", None) is not None
                and not fully_sharded
                and not mul_routed_weight
                and local_rank == full_rank
                and fused_dims_supported
            )
            if not bgmv_uses_torch_ops and not use_fused_bgmv:
                if x2d.shape[1] <= local_rank:
                    raise ValueError(
                        "Native split MoE LoRA input dimension must be greater than A rank; "
                        f"slice {slice_idx} has H={x2d.shape[1]} and rank={local_rank}."
                    )
                if out_size < full_rank:
                    raise ValueError(
                        "Native split MoE LoRA output dimension must be at least B rank; "
                        f"slice {slice_idx} has O={out_size} and rank={full_rank}."
                    )

            slice_specs.append((a_flat, b_flat, local_rank, full_rank, out_size, use_fused_bgmv))
            final_offset += out_size

        if x2d.shape[0] == 0:
            return

        if mul_routed_weight:
            assert topk_weights is not None
            topk_weights_fp32 = topk_weights.reshape(-1).to(dtype=torch.float32).contiguous()

        split_indices = combined_idx
        torch_invalid_rows = None
        if bgmv_uses_torch_ops:
            # Torch indexing maps -1 to the last weight and raises on a high
            # index; native kernels treat either case as a disabled row.
            torch_invalid_rows = (combined_idx < 0) | (combined_idx >= num_weight_groups)
            split_indices = combined_idx.clamp(min=0, max=num_weight_groups - 1).contiguous()

        cur_offset = offset
        for a_flat, b_flat, local_rank, full_rank, out_size, use_fused_bgmv in slice_specs:
            if use_fused_bgmv:
                self.moe_lora_bgmv_fused(
                    x2d,
                    a_flat,
                    b_flat,
                    combined_idx,
                    y2d,
                    cur_offset,
                    out_size,
                    1.0,
                )
                cur_offset += out_size
                continue

            # bgmv_shrink writes fp32 (its Y_T); bgmv_expand reads fp32
            # (its X_T), so the shrink buffer is fp32.
            shrink_out = torch.zeros(
                (x2d.shape[0], local_rank),
                dtype=torch.float32,
                device=x2d.device,
            )

            self.bgmv_shrink(x2d, a_flat, shrink_out, split_indices, 1.0)
            if torch_invalid_rows is not None:
                shrink_out.masked_fill_(torch_invalid_rows.view(-1, 1), 0.0)

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

            delta = shrink_out
            if mul_routed_weight:
                assert topk_weights_fp32 is not None
                delta = shrink_out * topk_weights_fp32.view(-1, 1)

            self.bgmv_expand_slice(delta, b_flat, y2d, split_indices, cur_offset, out_size, add_inputs=True)
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
        Applies lora  specifically for LogitsProcessorWithLoRA.

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

        indices = torch.narrow(self._sampler_indices, 0, 0, x.size(0))

        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)

        y = y.view_as(y_org)
