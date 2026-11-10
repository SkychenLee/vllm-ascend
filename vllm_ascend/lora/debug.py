# SPDX-License-Identifier: Apache-2.0
"""Opt-in, eager-only diagnostics for the single-slot attention LoRA path."""

from dataclasses import dataclass, field

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger

from vllm_ascend import envs

DEBUG_SAMPLE_COLUMNS = 128


def configure_lora_debug(wrapper) -> None:
    wrapper._ascend_lora_debug_enabled = envs.VLLM_ASCEND_LORA_DEBUG
    if not wrapper._ascend_lora_debug_enabled:
        return
    model_config = get_current_vllm_config().model_config
    if not model_config.enforce_eager:
        raise ValueError("VLLM_ASCEND_LORA_DEBUG=1 requires --enforce-eager; graph replay does not run Python logs.")
    if not wrapper._single_lora_slot:
        raise ValueError("VLLM_ASCEND_LORA_DEBUG requires A2/A3, --max-loras 1 and non-fully-sharded LoRA.")
    logger.warning_once(
        "Ascend LoRA debug is enabled: TP rank 0 logs q_b_proj/kv_b_proj/o_proj. "
        "Sampled output differences synchronize the device; disable debug for performance measurements."
    )
    logger.info(
        "[AscendLoRADebug] event=config enforce_eager=True use_mla=%s model_dtype=%s",
        getattr(model_config, "use_mla", None),
        getattr(model_config, "dtype", None),
    )


def update_lora_debug_metadata(wrapper, mapping, lora_index_to_id) -> None:
    if getattr(wrapper, "_ascend_lora_debug_enabled", False):
        # These are host-side IDs, not device tensors or prompt contents.
        wrapper._ascend_lora_debug_mapping = tuple(mapping.index_mapping)
        wrapper._ascend_lora_debug_slots = tuple(lora_index_to_id)


@dataclass
class LinearLoRADebug:
    prefix: str
    tp_rank: int
    loaded_slots: set[int] = field(default_factory=set)
    seen: set[tuple] = field(default_factory=set)

    def set_slot(self, index: int, loaded: bool) -> None:
        if loaded:
            self.loaded_slots.add(index)
            logger.info("[AscendLoRADebug] event=load layer=%s tp_rank=%d slot=%d", self.prefix, self.tp_rank, index)
        else:
            self.loaded_slots.discard(index)
        self.seen.clear()

    def before(self, layer, x: torch.Tensor, output: torch.Tensor):
        if not is_forward_context_available() or output.numel() == 0:
            return None
        context = get_forward_context()
        if getattr(context, "capturing", False) or getattr(context, "in_profile_run", False):
            return None
        metadata = context.attn_metadata
        if metadata is None:
            return None
        if isinstance(metadata, dict):
            metadata = metadata.get(self.prefix.rsplit(".", 1)[0] + ".attn")
        mapping = getattr(layer.punica_wrapper, "_ascend_lora_debug_mapping", None)
        actual = getattr(metadata, "num_actual_tokens", None)
        if mapping is None or actual is None or not (0 < actual <= len(mapping) and actual <= x.shape[0]):
            logger.warning_once(
                "[AscendLoRADebug] event=call_unmeasured layer=%s: attention/token mapping unavailable; "
                "cannot attribute LoRA activity to this input.",
                self.prefix,
            )
            return None

        # mapping.is_prefill is not the execution phase (vLLM may always set it).
        prefill = getattr(metadata, "num_prefills", 0) > 0
        decode = getattr(metadata, "num_decode_tokens", 0) > 0
        phase = "mixed" if prefill and decode else "prefill" if prefill else "decode" if decode else "unknown"
        token_ids = mapping[:actual]
        adapter_ids = tuple(sorted({adapter_id for adapter_id in token_ids if adapter_id > 0}))
        slots = getattr(layer.punica_wrapper, "_ascend_lora_debug_slots", ())
        active_slots = tuple(slot for slot, adapter_id in enumerate(slots) if adapter_id in adapter_ids)
        loaded_slots = tuple(slot for slot in active_slots if slot in self.loaded_slots)
        key = (phase, adapter_ids, loaded_slots)
        if key in self.seen:
            return None
        self.seen.add(key)
        active_tokens = sum(adapter_id > 0 for adapter_id in token_ids)
        row = next((i for i, adapter_id in enumerate(token_ids) if adapter_id > 0), 0)
        columns = min(DEBUG_SAMPLE_COLUMNS, output.shape[-1])
        logger.info(
            "[AscendLoRADebug] event=enter layer=%s tp_rank=%d phase=%s wrapper=%s "
            "input_shape=%s input_dtype=%s weight_dtype=%s lora_dtype=%s "
            "adapter_ids=%s active_tokens=%d/%d active_slots=%s loaded_slots=%s "
            "no_lora=%s sample_row=%d sample_columns=%d",
            self.prefix,
            self.tp_rank,
            phase,
            type(layer).__name__,
            tuple(x.shape),
            x.dtype,
            layer.base_layer.weight.dtype,
            layer.lora_a_stacked[0].dtype,
            adapter_ids,
            active_tokens,
            actual,
            active_slots,
            loaded_slots,
            layer.punica_wrapper.no_lora,
            row,
            columns,
        )
        # Bound debug memory independently of the prompt length and hidden size.
        return phase, adapter_ids, row, columns, output[row, :columns].detach().clone()

    def after(self, sample, output: torch.Tensor) -> None:
        phase, adapter_ids, row, columns, before = sample
        delta = (output[row, :columns].detach().float() - before.float()).abs().amax().item()
        logger.info(
            "[AscendLoRADebug] event=apply_done layer=%s tp_rank=%d phase=%s adapter_ids=%s "
            "sample_local_delta_abs_max=%.9g (after output dtype rounding, before any TP output reduction)",
            self.prefix,
            self.tp_rank,
            phase,
            adapter_ids,
            delta,
        )


def initialize_linear_lora_debug(layer) -> None:
    layer._ascend_lora_debug = None
    if not envs.VLLM_ASCEND_LORA_DEBUG or layer.tp_rank != 0:
        return
    prefix = layer.base_layer.prefix
    if prefix.rsplit(".", 1)[-1] not in ("q_b_proj", "kv_b_proj", "o_proj"):
        return
    layer._ascend_lora_debug = LinearLoRADebug(prefix, layer.tp_rank)
    logger.info(
        "[AscendLoRADebug] event=wrap layer=%s tp_rank=%d wrapper=%s base=%s weight_dtype=%s lora_dtype=%s",
        prefix,
        layer.tp_rank,
        type(layer).__name__,
        type(layer.base_layer).__name__,
        layer.base_layer.weight.dtype,
        layer.lora_a_stacked[0].dtype,
    )
