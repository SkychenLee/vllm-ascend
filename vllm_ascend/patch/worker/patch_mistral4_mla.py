# SPDX-License-Identifier: Apache-2.0
"""Keep Mistral EAGLE's 256-dim model weights with 512-dim Ascend MLA caches."""

from dataclasses import replace

from vllm.model_executor.models.mistral_large_3_eagle import EagleMistralLarge3ForCausalLM
from vllm.platforms import current_platform

from vllm_ascend.attention.mla_v1 import AscendMLAAttention, AscendMLAImpl

MISTRAL_KV_RANK = 256
MISTRAL_ROPE_DIM = 64
ASCEND_MLA_CACHE_RANK = 512

_ORIGINAL_EAGLE_INIT = EagleMistralLarge3ForCausalLM.__init__
_ORIGINAL_GET_CACHE_SPEC = AscendMLAAttention.get_kv_cache_spec


def _patched_eagle_init(self, *, vllm_config, prefix: str = "") -> None:
    _ORIGINAL_EAGLE_INIT(self, vllm_config=vllm_config, prefix=prefix)
    if current_platform.device_name != "npu":
        return
    for layer in self.model.modules():
        if not isinstance(layer, AscendMLAAttention):
            continue
        if (layer.kv_lora_rank, layer.qk_rope_head_dim) != (MISTRAL_KV_RANK, MISTRAL_ROPE_DIM):
            continue
        if type(layer.impl) is not AscendMLAImpl:
            raise NotImplementedError("Mistral EAGLE padded MLA requires the dense backend with PCP=1 and DCP=1.")
        layer.impl._validate_mla_compatibility()
        if vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("Mistral EAGLE padded MLA does not yet support disaggregated KV transfer.")
        layer.impl.kv_cache_lora_rank = ASCEND_MLA_CACHE_RANK
        layer.impl.prefill_qk_head_dim = max(layer.qk_nope_head_dim, layer.v_head_dim)
        # Prolog weight absorption assumes the unpadded geometry. The ordinary
        # preprocessing path handles semantic RMSNorm before physical padding.
        layer.impl.enable_mlapo = False


def _patched_get_cache_spec(self, vllm_config):
    spec = _ORIGINAL_GET_CACHE_SPEC(self, vllm_config)
    cache_rank = getattr(self.impl, "kv_cache_lora_rank", self.kv_lora_rank)
    if cache_rank == self.kv_lora_rank:
        return spec
    expected = self.kv_lora_rank + self.qk_rope_head_dim
    if spec.head_size != expected:
        raise ValueError(f"Unexpected Mistral MLA cache head size {spec.head_size}; expected {expected}.")
    return replace(spec, head_size=cache_rank + self.qk_rope_head_dim)


EagleMistralLarge3ForCausalLM.__init__ = _patched_eagle_init
AscendMLAAttention.get_kv_cache_spec = _patched_get_cache_spec
