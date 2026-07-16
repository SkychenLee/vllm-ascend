import torch
from vllm.logger import logger
from .base import AscendAttentionScheme



def _c8_kv_scale_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Weight loader for dense-attention C8 KV cache scales/offsets."""
    loaded_weight = loaded_weight.squeeze()
    if param.data.shape != loaded_weight.shape:
        param.data = loaded_weight.to(param.dtype).clone()
    else:
        param.data.copy_(loaded_weight)

class AscendC4KVCacheAttentionMethod(AscendAttentionScheme):
    """C4 FP KV cache quantization for dense-attention models (e.g. Qwen3)."""

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Override kv_cache_torch_dtype so Attention.get_kv_cache_spec returns int8 automatically.
        
        # layer.kv_cache_torch_dtype = torch.int8
        
        # Upgrade impl to the C8-specific subclass so the C8 forward path is always used.
        
        # if hasattr(layer, "impl"):
        #     from vllm_ascend.attention.attention_v1 import AscendC8AttentionBackendImpl

        #     layer.impl.__class__ = AscendC8AttentionBackendImpl
        logger.info_once(f"AscendC4KVCacheAttentionMethod create_weights")
        layer.k_cache_scale = torch.nn.Parameter(torch.ones(512, dtype=torch.float32), requires_grad=False)
        layer.k_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        # layer.k_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        # layer.k_cache_offset.weight_loader = _c8_kv_scale_weight_loader
        layer.v_cache_scale = torch.nn.Parameter(torch.ones(512, dtype=torch.float32), requires_grad=False)
        layer.v_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        # layer.v_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        # layer.v_cache_offset.weight_loader = _c8_kv_scale_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.k_cache_scale.data = layer.k_cache_scale.data.flatten()
        # layer.k_cache_offset.data = layer.k_cache_offset.data.flatten()
        layer.v_cache_scale.data = layer.v_cache_scale.data.flatten()
        # layer.v_cache_offset.data = layer.v_cache_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "AscendC8KVCacheAttentionMethod.apply should not be called. "
            "C8 KV cache quantization is handled by the attention backend."
        )
        # logger.info_once(f"query.shape :{query.shape}, key.shape: {key.shape}, value.shape: {value.shape}, layer: {layer}")
        # return layer.impl.forward(query, key, value, kv_cache, attn_metadata, attn_type, scale, output)
