# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Load-time BF16 conversion for Mistral's per-tensor FP8 checkpoints.

This module deliberately has no vLLM or NPU imports. Fused HF expert tensors
must be unpacked into individual projection views before entering the iterator,
so a worker never materializes a BF16 copy of an entire expert stack.
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

FP8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    if hasattr(torch, name)
)
DEQUANT_ROWS_PER_STEP = 1024
MISTRAL_EAGLE_ARCHITECTURE = "EagleMistralLarge3ForCausalLM"


def _field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_candidates(config: Any) -> tuple[Any, ...]:
    hf_config = _field(config, "hf_config")
    return (
        _field(config, "model_arch_config"),
        _field(config, "hf_text_config"),
        hf_config,
        _field(hf_config, "text_config"),
    )


def is_mistral4_config(config: Any) -> bool:
    return any(_field(candidate, "model_type") == "mistral4" for candidate in _config_candidates(config))


def is_mistral_eagle_config(config: Any) -> bool:
    # Architecture identifies the draft before and after SpeculativeConfig
    # wraps its DeepSeek-compatible config as model_type="eagle". Requiring a
    # local params.json here would incorrectly exclude Hugging Face repo IDs.
    return any(
        MISTRAL_EAGLE_ARCHITECTURE in (_field(candidate, "architectures") or ())
        for candidate in _config_candidates(config)
    )


def get_mistral_quantization_config(config: Any) -> Any:
    for candidate in _config_candidates(config):
        quant_config = _field(candidate, "quantization_config")
        if quant_config is not None:
            return quant_config
    return None


def is_per_tensor_fp8(quant_config: Any) -> bool:
    return (
        _field(quant_config, "quant_method") == "fp8"
        and _field(quant_config, "weight_block_size") is None
        and _field(quant_config, "activation_scheme", "static") in ("static", "dynamic")
    )


def dequantize_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Resolve one projection, with a bounded FP32 multiplication buffer.

    Each HF fused expert has one scalar scale; its gate/up views share that
    scalar. Native Mistral projections may have different gate/up scales and
    are resolved independently before the ordinary loader merges them.
    """
    if weight.dtype not in FP8_DTYPES:
        raise ValueError(f"Expected a Mistral FP8 weight, got {weight.dtype}.")
    if weight.ndim != 2 or scale.numel() != 1:
        raise ValueError(
            "Expected a 2D Mistral FP8 projection and a scalar scale, "
            f"got weight={tuple(weight.shape)}, scale={tuple(scale.shape)}. "
            "Unpack fused experts before BF16 conversion."
        )
    scale_fp32 = scale.to(device=weight.device, dtype=torch.float32).reshape(())
    resolved = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
    for start in range(0, weight.shape[0], DEQUANT_ROWS_PER_STEP):
        end = min(start + DEQUANT_ROWS_PER_STEP, weight.shape[0])
        values = weight[start:end].to(dtype=torch.float32)
        values.mul_(scale_fp32)
        resolved[start:end].copy_(values)
    return resolved


def _weight_name_for_scale(name: str) -> str | None:
    if name.endswith("_scale_inv"):
        return name[: -len("_scale_inv")]
    if name.endswith(".weight_scale"):
        return name[: -len("_scale")]
    if name.endswith(".qscale_weight"):
        return name[: -len(".qscale_weight")] + ".weight"
    return None


def fp8_dequant_weight_iter(
    weights: Iterable[tuple[str, torch.Tensor]],
    checkpoint_label: str = "Mistral 4",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pair either scale order, rejecting missing or duplicate weight metadata.

    Pending values remain in their source dtype. Converted matrices are yielded
    immediately and are never kept in the pending dictionaries. BF16 modules
    without FP8 metadata (norms, router, embeddings, vision) pass through.
    """
    pending_weights: dict[str, torch.Tensor] = {}
    pending_scales: dict[str, torch.Tensor] = {}
    seen_weights: set[str] = set()
    seen_scales: set[str] = set()
    float_weights: set[str] = set()

    for name, tensor in weights:
        if name.endswith(("activation_scale", ".input_scale", ".qscale_act")):
            # Static FP8 activation scales are unused by BF16 execution.
            continue

        weight_name = _weight_name_for_scale(name)
        if weight_name is not None:
            if weight_name in seen_scales:
                raise ValueError(f"Duplicate {checkpoint_label} FP8 scale for {weight_name}.")
            if weight_name in float_weights:
                raise ValueError(f"Unexpected {checkpoint_label} FP8 scale for non-FP8 weight {weight_name}.")
            seen_scales.add(weight_name)
            pending_weight = pending_weights.pop(weight_name, None)
            if pending_weight is None:
                pending_scales[weight_name] = tensor
                continue
            yield weight_name, dequantize_fp8_weight(pending_weight, tensor)
            continue

        if name in seen_weights:
            raise ValueError(f"Duplicate {checkpoint_label} weight {name}.")
        seen_weights.add(name)
        if tensor.dtype not in FP8_DTYPES:
            if name in pending_scales:
                raise ValueError(f"Unexpected {checkpoint_label} FP8 scale for non-FP8 weight {name}.")
            float_weights.add(name)
            yield name, tensor
            continue

        scale = pending_scales.pop(name, None)
        if scale is None:
            pending_weights[name] = tensor
            continue
        yield name, dequantize_fp8_weight(tensor, scale)

    if pending_weights or pending_scales:
        raise ValueError(
            f"Unpaired {checkpoint_label} FP8 weight/scale tensors: "
            f"pending_weights={sorted(pending_weights)}, pending_scales={sorted(pending_scales)}"
        )
