# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Select load-time BF16 conversion for Mistral 4 and its EAGLE draft."""

import json

from vllm.config.model import ModelConfig
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.quantization import utils as quantization_utils
from vllm_ascend.quantization.mistral_fp8 import (
    get_mistral_quantization_config,
    is_mistral4_config,
    is_mistral_eagle_config,
    is_per_tensor_fp8,
)

_ORIGINAL_VERIFY_QUANTIZATION = ModelConfig._verify_quantization
_ORIGINAL_AUTO_DETECT_QUANTIZATION = quantization_utils.maybe_auto_detect_quantization


def _checkpoint_quantization_config(config: ModelConfig):
    quant_config = get_mistral_quantization_config(config)
    if quant_config is not None:
        return quant_config

    # An inner language ModelConfig may omit the outer quantization config.
    # Use the same local/Hub resolver as Ascend's quantization auto-detector.
    model = getattr(config, "model", None)
    if model is None:
        return None
    filename = "params.json" if is_mistral_eagle_config(config) else "config.json"
    path = quantization_utils.get_model_file(model, filename, revision=getattr(config, "revision", None))
    if path is None:
        return None
    try:
        with path.open(encoding="utf-8") as config_file:
            checkpoint_config = json.load(config_file)
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(checkpoint_config, dict):
        return None
    if filename == "config.json":
        return checkpoint_config.get("quantization_config")
    native_quant = checkpoint_config.get("quantization")
    if (
        isinstance(native_quant, dict)
        and native_quant.get("qformat_weight") == "fp8_e4m3"
        and native_quant.get("qscheme_act", "TENSOR") == "TENSOR"
    ):
        return {"quant_method": "fp8", "activation_scheme": "static"}
    return None


def _should_dequantize(config: ModelConfig) -> bool:
    if current_platform.device_name != "npu":
        return False
    if not (is_mistral4_config(config) or is_mistral_eagle_config(config)):
        return False
    # Preserve upstream handling of explicit, incompatible quantization choices.
    if getattr(config, "quantization", None) not in (None, "fp8"):
        return False
    return is_per_tensor_fp8(_checkpoint_quantization_config(config))


def _patched_verify_quantization(self: ModelConfig) -> None:
    if _should_dequantize(self):
        # v0.30 discovers FP8 from model_arch_config inside its verifier. Choose
        # the model-specific BF16 loader before that discovery; do not mutate
        # the process-wide platform verifier while another config is built.
        self.quantization = None
        logger.info("Loading Mistral per-tensor FP8 checkpoint as dequantized BF16 on NPU.")
        return
    return _ORIGINAL_VERIFY_QUANTIZATION(self)


def _patched_maybe_auto_detect_quantization(vllm_config) -> None:
    if _should_dequantize(vllm_config.model_config):
        vllm_config.model_config.quantization = None
        vllm_config.quant_config = None
        return
    return _ORIGINAL_AUTO_DETECT_QUANTIZATION(vllm_config)


ModelConfig._verify_quantization = _patched_verify_quantization
quantization_utils.maybe_auto_detect_quantization = _patched_maybe_auto_detect_quantization
