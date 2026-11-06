# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Mistral HF main model / native EAGLE FP8 loading on Ascend."""

from collections.abc import Iterable

import torch
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.mistral_large_3_eagle import EagleMistralLarge3ForCausalLM
from vllm.platforms import current_platform

from vllm_ascend.quantization.mistral_fp8 import fp8_dequant_weight_iter

_ORIGINAL_EAGLE_LOAD_WEIGHTS = EagleMistralLarge3ForCausalLM.load_weights


def _patched_eagle_load_weights(
    self: EagleMistralLarge3ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    if current_platform.device_name == "npu" and self.quant_config is None:
        # Native qscale names must be resolved before v0.30's WeightsMapper.
        # v0.30 already initializes the DeepSeek loader fields in the draft;
        # unlike the older patch, no use_mha/EPLB state repair is required.
        weights = fp8_dequant_weight_iter(weights, "Mistral native EAGLE")
    return _ORIGINAL_EAGLE_LOAD_WEIGHTS(self, weights)


EagleMistralLarge3ForCausalLM.load_weights = _patched_eagle_load_weights
# v0.30 borrows SupportsMultiModal.embed_input_ids without inheriting the
# corresponding private helper or its OOV-token flag.
EagleMistralLarge3ForCausalLM._embed_text_input_ids = SupportsMultiModal._embed_text_input_ids
EagleMistralLarge3ForCausalLM._has_oov_mm_tokens = False


try:
    from vllm.model_executor.models.mistral4 import Mistral4Model, unpack_mistral4_expert_weights
except ModuleNotFoundError as error:
    if error.name != "vllm.model_executor.models.mistral4":
        raise
else:
    # Keep other models usable with stock v0.30; Mistral 4's HF model requires
    # the matching core patch that registers Mistral4ForCausalLM.
    _ORIGINAL_LOAD_WEIGHTS = Mistral4Model.load_weights

    def _patched_load_weights(
        self: Mistral4Model,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        if current_platform.device_name == "npu" and self.quant_config is None:
            weights = unpack_mistral4_expert_weights(weights, self.config.n_routed_experts)
            weights = fp8_dequant_weight_iter(weights)
        return _ORIGINAL_LOAD_WEIGHTS(self, weights)

    Mistral4Model.load_weights = _patched_load_weights
