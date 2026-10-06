# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from functools import partial, wraps
from hashlib import sha256

import vllm.envs as vllm_envs
from vllm.config import SpeculativeConfig

from vllm_ascend import envs as ascend_envs


def _grouped_prefill_compile_factor() -> bool:
    # Hash the import-time value used by the traced branch, including when
    # AOT loads before AscendCompiler.compute_hash is called.
    from vllm_ascend.lora import grouped_prefill

    return grouped_prefill.ENABLE_GROUPED_PREFILL


vllm_envs.environment_variables["VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL"] = _grouped_prefill_compile_factor

# Outer AOT may load an artifact before AscendCompiler sees the graph. Read
# the same strict boolean values used to select its operators, so both cache
# layers distinguish opt-in modes. Set these before constructing the engine.
for _flag in (
    "VLLM_ASCEND_MOE_LORA_FUSED_ROUTING",
    "VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION",
    "VLLM_ASCEND_MOE_LORA_FUSED_SHRINK_PAIR",
    "VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY",
):
    vllm_envs.environment_variables[_flag] = partial(getattr, ascend_envs, _flag)


def _with_mtp_method_compile_factor(original):
    # Upstream hashes a shared aux-output boolean for DSpark/eagle3. With
    # this optimization enabled their ordinary MTP stash graphs can differ.
    # Runner selection is already part of upstream env compile_factors.
    if getattr(original, "_ascend_mtp_method_compile_factor", False):
        return original

    @wraps(original)
    def compute_hash(config):
        original_hash = original(config)
        if not ascend_envs.VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY:
            return original_hash
        return sha256(str((original_hash, config.method)).encode(), usedforsecurity=False).hexdigest()

    compute_hash._ascend_mtp_method_compile_factor = True
    return compute_hash


SpeculativeConfig.compute_hash = _with_mtp_method_compile_factor(SpeculativeConfig.compute_hash)
