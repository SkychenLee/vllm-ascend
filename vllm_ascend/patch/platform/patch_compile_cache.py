# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from functools import partial

import vllm.envs as vllm_envs

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
    "VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY",
):
    vllm_envs.environment_variables[_flag] = partial(getattr, ascend_envs, _flag)
