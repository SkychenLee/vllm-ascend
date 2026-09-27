# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import vllm.envs as vllm_envs


def _grouped_prefill_compile_factor() -> bool:
    # Hash the import-time value used by the traced branch, including when
    # AOT loads before AscendCompiler.compute_hash is called.
    from vllm_ascend.lora import grouped_prefill

    return grouped_prefill.ENABLE_GROUPED_PREFILL


vllm_envs.environment_variables["VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL"] = _grouped_prefill_compile_factor
