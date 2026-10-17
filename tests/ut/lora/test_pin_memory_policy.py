# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of the Ascend pinned-memory policy used by the LoRA loader."""

import os
import subprocess
import sys
import textwrap

import pytest

from vllm_ascend import envs
from vllm_ascend.platform import NPUPlatform


@pytest.mark.parametrize("value,available", [(None, True), ("0", True), ("1", False)])
def test_platform_pin_memory_policy(monkeypatch, value, available):
    if value is None:
        monkeypatch.delenv("VLLM_ASCEND_DISABLE_PIN_MEMORY", raising=False)
    else:
        monkeypatch.setenv("VLLM_ASCEND_DISABLE_PIN_MEMORY", value)
    assert "VLLM_ASCEND_DISABLE_PIN_MEMORY" in envs.env_variables
    assert NPUPlatform.is_pin_memory_available() is available


@pytest.mark.parametrize("disable", ["0", "1"])
def test_real_lora_loader_inherits_policy_at_import(disable):
    # vLLM snapshots PIN_MEMORY during import. A fresh process validates the
    # actual import order without patching that constant or the platform.
    script = textwrap.dedent(
        """
        import os
        import torch
        import torch_npu

        disabled = os.environ["VLLM_ASCEND_DISABLE_PIN_MEMORY"] == "1"
        pin_calls = []

        def forbid_npu(*args, **kwargs):
            raise AssertionError("CPU LoRA loading must not initialize the NPU")

        def record_pin(self, *args, **kwargs):
            assert not disabled, "pin_memory called despite the disable flag"
            pin_calls.append(self)
            return self

        torch.npu._lazy_init = forbid_npu
        torch.Tensor.pin_memory = record_pin

        from vllm.lora.lora_model import LoRAModel, PIN_MEMORY
        from vllm.lora.peft_helper import PEFTHelper
        from vllm.platforms import current_platform
        from vllm_ascend.platform import NPUPlatform

        assert isinstance(current_platform, NPUPlatform)
        assert PIN_MEMORY is not disabled
        weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight":
                torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight":
                torch.arange(12, dtype=torch.float32).reshape(6, 2),
        }
        model = LoRAModel.from_lora_tensors(
            1, weights, PEFTHelper(r=2, lora_alpha=4, target_modules=["q_proj"]),
            device="cpu", dtype=torch.bfloat16,
        )
        layer = next(iter(model.loras.values()))
        assert layer.lora_a.dtype == layer.lora_b.dtype == torch.bfloat16
        assert layer.lora_a.device.type == layer.lora_b.device.type == "cpu"
        torch.testing.assert_close(layer.lora_a, next(iter(weights.values())).to(torch.bfloat16))
        torch.testing.assert_close(layer.lora_b, list(weights.values())[1].to(torch.bfloat16))
        assert len(pin_calls) == (0 if disabled else 2), pin_calls
        assert not torch.npu.is_initialized()
        print(f"PASS: disable={disabled}, PIN_MEMORY={PIN_MEMORY}, pin_calls={len(pin_calls)}")
        """
    )
    child_env = os.environ.copy()
    child_env["VLLM_ASCEND_DISABLE_PIN_MEMORY"] = disable
    result = subprocess.run([sys.executable, "-c", script], env=child_env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
