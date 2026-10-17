# SPDX-License-Identifier: Apache-2.0
"""CPU checks for explicit MoE LoRA EP communication selection."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import VllmConfig

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_config import AscendConfig


@pytest.fixture(autouse=True)
def forbid_npu_initialization(monkeypatch):
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "_lazy_init", Mock(side_effect=AssertionError("CPU-only EP config tests")))


@pytest.mark.parametrize(
    "additional,expected",
    [
        (None, "auto"),
        ({}, "auto"),
        ({"moe_lora_ep_backend": "auto"}, "auto"),
        ({"moe_lora_ep_backend": "allgather"}, "allgather"),
    ],
)
def test_config_parses_ep_backend(additional, expected):
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        config = VllmConfig()
    config.additional_config = additional
    assert AscendConfig(config).moe_lora_ep_backend == expected


@pytest.mark.parametrize("value", ["alltoall", "ALLGATHER", "", None, True, 1, [], {}])
def test_config_rejects_invalid_ep_backend(value):
    # Invalid values are rejected before constructing unrelated configuration.
    config = SimpleNamespace(additional_config={"moe_lora_ep_backend": value})
    with pytest.raises(ValueError, match="moe_lora_ep_backend must be"):
        AscendConfig(config)


def select(monkeypatch, device, backend, *, lora=True, ep=True, ep_size=8, tokens=48):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 48)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(moe_lora_ep_backend=backend))
    config = SimpleNamespace(
        lora_config=object() if lora else None, parallel_config=SimpleNamespace(enable_expert_parallel=ep)
    )
    return afc.select_moe_comm_method(tokens, config)


@pytest.mark.parametrize("device", [afc.AscendDeviceType.A2, afc.AscendDeviceType.A3, afc.AscendDeviceType.A5])
@pytest.mark.parametrize("tokens", [5, 48, 16384])
def test_explicit_allgather_for_decode_and_prefill(monkeypatch, device, tokens):
    assert select(monkeypatch, device, "allgather", tokens=tokens) == afc.MoECommType.ALLGATHER


@pytest.mark.parametrize(
    "device,expected",
    [
        (afc.AscendDeviceType.A2, afc.MoECommType.ALLGATHER),
        (afc.AscendDeviceType.A3, afc.MoECommType.ALLTOALL),
        (afc.AscendDeviceType.A5, afc.MoECommType.ALLTOALL),
    ],
)
def test_auto_preserves_existing_lora_policy(monkeypatch, device, expected):
    assert select(monkeypatch, device, "auto") == expected


def test_override_does_not_change_service_without_lora(monkeypatch):
    choose = Mock(return_value=afc.MoECommType.MC2)
    monkeypatch.setattr(afc, "_select_a3_moe_comm_method", choose)
    assert select(monkeypatch, afc.AscendDeviceType.A3, "allgather", lora=False) == afc.MoECommType.MC2
    choose.assert_called_once()


@pytest.mark.parametrize("ep,ep_size", [(False, 8), (True, 1)])
def test_no_effective_ep_retains_allgather(monkeypatch, ep, ep_size):
    assert select(monkeypatch, afc.AscendDeviceType.A3, "auto", ep=ep, ep_size=ep_size) == afc.MoECommType.ALLGATHER
