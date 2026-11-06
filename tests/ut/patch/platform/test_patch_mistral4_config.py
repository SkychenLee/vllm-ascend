# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform import patch_mistral4_config as patch


def _config(quantization=None, quant_config=None, *, eagle=False):
    if quant_config is None:
        quant_config = {"quant_method": "fp8", "activation_scheme": "static"}
    text = SimpleNamespace(model_type="mistral4")
    hf_config = SimpleNamespace(model_type="mistral3", text_config=text)
    if eagle:
        hf_config = SimpleNamespace(model_type="deepseek_mtp", architectures=["EagleMistralLarge3ForCausalLM"])
    return SimpleNamespace(
        model="mistralai/Mistral-Small-4-119B-2603-eagle" if eagle else "mistralai/Mistral-Small-4-119B-2603",
        revision="pinned-revision",
        hf_config=hf_config,
        model_arch_config=SimpleNamespace(quantization_config=quant_config),
        quantization=quantization,
    )


@pytest.fixture(autouse=True)
def _npu(monkeypatch):
    monkeypatch.setattr(patch, "current_platform", SimpleNamespace(device_name="npu"))


@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("quantization", [None, "fp8"])
def test_per_tensor_fp8_is_disabled_before_upstream_verification(monkeypatch, eagle, quantization):
    config = _config(quantization=quantization, eagle=eagle)

    def reject_original(_config):
        raise AssertionError("Per-tensor FP8 must not enter the unsupported backend")

    monkeypatch.setattr(patch, "_ORIGINAL_VERIFY_QUANTIZATION", reject_original)
    patch._patched_verify_quantization(config)
    assert config.quantization is None


def test_post_config_auto_detection_cannot_restore_fp8(monkeypatch):
    config = _config()
    vllm_config = SimpleNamespace(model_config=config, quant_config="stale")

    def reject_original(_config):
        raise AssertionError("Must not recreate an AscendFp8Config")

    monkeypatch.setattr(patch, "_ORIGINAL_AUTO_DETECT_QUANTIZATION", reject_original)
    patch._patched_maybe_auto_detect_quantization(vllm_config)
    assert config.quantization is None
    assert vllm_config.quant_config is None


@pytest.mark.parametrize("mode", ["cpu", "block", "other_quant", "other_model"])
def test_unrelated_cases_keep_upstream_verification(monkeypatch, mode):
    config = _config()
    if mode == "cpu":
        monkeypatch.setattr(patch, "current_platform", SimpleNamespace(device_name="cpu"))
    elif mode == "block":
        config.model_arch_config.quantization_config["weight_block_size"] = [128, 128]
    elif mode == "other_quant":
        config.quantization = "compressed-tensors"
    else:
        config.hf_config.text_config.model_type = "llama"
    calls = []
    monkeypatch.setattr(patch, "_ORIGINAL_VERIFY_QUANTIZATION", calls.append)
    patch._patched_verify_quantization(config)
    assert calls == [config]


@pytest.mark.parametrize("eagle", [False, True])
def test_inner_or_remote_config_uses_cached_hub_metadata(monkeypatch, tmp_path, eagle):
    config = _config(eagle=eagle)
    config.model_arch_config.quantization_config = None
    filename = "params.json" if eagle else "config.json"
    payload = (
        {"quantization": {"qformat_weight": "fp8_e4m3", "qscheme_act": "TENSOR"}}
        if eagle
        else {"quantization_config": {"quant_method": "fp8", "activation_scheme": "static"}}
    )
    config_path = tmp_path / filename
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    calls = []

    def resolve(model, name, revision=None):
        calls.append((model, name, revision))
        return config_path

    monkeypatch.setattr(patch.quantization_utils, "get_model_file", resolve)
    assert patch._should_dequantize(config)
    assert calls == [(config.model, filename, "pinned-revision")]
