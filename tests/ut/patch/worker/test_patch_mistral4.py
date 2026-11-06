# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.mistral_large_3_eagle import EagleMistralLarge3ForCausalLM

from vllm_ascend.patch.worker import patch_mistral4 as patch


def test_native_eagle_is_dequantized_before_v030_name_mapping(monkeypatch):
    captured = {}

    def load_weights(_self, weights):
        captured.update(dict(EagleMistralLarge3ForCausalLM.hf_to_vllm_mapper.apply(weights)))
        return set(captured)

    monkeypatch.setattr(patch, "_ORIGINAL_EAGLE_LOAD_WEIGHTS", load_weights)
    monkeypatch.setattr(patch, "current_platform", SimpleNamespace(device_name="npu"))
    loaded = patch._patched_eagle_load_weights(
        SimpleNamespace(quant_config=None),
        [
            ("eagle_linear.qscale_weight", torch.tensor(0.25)),
            ("eagle_linear.weight", torch.ones((2, 2), dtype=torch.float8_e4m3fn)),
            ("layers.0.attention.wq_a.weight", torch.ones((2, 2), dtype=torch.float8_e4m3fn)),
            ("layers.0.attention.wq_a.qscale_weight", torch.tensor(0.5)),
        ],
    )
    assert loaded == {"model.fc.weight", "model.layers.0.self_attn.q_a_proj.weight"}
    torch.testing.assert_close(captured["model.fc.weight"], torch.full((2, 2), 0.25, dtype=torch.bfloat16))
    torch.testing.assert_close(
        captured["model.layers.0.self_attn.q_a_proj.weight"], torch.full((2, 2), 0.5, dtype=torch.bfloat16)
    )


def test_hf_experts_are_unpacked_before_bf16_conversion(monkeypatch):
    captured = {}

    def load_weights(_self, weights):
        captured.update(dict(weights))
        return set(captured)

    monkeypatch.setattr(patch, "_ORIGINAL_LOAD_WEIGHTS", load_weights)
    monkeypatch.setattr(patch, "current_platform", SimpleNamespace(device_name="npu"))
    weights = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4).to(torch.float8_e4m3fn)
    scales = torch.tensor([0.25, 0.5], dtype=torch.bfloat16).view(2, 1, 1)
    patch._patched_load_weights(
        SimpleNamespace(quant_config=None, config=SimpleNamespace(n_routed_experts=2)),
        [
            ("layers.0.mlp.experts.gate_up_proj_scale_inv", scales),
            ("layers.0.mlp.experts.gate_up_proj", weights),
        ],
    )
    assert len(captured) == 4
    for expert in range(2):
        for index, projection in enumerate(("gate_proj", "up_proj")):
            expected = (weights[expert, index * 2 : (index + 1) * 2].float() * scales[expert].float()).bfloat16()
            actual = captured[f"layers.0.mlp.experts.{expert}.{projection}.weight"]
            assert actual.ndim == 2
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_eagle_embedding_protocol_uses_v030_language_model():
    assert EagleMistralLarge3ForCausalLM._embed_text_input_ids is SupportsMultiModal._embed_text_input_ids

    class LanguageModel(torch.nn.Module):
        def embed_input_ids(self, input_ids):
            return input_ids.float().unsqueeze(-1)

    eagle = EagleMistralLarge3ForCausalLM.__new__(EagleMistralLarge3ForCausalLM)
    torch.nn.Module.__init__(eagle)
    eagle.model = LanguageModel()
    torch.testing.assert_close(eagle.embed_input_ids(torch.tensor([1, 2])), torch.tensor([[1.0], [2.0]]))
