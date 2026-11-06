# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.quantization import mistral_fp8
from vllm_ascend.quantization.mistral_fp8 import (
    dequantize_fp8_weight,
    fp8_dequant_weight_iter,
    get_mistral_quantization_config,
    is_mistral4_config,
    is_mistral_eagle_config,
    is_per_tensor_fp8,
)


def test_dequantization_matches_fp32_reference_across_row_chunks(monkeypatch):
    monkeypatch.setattr(mistral_fp8, "DEQUANT_ROWS_PER_STEP", 2)
    values = torch.arange(-12, 12, dtype=torch.float32).reshape(6, 4) / 4
    weight = values.to(torch.float8_e4m3fn)
    scale = torch.tensor(0.371, dtype=torch.bfloat16)
    actual = dequantize_fp8_weight(weight, scale)
    expected = (weight.float() * scale.float()).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(weight.float(), values, rtol=0, atol=0)


def test_scalar_scale_shape_does_not_change_matrix_shape():
    weight = torch.ones((2, 3), dtype=torch.float8_e4m3fn)
    actual = dequantize_fp8_weight(weight, torch.tensor([[[0.5]]]))
    assert actual.shape == weight.shape
    torch.testing.assert_close(actual, torch.full((2, 3), 0.5, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("weight_shape", "scale_shape"),
    [((2, 3), (2, 1)), ((2, 3), (2, 2)), ((2, 3, 4), (2, 1, 1))],
)
def test_rejects_unsupported_scales_and_unexpanded_expert_stacks(weight_shape, scale_shape):
    with pytest.raises(ValueError, match="Unpack fused experts"):
        dequantize_fp8_weight(
            torch.ones(weight_shape, dtype=torch.float8_e4m3fn),
            torch.ones(scale_shape),
        )


@pytest.mark.parametrize("scale_suffix", [".weight_scale_inv", ".weight_scale", ".qscale_weight"])
@pytest.mark.parametrize("scale_first", [False, True])
def test_pairing_accepts_hf_mapped_and_native_names_in_both_orders(scale_suffix, scale_first):
    weight = ("layer.weight", torch.ones((2, 3), dtype=torch.float8_e4m3fn))
    scale = ("layer" + scale_suffix, torch.tensor(0.25))
    inputs = [scale, weight] if scale_first else [weight, scale]
    inputs += [
        ("layer.activation_scale", torch.tensor(3.0)),
        ("layer.input_scale", torch.tensor(4.0)),
        ("layer.qscale_act", torch.tensor(5.0)),
    ]
    result = dict(fp8_dequant_weight_iter(inputs))
    assert set(result) == {"layer.weight"}
    torch.testing.assert_close(result["layer.weight"], torch.full((2, 3), 0.25, dtype=torch.bfloat16))


def test_gate_and_up_keep_independent_native_scales():
    weights = [
        ("layers.0.experts.0.w1.weight", torch.ones((2, 3), dtype=torch.float8_e4m3fn)),
        ("layers.0.experts.0.w3.weight", torch.ones((2, 3), dtype=torch.float8_e4m3fn)),
        ("layers.0.experts.0.w3.qscale_weight", torch.tensor(0.75)),
        ("layers.0.experts.0.w1.qscale_weight", torch.tensor(0.25)),
    ]
    result = dict(fp8_dequant_weight_iter(weights))
    torch.testing.assert_close(result["layers.0.experts.0.w1.weight"], torch.full((2, 3), 0.25, dtype=torch.bfloat16))
    torch.testing.assert_close(result["layers.0.experts.0.w3.weight"], torch.full((2, 3), 0.75, dtype=torch.bfloat16))


def test_mixed_bf16_modules_pass_through_unchanged():
    norm = torch.arange(4, dtype=torch.bfloat16)
    router = torch.zeros((2, 4), dtype=torch.bfloat16)
    result = dict(fp8_dequant_weight_iter([("norm.weight", norm), ("mlp.gate.weight", router)]))
    assert result["norm.weight"] is norm
    assert result["mlp.gate.weight"] is router


@pytest.mark.parametrize("scale_first", [False, True])
def test_non_fp8_weight_with_scale_is_rejected(scale_first):
    weight = ("layer.weight", torch.ones((2, 3), dtype=torch.bfloat16))
    scale = ("layer.weight_scale", torch.tensor(0.5))
    with pytest.raises(ValueError, match="non-FP8 weight"):
        list(fp8_dequant_weight_iter([scale, weight] if scale_first else [weight, scale]))


@pytest.mark.parametrize(
    "inputs",
    [
        [("layer.weight", torch.ones((2, 2), dtype=torch.float8_e4m3fn))],
        [("layer.weight_scale", torch.tensor(0.5))],
    ],
)
def test_missing_weight_or_scale_raises(inputs):
    with pytest.raises(ValueError, match="Unpaired Mistral 4"):
        list(fp8_dequant_weight_iter(inputs))


def test_duplicate_scale_aliases_are_rejected():
    with pytest.raises(ValueError, match="Duplicate Mistral 4 FP8 scale"):
        list(
            fp8_dequant_weight_iter(
                [
                    ("layer.weight_scale", torch.tensor(0.25)),
                    ("layer.weight", torch.ones((2, 2), dtype=torch.float8_e4m3fn)),
                    ("layer.weight_scale_inv", torch.tensor(0.5)),
                ]
            )
        )


def test_duplicate_weight_is_rejected():
    weight = ("layer.weight", torch.ones((2, 2), dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="Duplicate Mistral 4 weight"):
        list(fp8_dequant_weight_iter([weight, weight]))


def test_completed_pair_is_yielded_without_reading_the_next_projection():
    read_next_projection = False

    def inputs():
        nonlocal read_next_projection
        yield "first.weight_scale", torch.tensor(0.5)
        yield "first.weight", torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        read_next_projection = True
        yield "norm.weight", torch.ones(2, dtype=torch.bfloat16)

    converted = fp8_dequant_weight_iter(inputs())
    name, first = next(converted)
    assert name == "first.weight"
    assert not read_next_projection
    torch.testing.assert_close(first, torch.full((2, 2), 0.5, dtype=torch.bfloat16))
    assert len(list(converted)) == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_mistral4_config_detection_before_and_after_text_config_extraction(wrapped):
    text = SimpleNamespace(model_type="mistral4")
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="mistral3", text_config=text))
    if wrapped:
        config.hf_text_config = text
    assert is_mistral4_config(config)
    assert not is_mistral_eagle_config(config)


@pytest.mark.parametrize("model_type", ["deepseek_mtp", "eagle"])
def test_eagle_detection_accepts_remote_repo_ids_and_config_lifecycle(model_type):
    config = SimpleNamespace(
        model="mistralai/Mistral-Small-4-119B-2603-eagle",
        hf_config=SimpleNamespace(
            model_type=model_type,
            architectures=["EagleMistralLarge3ForCausalLM"],
        ),
    )
    assert is_mistral_eagle_config(config)
    assert not is_mistral4_config(config)


def test_native_main_and_other_eagle_models_are_not_selected():
    for architecture in ("MistralLarge3ForCausalLM", "EagleDeepseekV3ForCausalLM"):
        config = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v3", architectures=[architecture]))
        assert not is_mistral4_config(config)
        assert not is_mistral_eagle_config(config)


def test_quantization_detection_uses_nested_metadata_and_leaves_block_fp8_alone():
    quant_config = {"quant_method": "fp8", "activation_scheme": "static", "weight_block_size": None}
    config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=SimpleNamespace(model_type="mistral4", quantization_config=quant_config))
    )
    assert get_mistral_quantization_config(config) is quant_config
    assert is_per_tensor_fp8(quant_config)
    assert not is_per_tensor_fp8({**quant_config, "weight_block_size": [128, 128]})
    assert not is_per_tensor_fp8({"quant_method": "compressed-tensors"})
