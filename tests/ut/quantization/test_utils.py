import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch
from vllm.config import KVTransferConfig

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import FAKQUANT_CONFIG, W8A8_CONFIG
from vllm_ascend.quantization import AscendCompressedTensorsConfig
from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME, AscendModelSlimConfig
from vllm_ascend.quantization.utils import (
    MXFP4_BLOCK_SIZE,
    detect_quantization_method,
    enable_fa_quant,
    enable_hif4_qkv_quant,
    maybe_auto_detect_quantization,
    quant_dequant_hif4,
    quant_dequant_mxfp4,
    quant_dequant_mxfp4_grouped,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD


class TestDetectQuantizationMethod(TestBase):
    def test_returns_none_for_non_existent_path(self):
        result = detect_quantization_method("/non/existent/path")
        self.assertIsNone(result)

    def test_detects_modelslim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump({"layer.weight": "INT8"}, f)

            result = detect_quantization_method(tmpdir)
            self.assertEqual(result, ASCEND_QUANTIZATION_METHOD)

    def test_detects_compressed_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"quantization_config": {"quant_method": "compressed-tensors"}}, f)

            result = detect_quantization_method(tmpdir)
            self.assertEqual(result, COMPRESSED_TENSORS_METHOD)

    def test_returns_none_for_no_quant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_quantization_method(tmpdir)
            self.assertIsNone(result)

    def test_returns_none_for_non_compressed_tensors_quant_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"quantization_config": {"quant_method": "gptq"}}, f)

            result = detect_quantization_method(tmpdir)
            self.assertIsNone(result)

    def test_returns_none_for_config_without_quant_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "llama"}, f)

            result = detect_quantization_method(tmpdir)
            self.assertIsNone(result)

    def test_returns_none_for_malformed_config_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                f.write("not valid json{{{")

            result = detect_quantization_method(tmpdir)
            self.assertIsNone(result)

    def test_modelslim_takes_priority_over_compressed_tensors(self):
        """When both ModelSlim config and compressed-tensors config exist,
        ModelSlim should take priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modelslim_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(modelslim_path, "w") as f:
                json.dump({"layer.weight": "INT8"}, f)

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"quantization_config": {"quant_method": "compressed-tensors"}}, f)

            result = detect_quantization_method(tmpdir)
            self.assertEqual(result, ASCEND_QUANTIZATION_METHOD)


class TestMaybeAutoDetectQuantization(TestBase):
    def _make_vllm_config(self, model_path="/fake/model", quantization=None, revision=None):
        vllm_config = MagicMock()
        vllm_config.model_config.model = model_path
        vllm_config.model_config.quantization = quantization
        vllm_config.model_config.revision = revision
        return vllm_config

    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=None)
    def test_no_detection_does_nothing(self, mock_detect):
        vllm_config = self._make_vllm_config()
        maybe_auto_detect_quantization(vllm_config)
        self.assertIsNone(vllm_config.model_config.quantization)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=ASCEND_QUANTIZATION_METHOD)
    def test_user_specified_same_method_no_change(self, mock_detect):
        vllm_config = self._make_vllm_config(quantization=ASCEND_QUANTIZATION_METHOD)
        maybe_auto_detect_quantization(vllm_config)
        self.assertEqual(vllm_config.model_config.quantization, ASCEND_QUANTIZATION_METHOD)

    @patch("vllm.config.VllmConfig._get_quantization_config", return_value=MagicMock())
    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=ASCEND_QUANTIZATION_METHOD)
    def test_auto_detect_sets_quantization_and_logs_info(self, mock_detect, mock_get_quant_config):
        """When no --quantization is specified but ModelSlim config is found,
        the method should auto-set quantization and emit an INFO log."""
        vllm_config = self._make_vllm_config(model_path="/fake/quant_model", quantization=None)

        with patch("vllm_ascend.quantization.utils.logger") as mock_logger:
            maybe_auto_detect_quantization(vllm_config)

        self.assertEqual(vllm_config.model_config.quantization, ASCEND_QUANTIZATION_METHOD)
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        self.assertIn("Auto-detected quantization method", call_args[0])
        self.assertIn(ASCEND_QUANTIZATION_METHOD, call_args)
        self.assertIn("/fake/quant_model", call_args)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=ASCEND_QUANTIZATION_METHOD)
    def test_user_mismatch_logs_warning(self, mock_detect):
        """When user specifies a different method than auto-detected,
        a WARNING should be emitted and user's choice should be respected."""
        vllm_config = self._make_vllm_config(model_path="/fake/quant_model", quantization=COMPRESSED_TENSORS_METHOD)

        with patch("vllm_ascend.quantization.utils.logger") as mock_logger:
            maybe_auto_detect_quantization(vllm_config)

        self.assertEqual(vllm_config.model_config.quantization, COMPRESSED_TENSORS_METHOD)
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        self.assertIn("Auto-detected quantization method", call_args[0])
        self.assertIn(ASCEND_QUANTIZATION_METHOD, call_args)
        self.assertIn(COMPRESSED_TENSORS_METHOD, call_args)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=None)
    def test_no_detection_emits_info_log(self, mock_detect):
        """When no quantization is detected, an info log tells the user the model loads as float."""
        vllm_config = self._make_vllm_config(quantization=None)

        with patch("vllm_ascend.quantization.utils.logger") as mock_logger:
            maybe_auto_detect_quantization(vllm_config)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        self.assertIn("No quantization signature detected", call_args[0])
        self.assertIn("/fake/model", call_args)
        mock_logger.warning.assert_not_called()
        self.assertIsNone(vllm_config.model_config.quantization)

    @patch("vllm.config.VllmConfig._get_quantization_config", return_value=MagicMock())
    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=ASCEND_QUANTIZATION_METHOD)
    def test_passes_revision_to_detect(self, mock_detect, mock_get_quant):
        """Verify that model revision is forwarded to detect_quantization_method."""
        vllm_config = self._make_vllm_config(model_path="org/model-name", revision="v1.0", quantization=None)
        maybe_auto_detect_quantization(vllm_config)
        mock_detect.assert_called_once_with("org/model-name", revision="v1.0")


class TestEnableFaQuant(TestBase):
    def test_non_quantization_scenarios(self):
        # non quantization scene
        vllm_config = MagicMock()
        vllm_config.quant_config = None
        result = enable_fa_quant(vllm_config)
        self.assertFalse(result)

        # CompressedTensors scene
        vllm_config.quant_config = AscendCompressedTensorsConfig({}, [], "", {})
        result = enable_fa_quant(vllm_config)
        self.assertFalse(result)

        # non fa3 quant scene
        vllm_config.quant_config = AscendModelSlimConfig(W8A8_CONFIG)
        result = enable_fa_quant(vllm_config)
        self.assertFalse(result)

    def test_fa3_quantization_scenario(self):
        vllm_config = MagicMock()
        vllm_config.quant_config = AscendModelSlimConfig(FAKQUANT_CONFIG)
        vllm_config.kv_transfer_config = KVTransferConfig(kv_connector="MultiConnector", kv_role="kv_consumer")
        result = enable_fa_quant(vllm_config)
        self.assertTrue(result)
        result = enable_fa_quant(vllm_config, layer_name="test_layer")
        self.assertFalse(result)


class TestMXFP4PseudoQuant(TestBase):
    @staticmethod
    def _reference_mxfp4(x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        blocks = x.unflatten(-1, (-1, MXFP4_BLOCK_SIZE))
        max_values = torch.amax(blocks.abs(), -1, keepdim=True)
        shared_exponents = torch.ceil(torch.log2(max_values.clamp(min=1.17e-38) / 7)).clamp(-127, 127)
        scaled_blocks = blocks * torch.exp2(-shared_exponents)
        private_exponents = torch.floor(torch.log2(scaled_blocks.abs().clamp(min=1.17e-38))).clamp(min=0)
        mantissas = scaled_blocks * torch.exp2(-private_exponents) * 2
        mantissas = torch.sign(mantissas) * torch.floor(mantissas.abs() + 0.5)
        quantized = (mantissas * 0.5 * torch.exp2(private_exponents)).clamp(-6, 6)
        return (quantized * torch.exp2(shared_exponents)).reshape(original_shape)

    @staticmethod
    def _reference_mxfp4_grouped(x: torch.Tensor) -> torch.Tensor:
        max_values = torch.amax(x.abs(), 1, keepdim=True)
        shared_exponents = torch.ceil(torch.log2(max_values.clamp(min=1.17e-38) / 7)).clamp(-127, 127)
        scaled_blocks = x * torch.exp2(-shared_exponents)
        private_exponents = torch.floor(torch.log2(scaled_blocks.abs().clamp(min=1.17e-38))).clamp(min=0)
        mantissas = scaled_blocks * torch.exp2(-private_exponents) * 2
        mantissas = torch.sign(mantissas) * torch.floor(mantissas.abs() + 0.5)
        quantized = (mantissas * 0.5 * torch.exp2(private_exponents)).clamp(-6, 6)
        return quantized * torch.exp2(shared_exponents)

    def test_zero_blocks_stay_finite_and_zero(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                x = torch.zeros(2, 3, 64, dtype=dtype)

                output = quant_dequant_mxfp4(x, -1)

                self.assertEqual(MXFP4_BLOCK_SIZE, 32)
                self.assertEqual(output.shape, x.shape)
                self.assertEqual(output.dtype, x.dtype)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.equal(output, x))

    def test_quantization_dimension_must_be_divisible_by_32(self):
        x = torch.randn(2, 3, 35, dtype=torch.float32)

        with self.assertRaises(RuntimeError):
            quant_dequant_mxfp4(x, -1)

    def test_bfloat16_matches_attention_reference_math(self):
        x = torch.randn(4, 64, dtype=torch.bfloat16)

        output = quant_dequant_mxfp4(x, -1)
        expected = self._reference_mxfp4(x)

        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_grouped_bfloat16_matches_attention_reference_math(self):
        x = torch.randn(4, MXFP4_BLOCK_SIZE, 2, 8, dtype=torch.bfloat16)

        output = quant_dequant_mxfp4_grouped(x)
        expected = self._reference_mxfp4_grouped(x)

        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_each_32_value_block_has_an_independent_shared_scale(self):
        first_block = torch.linspace(-1, 1, MXFP4_BLOCK_SIZE)
        baseline = torch.cat((first_block, torch.zeros(MXFP4_BLOCK_SIZE)))
        large_second_block = torch.cat((first_block, torch.full((MXFP4_BLOCK_SIZE,), 1024.0)))

        baseline_output = quant_dequant_mxfp4(baseline, -1)
        large_output = quant_dequant_mxfp4(large_second_block, -1)

        torch.testing.assert_close(
            baseline_output[:MXFP4_BLOCK_SIZE],
            large_output[:MXFP4_BLOCK_SIZE],
            rtol=0,
            atol=0,
        )


class TestHiF4PseudoQuant(TestBase):
    def test_zero_blocks_stay_finite_and_zero(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                x = torch.zeros(2, 3, 70, dtype=dtype)

                output = quant_dequant_hif4(x)

                self.assertEqual(output.shape, x.shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.equal(output, x))

    def test_non_multiple_dimension_is_padded_and_cropped(self):
        x = torch.randn(2, 3, 70, dtype=torch.float32)

        output = quant_dequant_hif4(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.isfinite(output).all())

    def test_quantization_axis_is_honored(self):
        x = torch.randn(2, 70, 3, dtype=torch.float32)

        output = quant_dequant_hif4(x, axe=1)
        expected = quant_dequant_hif4(x.movedim(1, -1)).movedim(-1, 1)

        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_qkv_quantization_is_enabled_only_for_hif4_projections(self):
        vllm_config = MagicMock()
        vllm_config.quant_config.quant_description = {
            "model.layers.0.self_attn.q_proj.weight": "W4A4_HIFP4",
            "model.layers.0.self_attn.k_proj.weight": "W4A4_HIFP4",
            "model.layers.0.self_attn.v_proj.weight": "W4A4_HIFP4",
            "model.layers.0.mlp.gate_proj.weight": "W4A4_HIFP4",
        }
        self.assertTrue(enable_hif4_qkv_quant(vllm_config))

        vllm_config.quant_config.quant_description = {
            "model.layers.0.self_attn.qkv_proj.weight": "W4A4_HIFP4",
        }
        self.assertTrue(enable_hif4_qkv_quant(vllm_config))

        vllm_config.quant_config.quant_description = {
            "model.layers.0.self_attn.q_proj.weight": "W4A4_HIFP4",
            "model.layers.0.self_attn.k_proj.weight": "W4A4_MXFP4",
            "model.layers.0.self_attn.v_proj.weight": "W4A4_HIFP4",
            "model.layers.0.mlp.gate_proj.weight": "W4A4_HIFP4",
        }
        self.assertFalse(enable_hif4_qkv_quant(vllm_config))
