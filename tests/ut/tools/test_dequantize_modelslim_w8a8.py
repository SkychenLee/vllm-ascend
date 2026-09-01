from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT_PATH = Path(__file__).parents[3] / "tools" / "dequantize_modelslim_w8a8.py"
SPEC = importlib.util.spec_from_file_location("dequantize_modelslim_w8a8", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
dequant = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dequant
SPEC.loader.exec_module(dequant)


def _write_checkpoint(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path.mkdir()
    weight = torch.tensor([[1, -2, 3], [-4, 5, 6]], dtype=torch.int8)
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    offset = torch.tensor([[1.0], [-2.0]], dtype=torch.float32)
    tensors = {
        "linear.weight": weight,
        "linear.weight_scale": scale,
        "linear.weight_offset": offset,
        "norm.weight": torch.tensor([1.25, -2.5], dtype=torch.float32),
        "routing.tid2eid": torch.tensor([[3, 1]], dtype=torch.int32),
    }
    shard_name = "quant_model_weights-00001-of-00001.safetensors"
    save_file(tensors, path / shard_name)
    description = {
        "linear.weight": "W8A8_DYNAMIC",
        "linear.weight_scale": "W8A8_DYNAMIC",
        "linear.weight_offset": "W8A8_DYNAMIC",
        "norm.weight": "FLOAT",
        "routing.tid2eid": "FLOAT",
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        "metadata": {},
        "group_size": 0,
    }
    (path / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")
    index = {
        "metadata": {},
        "weight_map": {name: shard_name for name in tensors},
    }
    (path / "quant_model_weights.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "torch_dtype": "float32",
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "modelslim"},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return weight, scale, offset


def test_convert_checkpoint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    weight, scale, offset = _write_checkpoint(input_dir)

    assert dequant.main([str(input_dir), str(output_dir), "--max-shard-size", "1KiB"]) == 0

    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(index["weight_map"]) == {"linear.weight", "norm.weight", "routing.tid2eid"}
    shard_path = output_dir / next(iter(index["weight_map"].values()))
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        actual_weight = handle.get_tensor("linear.weight")
        expected_weight = ((weight.float() - offset) * scale).to(torch.bfloat16)
        torch.testing.assert_close(actual_weight, expected_weight)
        assert handle.get_tensor("norm.weight").dtype == torch.bfloat16
        assert handle.get_tensor("routing.tid2eid").dtype == torch.int32
        output_names = set(handle.keys())
        assert "linear.weight_scale" not in output_names
        assert "linear.weight_offset" not in output_names

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["torch_dtype"] == "bfloat16"
    assert config["expert_dtype"] == "bf16"
    assert "quantization_config" not in config
    assert not (output_dir / "quant_model_description.json").exists()
    assert (output_dir / "tokenizer_config.json").is_file()


def test_dry_run_does_not_create_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_checkpoint(input_dir)

    assert dequant.main([str(input_dir), str(output_dir), "--dry-run"]) == 0
    assert not output_dir.exists()


def test_resume_reuses_valid_shards(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_checkpoint(input_dir)

    assert dequant.main([str(input_dir), str(output_dir)]) == 0
    assert dequant.main([str(input_dir), str(output_dir), "--resume"]) == 0
