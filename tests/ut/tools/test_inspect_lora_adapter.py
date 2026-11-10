# SPDX-License-Identifier: Apache-2.0
"""Real-file CPU regressions for the standalone adapter-inspection CLI."""

import csv
import importlib.util
import io
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/inspect_lora_adapter.py"
KEY_A = "language_model.model.layers.0.self_attn.q_b_proj.lora_A.default.weight"
KEY_B = "language_model.model.layers.0.self_attn.q_b_proj.lora_B.default.weight"


@pytest.fixture(scope="module")
def inspector():
    spec = importlib.util.spec_from_file_location("inspect_adapter_test_target", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tensors():
    return {
        KEY_A: torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        KEY_B: torch.arange(12, dtype=torch.float32).reshape(6, 2),
        "scalar": torch.tensor(3, dtype=torch.int64),
        "empty": torch.empty(0, 3, dtype=torch.float16),
    }


@pytest.fixture
def safetensors_file(tmp_path, tensors):
    path = tmp_path / "adapter_model.safetensors"
    safetensors.torch.save_file(tensors, path, metadata={"format": "pt"})
    return path


def run_cli(*args, no_site=False):
    command = [sys.executable]
    if no_site:
        command.append("-S")
    command.extend([str(SCRIPT), *(str(argument) for argument in args)])
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)


def test_safetensors_preserves_raw_names_scalar_empty_and_dtype(inspector, safetensors_file, tensors):
    rows = inspector.inspect_adapter(safetensors_file)
    assert [row["key"] for row in rows] == sorted(tensors)
    by_key = {row["key"]: row for row in rows}
    assert KEY_A in by_key and KEY_B in by_key
    assert by_key[KEY_A]["shape"] == [2, 4]
    assert by_key[KEY_A]["dtype"] == "BF16"
    assert by_key[KEY_B]["dtype"] == "F32"
    assert by_key["scalar"]["shape"] == []
    assert by_key["empty"]["shape"] == [0, 3]
    assert {row["file"] for row in rows} == {safetensors_file.name}


def test_safetensors_cli_runs_without_site_packages_or_torch(safetensors_file):
    result = run_cli(safetensors_file, "--format", "json", no_site=True)
    assert result.returncode == 0, result.stderr
    assert {row["key"] for row in json.loads(result.stdout)} == {
        KEY_A,
        KEY_B,
        "scalar",
        "empty",
    }
    assert "Total: 4 tensor keys" in result.stderr


def test_directory_prefers_standard_safetensors_over_bin(inspector, safetensors_file):
    torch.save({"bin_only": torch.ones(1)}, safetensors_file.with_suffix(".bin"))
    rows = inspector.inspect_adapter(safetensors_file.parent)
    assert len(rows) == 4
    assert all(row["file"] == "adapter_model.safetensors" for row in rows)
    assert "bin_only" not in {row["key"] for row in rows}


def write_index(tmp_path, first, second, mapping=None, *, binary=False):
    suffix = "bin" if binary else "safetensors"
    names = [
        f"adapter_model-00001-of-00002.{suffix}",
        f"adapter_model-00002-of-00002.{suffix}",
    ]
    for name, payload in zip(names, (first, second)):
        if binary:
            torch.save(payload, tmp_path / name)
        else:
            safetensors.torch.save_file(payload, tmp_path / name)
    index = tmp_path / f"adapter_model.{suffix}.index.json"
    mapping = mapping if mapping is not None else {KEY_A: names[0], KEY_B: names[1]}
    index.write_text(json.dumps({"metadata": {}, "weight_map": mapping}), encoding="utf-8")
    return index, names


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("use_directory", [False, True])
def test_shard_index_returns_every_original_key(inspector, tmp_path, binary, use_directory):
    index, names = write_index(tmp_path, {KEY_A: torch.ones(2, 4)}, {KEY_B: torch.ones(6, 2)}, binary=binary)
    rows = inspector.inspect_adapter(tmp_path if use_directory else index)
    assert [(row["key"], row["shape"], row["file"]) for row in rows] == [
        (KEY_A, [2, 4], names[0]),
        (KEY_B, [6, 2], names[1]),
    ]


@pytest.mark.parametrize(
    "problem",
    ["missing_shard", "missing_key", "extra_key", "wrong_shard", "duplicate_key"],
)
def test_shard_index_rejects_inconsistent_contents(inspector, tmp_path, problem):
    first, second = {KEY_A: torch.ones(2, 4)}, {KEY_B: torch.ones(6, 2)}
    if problem == "extra_key":
        second["extra"] = torch.ones(1)
    elif problem == "wrong_shard":
        first, second = second, first
    elif problem == "duplicate_key":
        second[KEY_A] = torch.ones(2, 4)
    index, names = write_index(tmp_path, first, second)
    if problem == "missing_shard":
        (tmp_path / names[1]).unlink()
        message = "Missing shard"
    elif problem == "missing_key":
        payload = json.loads(index.read_text(encoding="utf-8"))
        payload["weight_map"]["missing"] = names[0]
        index.write_text(json.dumps(payload), encoding="utf-8")
        message = "missing=\\['missing'\\]"
    elif problem == "extra_key":
        message = "extra=\\['extra'\\]"
    elif problem == "wrong_shard":
        message = "wrong_shard=\\['language_model"
    else:
        message = "Duplicate tensor key"
    with pytest.raises(ValueError, match=message):
        inspector.inspect_adapter(index)


def test_unindexed_safetensors_directory_also_rejects_duplicate_keys(inspector, tmp_path):
    safetensors.torch.save_file({KEY_A: torch.ones(2, 4)}, tmp_path / "first.safetensors")
    safetensors.torch.save_file({KEY_A: torch.ones(2, 4)}, tmp_path / "second.safetensors")
    with pytest.raises(ValueError, match="Duplicate tensor key"):
        inspector.inspect_adapter(tmp_path)


@pytest.mark.parametrize("wrapped,legacy", [(False, False), (True, False), (False, True)])
def test_pytorch_plain_wrapped_and_legacy_shapes(inspector, tmp_path, tensors, wrapped, legacy, capsys):
    path = tmp_path / "adapter_model.bin"
    state = {**tensors, "metadata": "not a tensor"}
    payload = {"state_dict": state, "epoch": 2} if wrapped else state
    torch.save(payload, path, _use_new_zipfile_serialization=not legacy)
    rows = inspector.inspect_adapter(path)
    assert [row["key"] for row in rows] == sorted(tensors)
    assert {row["key"]: row["shape"] for row in rows} == {key: list(value.shape) for key, value in tensors.items()}
    assert next(row for row in rows if row["key"] == KEY_A)["dtype"] == "torch.bfloat16"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Skipping non-tensor entry 'metadata'" in captured.err


@pytest.mark.parametrize("output_format", ["json", "csv"])
@pytest.mark.parametrize("output_file", [False, True])
def test_cli_exports_machine_readable_data_without_total_on_stdout(
    safetensors_file, tmp_path, output_format, output_file
):
    path = tmp_path / f"keys.{output_format}"
    args = [safetensors_file]
    args.extend(["--output", path] if output_file else ["--format", output_format])
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    assert "Total: 4 tensor keys" in result.stderr
    assert "Total:" not in result.stdout
    if output_file:
        assert result.stdout == ""
    data = path.read_text(encoding="utf-8") if output_file else result.stdout
    if output_format == "json":
        rows = json.loads(data)
    else:
        rows = list(csv.DictReader(io.StringIO(data)))
        for row in rows:
            row["shape"] = json.loads(row["shape"])
    assert [row["key"] for row in rows] == sorted([KEY_A, KEY_B, "scalar", "empty"])
    assert next(row for row in rows if row["key"] == "scalar")["shape"] == []
    assert next(row for row in rows if row["key"] == "empty")["shape"] == [0, 3]


def test_cli_refuses_overwriting_existing_output(safetensors_file, tmp_path):
    output = tmp_path / "keys.json"
    original = b"previous result must remain unchanged\n"
    output.write_bytes(original)
    result = run_cli(safetensors_file, "--output", output)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Total:" not in result.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize("use_directory", [False, True])
def test_same_named_shards_in_subdirectories_keep_distinct_relative_paths(inspector, tmp_path, use_directory):
    mapping = {KEY_A: "first/shard.safetensors", KEY_B: "second/shard.safetensors"}
    for key, relative in mapping.items():
        path = tmp_path / relative
        path.parent.mkdir()
        safetensors.torch.save_file({key: torch.ones(2, 3)}, path)
    index = tmp_path / "adapter_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")
    rows = inspector.inspect_adapter(tmp_path if use_directory else index)
    assert {row["key"]: row["file"] for row in rows} == mapping


@pytest.mark.parametrize("location", ["header", "index"])
def test_duplicate_json_keys_are_rejected_instead_of_silently_collapsed(
    inspector, safetensors_file, tmp_path, location
):
    if location == "header":
        data = safetensors_file.read_bytes()
        length = struct.unpack("<Q", data[:8])[0]
        original = data[8 : 8 + length].decode("utf-8").strip()
        entry = json.loads(original)["scalar"]
        header = (original[:-1] + ',"scalar":' + json.dumps(entry) + "}").encode("utf-8")
        safetensors_file.write_bytes(struct.pack("<Q", len(header)) + header + data[8 + length :])
        path = safetensors_file
    else:
        path = tmp_path / "adapter_model.safetensors.index.json"
        name = json.dumps(safetensors_file.name)
        path.write_text('{"weight_map":{"same":' + name + ',"same":' + name + "}}", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        inspector.inspect_adapter(path)
