#!/usr/bin/env python3
"""Convert a ModelSlim W8A8_DYNAMIC checkpoint to BF16 safetensors.

The converter is intentionally model-loading-free: it reads the ModelSlim
description and safetensors index directly, dequantizes one output shard at a
time on CPU, and keeps non-floating tensors (for example routing tables) in
their original dtype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOGGER = logging.getLogger("dequantize_modelslim_w8a8")

QUANT_DESCRIPTION_NAME = "quant_model_description.json"
INPUT_INDEX_CANDIDATES = (
    "quant_model_weights.safetensors.index.json",
    "model.safetensors.index.json",
)
OUTPUT_INDEX_NAME = "model.safetensors.index.json"
MODEL_QUANT_TYPE = "W8A8_DYNAMIC"
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}
DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
SIZE_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    source_shard: str
    shape: tuple[int, ...]
    source_dtype: str
    output_dtype: str
    output_nbytes: int
    quantized: bool


@dataclass(frozen=True)
class CheckpointPlan:
    input_dir: Path
    input_index: Path
    description_path: Path
    description: dict[str, Any]
    weight_map: dict[str, str]
    quantized_weights: frozenset[str]
    dropped_quant_params: frozenset[str]
    tensors: tuple[TensorSpec, ...]

    @property
    def output_nbytes(self) -> int:
        return sum(spec.output_nbytes for spec in self.tensors)


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*", value, flags=re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}; examples: 5GB, 2GiB")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    result = int(number * SIZE_UNITS[unit])
    if result <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return result


def format_size(value: int) -> str:
    for unit, divisor in (("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value >= divisor:
            return f"{value / divisor:.2f} {unit}"
    return f"{value} B"


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _find_input_index(input_dir: Path) -> Path:
    for name in INPUT_INDEX_CANDIDATES:
        candidate = input_dir / name
        if candidate.is_file():
            return candidate
    matches = sorted(input_dir.glob("*.safetensors.index.json"))
    if len(matches) == 1:
        return matches[0]
    names = ", ".join(INPUT_INDEX_CANDIDATES)
    raise FileNotFoundError(f"cannot find a unique safetensors index in {input_dir}; tried {names}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def inspect_checkpoint(input_dir: Path) -> CheckpointPlan:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    description_path = input_dir / QUANT_DESCRIPTION_NAME
    if not description_path.is_file():
        raise FileNotFoundError(f"missing ModelSlim description: {description_path}")
    description = _load_json(description_path)
    if description.get("model_quant_type") != MODEL_QUANT_TYPE:
        raise ValueError(f"expected model_quant_type={MODEL_QUANT_TYPE!r}, got {description.get('model_quant_type')!r}")
    if description.get("group_size", 0) not in (0, None):
        raise NotImplementedError("this converter currently supports per-channel W8A8 weights (group_size=0) only")

    input_index = _find_input_index(input_dir)
    index_data = _load_json(input_index)
    weight_map_value = index_data.get("weight_map")
    if not isinstance(weight_map_value, dict) or not weight_map_value:
        raise ValueError(f"missing or empty weight_map in {input_index}")
    weight_map = {str(name): str(shard) for name, shard in weight_map_value.items()}

    quantized_weights = frozenset(
        name
        for name, quant_type in description.items()
        if quant_type == MODEL_QUANT_TYPE and name.endswith(".weight") and name in weight_map
    )
    if not quantized_weights:
        raise ValueError("the quantization description contains no indexed W8A8_DYNAMIC weight tensors")

    dropped_quant_params: set[str] = set()
    for name in quantized_weights:
        prefix = name.removesuffix(".weight")
        for suffix in (".weight_scale", ".weight_offset"):
            parameter_name = prefix + suffix
            if parameter_name not in weight_map:
                raise ValueError(f"missing {parameter_name!r} for quantized tensor {name!r}")
            if weight_map[parameter_name] != weight_map[name]:
                raise ValueError(f"{parameter_name!r} and {name!r} must be stored in the same input shard")
            dropped_quant_params.add(parameter_name)

    indexed_quant_names = {name for name in weight_map if description.get(name) == MODEL_QUANT_TYPE}
    expected_quant_names = set(quantized_weights) | dropped_quant_params
    unexpected_quant_names = indexed_quant_names - expected_quant_names
    if unexpected_quant_names:
        sample = sorted(unexpected_quant_names)[:5]
        raise NotImplementedError(f"unsupported W8A8_DYNAMIC tensor names: {sample}")

    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        names_by_shard[shard].append(name)

    header_info: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard, expected_names in names_by_shard.items():
        shard_path = input_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing input shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            actual_names = set(handle.keys())
            missing = set(expected_names) - actual_names
            if missing:
                raise ValueError(f"{shard_path} is missing indexed tensors: {sorted(missing)[:5]}")
            for name in expected_names:
                tensor_slice = handle.get_slice(name)
                header_info[name] = (tuple(tensor_slice.get_shape()), str(tensor_slice.get_dtype()))

    specs: list[TensorSpec] = []
    for name, source_shard in weight_map.items():
        if name in dropped_quant_params:
            continue
        shape, source_dtype = header_info[name]
        quantized = name in quantized_weights
        if quantized:
            if source_dtype != "I8":
                raise TypeError(f"quantized tensor {name!r} must be I8, got {source_dtype}")
            if len(shape) != 2:
                raise ValueError(f"quantized tensor {name!r} must be 2D, got shape {shape}")
            output_dtype = "BF16"
        elif source_dtype in FLOAT_DTYPES:
            output_dtype = "BF16"
        elif source_dtype in DTYPE_BYTES:
            output_dtype = source_dtype
        else:
            raise TypeError(f"unsupported safetensors dtype {source_dtype!r} for tensor {name!r}")
        specs.append(
            TensorSpec(
                name=name,
                source_shard=source_shard,
                shape=shape,
                source_dtype=source_dtype,
                output_dtype=output_dtype,
                output_nbytes=_numel(shape) * DTYPE_BYTES[output_dtype],
                quantized=quantized,
            )
        )

    return CheckpointPlan(
        input_dir=input_dir,
        input_index=input_index,
        description_path=description_path,
        description=description,
        weight_map=weight_map,
        quantized_weights=quantized_weights,
        dropped_quant_params=frozenset(dropped_quant_params),
        tensors=tuple(specs),
    )


def plan_output_shards(tensors: tuple[TensorSpec, ...], max_shard_size: int) -> list[list[TensorSpec]]:
    shards: list[list[TensorSpec]] = []
    current: list[TensorSpec] = []
    current_size = 0
    for spec in tensors:
        if spec.output_nbytes > max_shard_size:
            raise ValueError(
                f"tensor {spec.name!r} is {format_size(spec.output_nbytes)}, larger than --max-shard-size "
                f"{format_size(max_shard_size)}"
            )
        if current and current_size + spec.output_nbytes > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(spec)
        current_size += spec.output_nbytes
    if current:
        shards.append(current)
    return shards


def _reshape_qparam(parameter: torch.Tensor, weight: torch.Tensor, name: str) -> torch.Tensor:
    if parameter.ndim == 0 or parameter.numel() == 1:
        return parameter
    if parameter.ndim == 1 and parameter.shape[0] == weight.shape[0]:
        return parameter.reshape(weight.shape[0], *([1] * (weight.ndim - 1)))
    try:
        torch.broadcast_shapes(parameter.shape, weight.shape)
    except RuntimeError as error:
        raise ValueError(
            f"quantization parameter {name!r} with shape {tuple(parameter.shape)} cannot broadcast to "
            f"weight shape {tuple(weight.shape)}"
        ) from error
    return parameter


def dequantize_weight(weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, name: str) -> torch.Tensor:
    """Apply the inverse of Q = round(V / scale) + offset."""
    if weight.dtype != torch.int8:
        raise TypeError(f"{name!r}: expected int8 weight, got {weight.dtype}")
    scale = _reshape_qparam(scale, weight, name + ".weight_scale").to(torch.float32)
    offset = _reshape_qparam(offset, weight, name + ".weight_offset").to(torch.float32)
    result = weight.to(torch.float32)
    result.sub_(offset).mul_(scale)
    return result.to(torch.bfloat16).contiguous()


def _expected_shard_name(index: int, total: int) -> str:
    return f"model-{index:05d}-of-{total:05d}.safetensors"


def _validate_existing_shard(path: Path, specs: list[TensorSpec]) -> bool:
    if not path.is_file():
        return False
    expected = {spec.name: spec for spec in specs}
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected):
                return False
            for name, spec in expected.items():
                tensor_slice = handle.get_slice(name)
                if tuple(tensor_slice.get_shape()) != spec.shape:
                    return False
                if str(tensor_slice.get_dtype()) != spec.output_dtype:
                    return False
    except Exception:
        return False
    return True


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_output_tensor(spec: TensorSpec, handles: dict[str, Any], plan: CheckpointPlan) -> torch.Tensor:
    handle = handles[spec.source_shard]
    tensor = handle.get_tensor(spec.name)
    if spec.quantized:
        prefix = spec.name.removesuffix(".weight")
        scale = handle.get_tensor(prefix + ".weight_scale")
        offset = handle.get_tensor(prefix + ".weight_offset")
        return dequantize_weight(tensor, scale, offset, spec.name)
    if tensor.is_floating_point():
        return tensor.to(torch.bfloat16).contiguous()
    return tensor.contiguous()


def convert_shards(
    plan: CheckpointPlan,
    output_dir: Path,
    output_shards: list[list[TensorSpec]],
    resume: bool,
) -> tuple[dict[str, str], int]:
    output_weight_map: dict[str, str] = {}
    skipped_bytes = 0
    total_shards = len(output_shards)

    for shard_index, specs in enumerate(output_shards, start=1):
        shard_name = _expected_shard_name(shard_index, total_shards)
        shard_path = output_dir / shard_name
        for spec in specs:
            output_weight_map[spec.name] = shard_name

        if shard_path.exists():
            if resume and _validate_existing_shard(shard_path, specs):
                shard_bytes = sum(spec.output_nbytes for spec in specs)
                skipped_bytes += shard_bytes
                LOGGER.info("[%d/%d] resume: valid shard already exists: %s", shard_index, total_shards, shard_path)
                continue
            reason = "is invalid" if resume else "already exists"
            raise FileExistsError(f"output shard {shard_path} {reason}; use a clean directory or --resume")

        shard_bytes = sum(spec.output_nbytes for spec in specs)
        LOGGER.info(
            "[%d/%d] converting %d tensors (%s) -> %s",
            shard_index,
            total_shards,
            len(specs),
            format_size(shard_bytes),
            shard_path,
        )
        source_shards = list(dict.fromkeys(spec.source_shard for spec in specs))
        with ExitStack() as stack:
            handles = {
                shard: stack.enter_context(safe_open(plan.input_dir / shard, framework="pt", device="cpu"))
                for shard in source_shards
            }
            output_tensors = {spec.name: _load_output_tensor(spec, handles, plan) for spec in specs}
            temporary_path = shard_path.with_name(shard_path.name + ".tmp")
            if temporary_path.exists():
                temporary_path.unlink()
            save_file(output_tensors, temporary_path, metadata={"format": "pt"})
            os.replace(temporary_path, shard_path)

    return output_weight_map, skipped_bytes


def _is_weight_artifact(relative_path: Path, input_index: Path) -> bool:
    if relative_path.name in {QUANT_DESCRIPTION_NAME, input_index.name, OUTPUT_INDEX_NAME}:
        return True
    if relative_path.suffix == ".safetensors":
        return True
    return relative_path.name.endswith(".safetensors.index.json")


def copy_auxiliary_files(plan: CheckpointPlan, output_dir: Path, preserve_expert_dtype: bool) -> None:
    for source in plan.input_dir.rglob("*"):
        if not source.is_file():
            continue
        relative_path = source.relative_to(plan.input_dir)
        if _is_weight_artifact(relative_path, plan.input_index):
            continue
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    config_path = output_dir / "config.json"
    if config_path.is_file():
        config = _load_json(config_path)
        for key in ("quantization_config", "quantize", "compression_config", "quant_method"):
            config.pop(key, None)
        config["torch_dtype"] = "bfloat16"
        if not preserve_expert_dtype and "expert_dtype" in config:
            config["expert_dtype"] = "bf16"
        _atomic_write_json(config_path, config)


def _check_output_directory(input_dir: Path, output_dir: Path, resume: bool) -> None:
    output_dir = output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("output directory must not be the input directory or a child of it")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use a clean directory or --resume")


def run(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    _check_output_directory(input_dir, output_dir, args.resume)

    LOGGER.info("inspecting checkpoint metadata in %s", input_dir)
    plan = inspect_checkpoint(input_dir)
    output_shards = plan_output_shards(plan.tensors, args.max_shard_size)
    LOGGER.info(
        "plan: %d input tensors, %d quantized weights, %d dropped scale/offset tensors",
        len(plan.weight_map),
        len(plan.quantized_weights),
        len(plan.dropped_quant_params),
    )
    LOGGER.info(
        "plan: %d output tensors, %d output shards, estimated output size %s",
        len(plan.tensors),
        len(output_shards),
        format_size(plan.output_nbytes),
    )

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    disk_usage = shutil.disk_usage(output_dir)
    existing_valid_bytes = 0
    if args.resume:
        for index, specs in enumerate(output_shards, start=1):
            path = output_dir / _expected_shard_name(index, len(output_shards))
            if _validate_existing_shard(path, specs):
                existing_valid_bytes += sum(spec.output_nbytes for spec in specs)
    remaining_bytes = plan.output_nbytes - existing_valid_bytes
    if disk_usage.free < remaining_bytes:
        raise OSError(
            f"insufficient disk space: need approximately {format_size(remaining_bytes)}, "
            f"but only {format_size(disk_usage.free)} is free"
        )

    output_weight_map, skipped_bytes = convert_shards(plan, output_dir, output_shards, args.resume)
    LOGGER.info("all weight shards complete; resumed data: %s", format_size(skipped_bytes))

    if not args.no_copy_aux_files:
        copy_auxiliary_files(plan, output_dir, args.preserve_expert_dtype)

    index = {
        "metadata": {"total_size": plan.output_nbytes},
        "weight_map": output_weight_map,
    }
    _atomic_write_json(output_dir / OUTPUT_INDEX_NAME, index)
    info = {
        "source": str(plan.input_dir),
        "source_index": plan.input_index.name,
        "source_description_sha256": _sha256(plan.description_path),
        "source_quant_type": plan.description.get("model_quant_type"),
        "dequantization_formula": "bf16((int8_weight - weight_offset) * weight_scale)",
        "output_dtype": "bfloat16",
        "quantized_weight_count": len(plan.quantized_weights),
        "output_tensor_count": len(plan.tensors),
        "output_shard_count": len(output_shards),
        "output_total_size": plan.output_nbytes,
    }
    _atomic_write_json(output_dir / "dequantization_info.json", info)
    LOGGER.info("conversion complete: %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="ModelSlim W8A8_DYNAMIC checkpoint directory")
    parser.add_argument("output_dir", type=Path, help="new directory for the BF16 checkpoint")
    parser.add_argument(
        "--max-shard-size",
        type=parse_size,
        default=parse_size("5GiB"),
        help="maximum tensor payload per output shard (default: 5GiB)",
    )
    parser.add_argument("--resume", action="store_true", help="reuse already completed, header-valid output shards")
    parser.add_argument("--dry-run", action="store_true", help="validate metadata and print the plan without writing")
    parser.add_argument(
        "--no-copy-aux-files",
        action="store_true",
        help="do not copy tokenizer/config/other non-weight files",
    )
    parser.add_argument(
        "--preserve-expert-dtype",
        action="store_true",
        help="keep config.json expert_dtype instead of changing it to bf16",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        args = build_parser().parse_args(argv)
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError, TypeError, NotImplementedError, OSError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
