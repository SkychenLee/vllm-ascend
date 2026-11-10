#!/usr/bin/env python3
"""List every original tensor key and shape in a local LoRA adapter.

Python 3.10+. Safetensors inspection uses only the standard library and reads
the header, not tensor data. PyTorch files require torch >= 2.1; zip checkpoints
use mmap with weights_only=True and map_location='meta'. Legacy PyTorch files
may use temporary CPU memory. No NPU or vLLM installation is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from collections.abc import Mapping
from pathlib import Path

MAX_HEADER_BYTES = 100_000_000
STANDARD_FILES = (
    "adapter_model.safetensors.index.json",
    "adapter_model.safetensors",
    "adapter_model.bin.index.json",
    "adapter_model.bin",
)


def unique_json_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_safetensors(path: Path) -> list[dict]:
    """Read key/shape/dtype metadata without materializing tensor contents."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > MAX_HEADER_BYTES or length > size - 8:
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(length), object_pairs_hook=unique_json_object)
    if not isinstance(header, dict):
        raise ValueError(f"Expected a safetensors header object: {path}")
    rows = []
    for key, info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(f"Invalid tensor entry {key!r} in {path}")
        shape, dtype = info.get("shape"), info.get("dtype")
        if (
            not isinstance(shape, list)
            or not all(type(dimension) is int and dimension >= 0 for dimension in shape)
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"Invalid tensor shape/dtype for {key!r} in {path}")
        offsets = info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(offset) is int for offset in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length
        ):
            raise ValueError(f"Invalid tensor data offsets for {key!r} in {path}")
        rows.append(dict(key=key, shape=shape, dtype=dtype, file=path.name))
    return rows


def read_pytorch(path: Path) -> list[dict]:
    try:
        import torch
    except ImportError as exc:
        raise ValueError("PyTorch files require torch >= 2.1. Install it with: pip install torch") from exc
    with path.open("rb") as stream:
        is_zip = stream.read(4) == b"PK\x03\x04"
    # Never fall back to unrestricted pickle loading.
    payload = torch.load(path, map_location="meta", weights_only=True, mmap=is_zip)
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a state_dict mapping in {path}")
    rows = []
    for key, value in payload.items():
        if torch.is_tensor(value):
            if not isinstance(key, str):
                raise ValueError(f"Expected a string tensor key in {path}: {key!r}")
            rows.append(
                dict(
                    key=key,
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    file=path.name,
                )
            )
        else:
            print(f"Skipping non-tensor entry {key!r} in {path.name}", file=sys.stderr)
    return rows


def resolve_files(path: Path) -> tuple[list[Path], dict[str, str] | None]:
    """Prefer the standard safetensors representation when both formats exist."""
    if path.is_dir():
        standard = next((path / name for name in STANDARD_FILES if (path / name).is_file()), None)
        if standard is not None:
            return resolve_files(standard)
        tensors = sorted(path.glob("*.safetensors"))
        if tensors:
            return tensors, None
        binaries = sorted(file for file in path.iterdir() if file.suffix.lower() in (".bin", ".pt", ".pth"))
        if len(binaries) == 1:
            return binaries, None
        raise ValueError(f"No unambiguous adapter weights in {path}; pass a weight file or shard index explicitly.")
    if not path.is_file():
        raise ValueError(f"File or directory does not exist: {path}")
    if path.name.endswith(".index.json"):
        index = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_object)
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing or empty weight_map in {path}")
        files = {}
        for key, filename in mapping.items():
            if not isinstance(key, str) or not isinstance(filename, str):
                raise ValueError(f"Invalid weight_map entry in {path}")
            shard = (path.parent / filename).resolve()
            if Path(filename).is_absolute() or not shard.is_relative_to(path.parent.resolve()):
                raise ValueError(f"Shard path must stay inside the adapter directory: {filename}")
            if not shard.is_file():
                raise ValueError(f"Missing shard: {shard}")
            files[filename] = shard
        return sorted(set(files.values())), {key: str(files[filename]) for key, filename in mapping.items()}
    return [path], None


def inspect_adapter(path: str | Path) -> list[dict]:
    source = Path(path).expanduser().resolve()
    directory = source if source.is_dir() else source.parent
    files, expected = resolve_files(source)
    rows, seen = [], {}
    for file in files:
        if file.suffix.lower() == ".safetensors":
            entries = read_safetensors(file)
        elif file.suffix.lower() in (".bin", ".pt", ".pth"):
            entries = read_pytorch(file)
        else:
            raise ValueError(f"Unsupported weight format: {file}")
        for entry in entries:
            key = entry["key"]
            if key in seen:
                raise ValueError(f"Duplicate tensor key {key!r} in {seen[key]} and {file}")
            seen[key] = str(file)
            entry["file"] = file.relative_to(directory).as_posix()
            rows.append(entry)
    if expected is not None and seen != expected:
        missing = sorted(expected.keys() - seen.keys())[:5]
        extra = sorted(seen.keys() - expected.keys())[:5]
        wrong_shard = sorted(key for key in seen.keys() & expected.keys() if seen[key] != expected[key])[:5]
        raise ValueError(f"Shard index mismatch: missing={missing}, extra={extra}, wrong_shard={wrong_shard}")
    if not rows:
        raise ValueError("No tensor weights found.")
    return sorted(rows, key=lambda entry: entry["key"])


def write_rows(rows: list[dict], output_format: str, stream) -> None:
    if output_format == "json":
        json.dump(rows, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    elif output_format == "csv":
        writer = csv.DictWriter(stream, fieldnames=("key", "shape", "dtype", "file"))
        writer.writeheader()
        writer.writerows({**row, "shape": json.dumps(row["shape"])} for row in rows)
    else:
        columns = [(row["key"], str(row["shape"]), row["dtype"], row["file"]) for row in rows]
        columns.insert(0, ("KEY", "SHAPE", "DTYPE", "FILE"))
        widths = [max(len(row[i]) for row in columns) for i in range(4)]
        for row in columns:
            stream.write("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter", help="Local adapter directory, weight file, or *.index.json")
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        help="Default: table; infer JSON/CSV from --output",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Save the result to a new file (default: stdout)",
    )
    args = parser.parse_args()
    output_format = args.format or (
        args.output.suffix.lower()[1:] if args.output and args.output.suffix.lower() in (".json", ".csv") else "table"
    )
    try:
        rows = inspect_adapter(args.adapter)
        if args.output:
            # Do not accidentally overwrite adapter weights/config or an earlier dump.
            with args.output.open("x", encoding="utf-8", newline="") as stream:
                write_rows(rows, output_format, stream)
        else:
            write_rows(rows, output_format, sys.stdout)
        print(f"Total: {len(rows)} tensor keys", file=sys.stderr)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
