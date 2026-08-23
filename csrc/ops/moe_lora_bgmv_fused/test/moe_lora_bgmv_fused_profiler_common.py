#!/usr/bin/env python3
"""Real-weight profiler helpers for fused MoE LoRA BGMV."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch_npu
from safetensors import safe_open


PROFILER_SCHEDULE_WARMUP = 5
PROFILER_SCHEDULE_ACTIVE = 5
ROUTER_KEY = "model.language_model.layers.0.mlp.gate.weight"
GATE_UP_KEY = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DOWN_KEY = "model.language_model.layers.0.mlp.experts.down_proj"
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class RealWeights:
    router: torch.Tensor
    w13_a: torch.Tensor
    w13_b: torch.Tensor
    w2_a: torch.Tensor
    w2_b: torch.Tensor
    metadata: dict[str, Any]


def load_custom_library() -> str:
    build_dir = os.environ.get("VLLM_ASCEND_BUILD_DIR")
    if not build_dir:
        raise RuntimeError("set VLLM_ASCEND_BUILD_DIR to the CMake build directory")
    torch.ops.load_library(
        os.path.join(build_dir, "lib", "libvllm_ascend_kernels.so")
    )
    modules = glob.glob(os.path.join(build_dir, "vllm_ascend_C*.so"))
    if len(modules) != 1:
        raise RuntimeError(f"expected one vllm_ascend_C module, got {modules}")
    torch.ops.load_library(modules[0])
    return modules[0]


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix != ".jsonl":
        raise ValueError(f"only JSONL performance cases are supported: {path}")
    with path.open(encoding="utf-8") as case_file:
        cases = [json.loads(line) for line in case_file if line.strip()]
    if len(cases) < 8:
        raise ValueError("at least 8 performance cases are required")
    return cases


def specs(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = {item["name"]: item for item in case["inputs"]}
    if len(values) != len(case["inputs"]):
        raise ValueError("case input names must be unique")
    return values


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _load_tensor(
    model_root: Path,
    weight_map: dict[str, str],
    key: str,
) -> torch.Tensor:
    shard = model_root / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_tensor(key).contiguous()


def _load_rank_weights(
    model_root: Path,
    weight_map: dict[str, str],
    key: str,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shard = model_root / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        tensor_slice = checkpoint.get_slice(key)
        left = tensor_slice[:, :rank, :].contiguous()
        right = tensor_slice[:, :, :rank].contiguous()
    return left, right


def load_real_weights(model_root: Path, rank: int) -> RealWeights:
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    if text_config.get("model_type") != "qwen3_5_moe_text":
        raise ValueError("checkpoint must be Qwen3.5 MoE")
    hidden_size = int(text_config["hidden_size"])
    expert_size = int(text_config["moe_intermediate_size"])
    num_experts = int(text_config["num_experts"])
    top_k = int(text_config["num_experts_per_tok"])

    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    router = _load_tensor(model_root, weight_map, ROUTER_KEY)
    w13_a, w13_b = _load_rank_weights(
        model_root, weight_map, GATE_UP_KEY, rank
    )
    w2_a, w2_b = _load_rank_weights(model_root, weight_map, DOWN_KEY, rank)
    weights = {
        "w13_a": w13_a,
        "w13_b": w13_b,
        "w2_a": w2_a,
        "w2_b": w2_b,
    }
    expected = {
        "router": (num_experts, hidden_size),
        "w13_a": (num_experts, rank, hidden_size),
        "w13_b": (num_experts, 2 * expert_size, rank),
        "w2_a": (num_experts, rank, expert_size),
        "w2_b": (num_experts, hidden_size, rank),
    }
    tensors = {"router": router, **weights}
    for name, shape in expected.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"unexpected {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )
    metadata = {
        "model_root": str(model_root),
        "model_type": text_config["model_type"],
        "hidden_size": hidden_size,
        "expert_size": expert_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "rank": rank,
        "keys": {"router": ROUTER_KEY, "w13": GATE_UP_KEY, "w2": DOWN_KEY},
        "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "fingerprints": {
            name: tensor_fingerprint(tensor) for name, tensor in tensors.items()
        },
    }
    return RealWeights(router=router, metadata=metadata, **weights)


def build_route(
    weights: RealWeights,
    tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = (tokens + weights.router.shape[0] - 1) // weights.router.shape[0]
    hidden = weights.router.repeat((repeats, 1))[:tokens].contiguous()
    logits = hidden.float() @ weights.router.float().T
    topk_ids = torch.topk(logits, top_k, dim=1).indices.to(torch.int32)
    flat_experts = topk_ids.reshape(-1)
    permutation = torch.argsort(flat_experts, stable=True)
    sorted_hidden = hidden.repeat_interleave(top_k, dim=0)[permutation].contiguous()
    sorted_experts = flat_experts[permutation].contiguous()
    return sorted_hidden, sorted_experts


def forward_separate(state: dict[str, Any]) -> torch.Tensor:
    torch.ops._C_ascend.bgmv_shrink(
        state["x"],
        state["a"],
        state["indices"],
        state["shrink"],
        state["scale"],
    )
    return torch.ops._C_ascend.bgmv_expand(
        state["shrink"],
        state["b"],
        state["indices"],
        state["separate_y"],
        0,
        state["output_size"],
    )


def forward_fused(state: dict[str, Any]) -> torch.Tensor:
    return torch.ops._C_ascend.moe_lora_bgmv_fused(
        state["x"],
        state["a"],
        state["b"],
        state["indices"],
        state["fused_y"],
        0,
        state["output_size"],
        state["scale"],
    )


def compute_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_err = (actual_f - expected_f).abs()
    rel_err = abs_err / (expected_f.abs() + 1e-7)
    return {
        "max_abs_err": abs_err.max().item(),
        "MERE": rel_err.mean().item(),
        "MARE": rel_err.max().item(),
    }


def _sum_total_time_us(csv_path: str) -> float:
    total = 0.0
    with open(csv_path, encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        key = next(
            (
                name
                for name in (reader.fieldnames or [])
                if "Total" in name.lstrip("\ufeff") and "us" in name.lower()
            ),
            None,
        )
        if key is None:
            raise RuntimeError(f"Total Time(us) column not found in {csv_path}")
        for row in reader:
            value = str(row.get(key, "")).strip()
            if value:
                total += float(value)
    return total


def _newest_statistic(root: str) -> str:
    matches = glob.glob(os.path.join(root, "**", "op_statistic.csv"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"op_statistic.csv not found under {root}")
    return max(matches, key=os.path.getmtime)


def read_op_statistics(handler_dir: str) -> list[dict[str, float | str]]:
    csv_path = _newest_statistic(handler_dir)
    statistics: list[dict[str, float | str]] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        total_key = next(
            (
                name
                for name in (reader.fieldnames or [])
                if "Total" in name.lstrip("\ufeff") and "us" in name.lower()
            ),
            None,
        )
        if total_key is None:
            raise RuntimeError(f"Total Time(us) column not found in {csv_path}")
        for row in reader:
            total = str(row.get(total_key, "")).strip()
            if total:
                statistics.append(
                    {
                        "op_type": str(row.get("OP Type", "unknown")),
                        "total_us": float(total) / PROFILER_SCHEDULE_ACTIVE,
                    }
                )
    return statistics


def profile_forward(forward: Callable[[], torch.Tensor], handler_dir: str) -> float:
    if os.path.isdir(handler_dir):
        shutil.rmtree(handler_dir)
    os.makedirs(handler_dir, exist_ok=True)
    schedule = torch_npu.profiler.schedule(
        wait=0,
        warmup=PROFILER_SCHEDULE_WARMUP,
        active=PROFILER_SCHEDULE_ACTIVE,
        repeat=1,
        skip_first=0,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(handler_dir),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
    ) as profiler:
        for _ in range(PROFILER_SCHEDULE_WARMUP + PROFILER_SCHEDULE_ACTIVE):
            forward()
            torch.npu.synchronize()
            profiler.step()
    torch.npu.synchronize()
    deadline = time.time() + 120
    while True:
        try:
            csv_path = _newest_statistic(handler_dir)
            break
        except FileNotFoundError:
            if time.time() >= deadline:
                raise
            time.sleep(0.2)
    return _sum_total_time_us(csv_path) / PROFILER_SCHEDULE_ACTIVE


def summarize_breakdown(
    op_statistics: list[dict[str, float | str]],
) -> dict[str, float]:
    breakdown = {"fused": 0.0, "shrink": 0.0, "expand": 0.0, "other": 0.0}
    for row in op_statistics:
        op_type = str(row["op_type"])
        total_us = float(row["total_us"])
        if op_type.startswith("moe_lora_bgmv_fused"):
            category = "fused"
        elif op_type.startswith("bgmv_shrink"):
            category = "shrink"
        elif op_type.startswith("bgmv_expand"):
            category = "expand"
        else:
            category = "other"
        breakdown[category] += total_us
    return breakdown
