#!/usr/bin/env python3
"""Profiler helpers for fused MoE-LoRA routing metadata."""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import time
from collections.abc import Callable

import torch
import torch_npu

PROFILER_SCHEDULE_WARMUP = 5
PROFILER_SCHEDULE_ACTIVE = 5


def load_cases(path: str) -> list[dict]:
    if not path.endswith(".jsonl"):
        raise ValueError(f"only JSONL performance cases are supported: {path}")
    with open(path, encoding="utf-8") as case_file:
        return [json.loads(line) for line in case_file if line.strip()]


def load_custom_library() -> str:
    build_dir = os.environ.get("VLLM_ASCEND_BUILD_DIR")
    if not build_dir:
        raise RuntimeError("set VLLM_ASCEND_BUILD_DIR to the CMake build directory")
    torch.ops.load_library(os.path.join(build_dir, "lib", "libvllm_ascend_kernels.so"))
    modules = glob.glob(os.path.join(build_dir, "vllm_ascend_C*.so"))
    if len(modules) != 1:
        raise RuntimeError(f"expected one vllm_ascend_C module, got {modules}")
    torch.ops.load_library(modules[0])
    return modules[0]


def _specs(case: dict) -> dict[str, dict]:
    return {item["name"]: item for item in case["inputs"]}


def build_inputs(case: dict, device: torch.device, seed: int) -> dict:
    specs = _specs(case)
    topk_shape = tuple(specs["topk_ids"]["shape"])
    tokens, top_k = topk_shape
    num_pairs = tokens * top_k
    num_experts = int(specs["num_experts"]["value"])
    max_loras = int(specs["adapter_enabled"]["shape"][0])
    adapter_dtype = torch.bool if specs["adapter_enabled"]["dtype"] == "bool" else torch.int8
    generator = torch.Generator().manual_seed(seed)

    expanded = torch.randperm(num_pairs, generator=generator, dtype=torch.int64).to(torch.int32)
    if num_pairs > 1:
        negative = (torch.arange(num_pairs) % 5 == 1) & (expanded != 0)
        expanded[negative] = -expanded[negative]
    topk_ids = torch.randint(
        0, num_experts, topk_shape, generator=generator, dtype=torch.int32
    )
    token_lora = torch.randint(
        -1, max_loras, (tokens,), generator=generator, dtype=torch.int64
    )
    adapter_enabled = torch.randint(
        0, 2, (max_loras,), generator=generator, dtype=torch.int8
    ).to(adapter_dtype)
    values = (expanded, topk_ids, token_lora, adapter_enabled)
    return {
        "expanded": values[0].to(device),
        "topk_ids": values[1].to(device),
        "token_lora": values[2].to(device),
        "adapter_enabled": values[3].to(device),
        "num_experts": num_experts,
        "tokens": tokens,
        "top_k": top_k,
        "dtype": specs["adapter_enabled"]["dtype"],
    }


def custom_forward(inputs: dict) -> torch.Tensor:
    return torch.ops._C_ascend.moe_lora_build_combined_idx(
        inputs["expanded"],
        inputs["topk_ids"],
        inputs["token_lora"],
        inputs["adapter_enabled"],
        inputs["num_experts"],
    )


def baseline_forward(inputs: dict) -> torch.Tensor:
    inv_perm = torch.argsort(torch.abs(inputs["expanded"]))
    expert_per_row = inputs["topk_ids"].reshape(-1)[inv_perm].long()
    lora_per_row = inputs["token_lora"][inv_perm // inputs["top_k"]]
    safe_lora = lora_per_row.clamp(min=0)
    enabled = (lora_per_row >= 0) & inputs["adapter_enabled"][safe_lora].bool()
    return torch.where(
        enabled,
        safe_lora * inputs["num_experts"] + expert_per_row,
        torch.full_like(lora_per_row, -1),
    ).to(torch.int32).contiguous()


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


def read_op_statistics(handler_dir: str) -> list[dict[str, float | int | str]]:
    """Return per-active-step operator statistics from the newest trace."""
    csv_path = _newest_statistic(handler_dir)
    statistics: list[dict[str, float | int | str]] = []
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
            if not total:
                continue
            count = str(row.get("Count", "0")).strip()
            statistics.append(
                {
                    "op_type": str(row.get("OP Type", "unknown")),
                    "total_us": float(total) / PROFILER_SCHEDULE_ACTIVE,
                    "count": int(count or 0),
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
            # Keep asynchronous NPU work inside the profiler step that issued it.
            # Otherwise warmup work can leak into the active op_statistic window.
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


def render_report(results: list[dict], case_file: str, trace_root: str) -> str:
    ratios = [row["speedup"] for row in results]
    lines = [
        "# 性能评估结果",
        "",
        "- 无标杆等价接口；标杆为 NPU 小算子拼接：abs/argsort/gather/div/clamp/index/where/contiguous。",
        f"- 用例文件：`{case_file}`",
        f"- Trace：`{trace_root}`",
        "- 指标：op_statistic.csv 全部算子的 Total Time(us) 求和 / active(5)。",
        "- 固定 schedule：warmup=5、active=5、repeat=1。",
        "",
        "## 性能对比",
        "",
        "| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |",
        "| ---- | ----- | ----- | ------------- | -------- | ------ |",
    ]
    for row in results:
        lines.append(
            f"| {row['case']} | [{row['tokens']}, {row['top_k']}] | {row['dtype']} | "
            f"{row['custom_us']:.3f} | {row['baseline_us']:.3f} | {row['speedup']:.3f} |"
        )
    custom_better = sum(ratio > 1 for ratio in ratios)
    lines.extend(
        [
            "",
            "## 全量汇总",
            "",
            "| 指标 | 值 |",
            "| ---- | -- |",
            f"| 用例数 | {len(results)} |",
            f"| 平均加速比 | {sum(ratios) / len(ratios):.3f} |",
            f"| 自定义算子更优 | {custom_better} |",
            f"| 标杆更优 | {len(results) - custom_better} |",
            "",
            "### 按数据类型汇总",
            "",
            "| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |",
            "| ----- | ------ | ---------- | -------------- | -------- |",
        ]
    )
    for dtype in ("bool", "int8"):
        rows = [row for row in results if row["dtype"] == dtype]
        dtype_ratios = [row["speedup"] for row in rows]
        better = sum(ratio > 1 for ratio in dtype_ratios)
        lines.append(
            f"| {dtype} | {len(rows)} | {sum(dtype_ratios) / len(dtype_ratios):.3f} | "
            f"{better} | {len(rows) - better} |"
        )
    lines.extend(
        [
            "",
            "## 简短分析",
            "",
            "- 融合路径消除了 AiCPU argsort 及多个 AICore/AiCPU 调度切换。",
            "- 小 shape 主要受 kernel launch 固定开销影响。",
            "- 大 shape 的单核标量 GM gather/scatter 呈线性增长，是下一轮优化重点。",
            "",
        ]
    )
    return "\n".join(lines)
