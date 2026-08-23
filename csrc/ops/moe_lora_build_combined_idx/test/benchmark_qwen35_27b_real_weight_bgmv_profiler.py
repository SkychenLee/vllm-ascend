#!/usr/bin/env python3
"""Profile int32/int64 BGMV with real Qwen3.5-27B checkpoint values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open

from moe_lora_build_combined_idx_profiler_common import (
    load_custom_library,
    profile_forward,
    read_op_statistics,
)


DEFAULT_MODEL_ROOT = "/opt/wqy2/model-weights/Qwen3.5-27B"
EXPECTED_MODEL_TYPE = "qwen3_5_text"
GATE_KEY = "model.language_model.layers.0.mlp.gate_proj.weight"
UP_KEY = "model.language_model.layers.0.mlp.up_proj.weight"
DOWN_KEY = "model.language_model.layers.0.mlp.down_proj.weight"


@dataclass
class DenseWeights:
    x_source: torch.Tensor
    w13_a: torch.Tensor
    w13_b: torch.Tensor
    w2_a: torch.Tensor
    w2_b: torch.Tensor
    hidden_size: int
    intermediate_size: int
    rank: int


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as case_file:
        cases = [json.loads(line) for line in case_file if line.strip()]
    if len(cases) < 8:
        raise ValueError("dense real-weight profiler requires at least 8 JSONL cases")
    return cases


def case_attrs(case: dict[str, Any]) -> dict[str, Any]:
    return {item["name"]: item["value"] for item in case["inputs"]}


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_real_weights(
    model_root: Path,
    rank: int,
    max_tokens: int,
    device: torch.device,
) -> tuple[DenseWeights, dict[str, Any]]:
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    if text_config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"{model_root} is {text_config.get('model_type')}, "
            f"expected {EXPECTED_MODEL_TYPE}"
        )
    hidden_size = int(text_config["hidden_size"])
    intermediate_size = int(text_config["intermediate_size"])
    if rank <= 0 or rank > min(hidden_size, intermediate_size):
        raise ValueError(f"invalid rank {rank}")

    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    source_keys = (GATE_KEY, UP_KEY, DOWN_KEY)
    if any(key not in weight_map for key in source_keys):
        missing = [key for key in source_keys if key not in weight_map]
        raise KeyError(f"checkpoint keys missing: {missing}")
    source_shards = {key: weight_map[key] for key in source_keys}

    gate_path = model_root / source_shards[GATE_KEY]
    up_path = model_root / source_shards[UP_KEY]
    down_path = model_root / source_shards[DOWN_KEY]
    with safe_open(gate_path, framework="pt", device="cpu") as checkpoint:
        gate = checkpoint.get_slice(GATE_KEY)
        x_source = gate[:max_tokens, :].contiguous()
        w13_a_cpu = gate[:rank, :].contiguous()
        gate_b = gate[:, :rank].contiguous()
    with safe_open(up_path, framework="pt", device="cpu") as checkpoint:
        up = checkpoint.get_slice(UP_KEY)
        up_b = up[:, :rank].contiguous()
    with safe_open(down_path, framework="pt", device="cpu") as checkpoint:
        down = checkpoint.get_slice(DOWN_KEY)
        w2_a_cpu = down[:rank, :].contiguous()
        w2_b_cpu = down[:, :rank].contiguous()

    w13_b_cpu = torch.cat((gate_b, up_b), dim=0).contiguous()
    tensors = {
        "x_source": x_source,
        "w13_a": w13_a_cpu,
        "w13_b": w13_b_cpu,
        "w2_a": w2_a_cpu,
        "w2_b": w2_b_cpu,
    }
    expected_shapes = {
        "x_source": (max_tokens, hidden_size),
        "w13_a": (rank, hidden_size),
        "w13_b": (2 * intermediate_size, rank),
        "w2_a": (rank, intermediate_size),
        "w2_b": (hidden_size, rank),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"unexpected {name}: shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype}"
            )

    metadata = {
        "model_root": str(model_root),
        "model_type": text_config["model_type"],
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "rank": rank,
        "source_keys": {
            "gate": GATE_KEY,
            "up": UP_KEY,
            "down": DOWN_KEY,
        },
        "source_shards": source_shards,
        "source_shapes": {
            name: list(tensor.shape) for name, tensor in tensors.items()
        },
        "fingerprints": {
            name: tensor_fingerprint(tensor) for name, tensor in tensors.items()
        },
    }
    return DenseWeights(
        x_source=x_source.to(device),
        w13_a=w13_a_cpu.unsqueeze(0).to(device),
        w13_b=w13_b_cpu.unsqueeze(0).to(device),
        w2_a=w2_a_cpu.unsqueeze(0).to(device),
        w2_b=w2_b_cpu.unsqueeze(0).to(device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rank=rank,
    ), metadata


def apply_bgmv(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    rows = x.shape[0]
    output_size = b.shape[1]
    shrink = torch.zeros((rows, a.shape[1]), dtype=torch.float32, device=x.device)
    output = torch.zeros(
        (rows, output_size), dtype=torch.bfloat16, device=x.device
    )
    torch.ops._C_ascend.bgmv_shrink(x, a, indices, shrink, 1.0)
    return torch.ops._C_ascend.bgmv_expand(
        shrink, b, indices, output, 0, output_size
    )


def dense_forward(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: DenseWeights,
) -> torch.Tensor:
    gate_up = apply_bgmv(x, weights.w13_a, weights.w13_b, indices)
    gate = gate_up[:, : weights.intermediate_size].float()
    up = gate_up[:, weights.intermediate_size :]
    activated = (torch.nn.functional.silu(gate).to(torch.bfloat16) * up).contiguous()
    return apply_bgmv(activated, weights.w2_a, weights.w2_b, indices)


def bind_forward(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: DenseWeights,
) -> Callable[[], torch.Tensor]:
    return lambda: dense_forward(x, indices, weights)


def summarize_breakdown(
    op_statistics: list[dict[str, float | int | str]],
) -> dict[str, float]:
    breakdown = {"shrink": 0.0, "expand": 0.0, "other": 0.0}
    for row in op_statistics:
        op_type = str(row["op_type"])
        total_us = float(row["total_us"])
        if op_type.startswith("bgmv_shrink"):
            category = "shrink"
        elif op_type.startswith("bgmv_expand"):
            category = "expand"
        else:
            category = "other"
        breakdown[category] += total_us
    return breakdown


def median_breakdown(trials: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: statistics.median(trial[name] for trial in trials)
        for name in trials[0]
    }


def render_report(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    case_file: Path,
    trace_root: Path,
) -> str:
    ratios = [result["speedup"] for result in results]
    fingerprints = metadata["fingerprints"]
    lines = [
        "# Qwen3.5-27B Dense 真实权重 BGMV 性能评估",
        "",
        f"- Checkpoint：`{metadata['model_root']}`",
        f"- Model type：`{metadata['model_type']}`（Dense，无 MoE router/expert）。",
        f"- 真实配置：hidden={metadata['hidden_size']}、intermediate={metadata['intermediate_size']}、BF16。",
        f"- 低秩测试 rank：{metadata['rank']}；A/B 为真实 Dense MLP 权重子块，不是训练得到的 LoRA adapter。",
        f"- Gate/Up/Down keys：`{metadata['source_keys']['gate']}` / `{metadata['source_keys']['up']}` / `{metadata['source_keys']['down']}`。",
        f"- 输入 shape={metadata['source_shapes']['x_source']}，fingerprint=`{fingerprints['x_source']}`。",
        f"- W13 A/B shape={metadata['source_shapes']['w13_a']}/{metadata['source_shapes']['w13_b']}，fingerprint=`{fingerprints['w13_a']}`/`{fingerprints['w13_b']}`。",
        f"- W2 A/B shape={metadata['source_shapes']['w2_a']}/{metadata['source_shapes']['w2_b']}，fingerprint=`{fingerprints['w2_a']}`/`{fingerprints['w2_b']}`。",
        "- 自定义路径：int32 BGMV indices；标杆路径：原 int64 BGMV indices。两条路径不含 routing。",
        f"- 独立采集：每条路径 {metadata['trials']} 轮；每轮 repeat=1，奇偶轮交换 int32/int64 顺序，表中为中位数。",
        f"- 可见物理 NPU：`ASCEND_RT_VISIBLE_DEVICES={metadata['visible_devices']}`。",
        f"- 用例：`{case_file}`",
        f"- Trace：`{trace_root}`",
        "- 指标：op_statistic.csv 全部算子的 Total Time(us) 求和 / active(5)；warmup=5、active=5。",
        "",
        "## 性能对比",
        "",
        "| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |",
        "| ---- | ----- | ----- | ------------- | -------- | ------ |",
    ]
    for result in results:
        lines.append(
            f"| {result['case']} | [{result['tokens']}, {metadata['hidden_size']}] "
            f"| bfloat16 | {result['int32_us']:.3f} | "
            f"{result['int64_us']:.3f} | {result['speedup']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 算子级拆分（每步中位数）",
            "",
            "| Case | Shape | Indices | BGMV shrink(us) | BGMV expand(us) | Other(us) |",
            "| ---- | ----- | ------- | --------------- | --------------- | --------- |",
        ]
    )
    for result in results:
        for mode in ("int32", "int64"):
            breakdown = result[f"{mode}_breakdown"]
            lines.append(
                f"| {result['case']} | [{result['tokens']}, {metadata['hidden_size']}] "
                f"| {mode} | {breakdown['shrink']:.3f} | "
                f"{breakdown['expand']:.3f} | {breakdown['other']:.3f} |"
            )
    custom_better = sum(ratio > 1.0 for ratio in ratios)
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
            f"| bfloat16 | {len(results)} | {sum(ratios) / len(ratios):.3f} | {custom_better} | {len(results) - custom_better} |",
            "",
            "## 简短分析",
            "",
            "- 数据值和维度直接来自 Qwen3.5-27B layer-0 Dense MLP checkpoint，输入也取自真实 gate_proj 权重行。",
            "- 该模型没有 expert/router，所以本报告仅回答 int32/int64 BGMV 分支差异，不能验证 MoE routing 融合收益。",
            "- 每个 profiler step 均在 `prof.step()` 前同步 NPU，避免异步 warmup 工作泄漏到 active 统计。",
            "- A/B 是 checkpoint 权重子块而非训练 LoRA adapter，结果是 kernel 性能测试，不是 adapter 模型精度测试。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--model-root", type=Path, default=Path(DEFAULT_MODEL_ROOT))
    parser.add_argument(
        "--case-file",
        type=Path,
        default=script_dir / "qwen35_27b_real_weight_bgmv_perf_cases.jsonl",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=Path("/opt/wqy2/temp/qwen35_27b_real_weight_bgmv_trace"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=script_dir / "qwen35_27b_real_weight_bgmv_profiler_report.md",
    )
    parser.add_argument("--only-tokens", type=int, action="append")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    load_custom_library()
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    cases = load_cases(args.case_file)
    if args.only_tokens:
        selected_tokens = set(args.only_tokens)
        cases = [
            case
            for case in cases
            if int(case_attrs(case)["tokens"]) in selected_tokens
        ]
        if not cases:
            raise ValueError(f"no cases match --only-tokens={sorted(selected_tokens)}")
    ranks = {int(case_attrs(case)["rank"]) for case in cases}
    if len(ranks) != 1:
        raise ValueError(f"all cases must use one rank, got {ranks}")
    max_tokens = max(int(case_attrs(case)["tokens"]) for case in cases)
    weights, metadata = load_real_weights(
        args.model_root, ranks.pop(), max_tokens, device
    )
    metadata["trials"] = args.trials
    metadata["visible_devices"] = os.environ.get(
        "ASCEND_RT_VISIBLE_DEVICES", "not-set"
    )

    results = []
    for case_id, case in enumerate(cases):
        tokens = int(case_attrs(case)["tokens"])
        x = weights.x_source[:tokens]
        indices = {
            "int32": torch.zeros(tokens, dtype=torch.int32, device=device),
            "int64": torch.zeros(tokens, dtype=torch.int64, device=device),
        }
        forwards = {
            mode: bind_forward(x, mode_indices, weights)
            for mode, mode_indices in indices.items()
        }
        int32_output = forwards["int32"]()
        int64_output = forwards["int64"]()
        torch.npu.synchronize()
        if not torch.equal(int32_output.cpu(), int64_output.cpu()):
            max_abs = (int32_output.float() - int64_output.float()).abs().max().item()
            raise AssertionError(
                f"int32/int64 mismatch tokens={tokens} max_abs={max_abs}"
            )

        timings: dict[str, list[float]] = {"int32": [], "int64": []}
        breakdowns: dict[str, list[dict[str, float]]] = {
            "int32": [],
            "int64": [],
        }
        for trial in range(args.trials):
            order = (
                ("int32", "int64") if trial % 2 == 0 else ("int64", "int32")
            )
            for mode in order:
                handler_dir = str(
                    args.trace_root
                    / f"trial_{trial:02d}"
                    / mode
                    / f"tokens_{tokens:04d}"
                )
                elapsed_us = profile_forward(forwards[mode], handler_dir)
                op_stats = read_op_statistics(handler_dir)
                timings[mode].append(elapsed_us)
                breakdowns[mode].append(summarize_breakdown(op_stats))
                print(
                    f"[INFO] case={case_id} tokens={tokens} trial={trial} "
                    f"mode={mode} elapsed={elapsed_us:.3f}us"
                )
        int32_us = statistics.median(timings["int32"])
        int64_us = statistics.median(timings["int64"])
        result = {
            "case": case_id,
            "tokens": tokens,
            "int32_us": int32_us,
            "int64_us": int64_us,
            "speedup": int64_us / int32_us,
            "int32_samples": timings["int32"],
            "int64_samples": timings["int64"],
            "int32_breakdown": median_breakdown(breakdowns["int32"]),
            "int64_breakdown": median_breakdown(breakdowns["int64"]),
        }
        results.append(result)
        print(
            f"[INFO] case={case_id} tokens={tokens} int32={int32_us:.3f}us "
            f"int64={int64_us:.3f}us speedup={result['speedup']:.3f}x"
        )

    args.report.write_text(
        render_report(results, metadata, args.case_file, args.trace_root),
        encoding="utf-8",
    )
    print(f"[INFO] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
