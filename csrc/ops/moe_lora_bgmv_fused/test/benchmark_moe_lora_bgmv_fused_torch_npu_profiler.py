#!/usr/bin/env python3
"""Compare fused BGMV against separate BGMV with real Qwen3.5 weights."""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path
from typing import Any

import torch

from moe_lora_bgmv_fused_profiler_common import (
    DTYPE_MAP,
    build_route,
    compute_metrics,
    forward_fused,
    forward_separate,
    load_cases,
    load_custom_library,
    load_real_weights,
    profile_forward,
    read_op_statistics,
    specs,
    summarize_breakdown,
)


def make_state(
    case: dict[str, Any],
    weights: Any,
    device: torch.device,
    route_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    tensor_cache: dict[tuple[str, torch.dtype], torch.Tensor],
    w2_input_cache: dict[tuple[int, torch.dtype], torch.Tensor],
) -> dict[str, Any]:
    case_specs = specs(case)
    phase = str(case_specs["phase"]["value"])
    tokens = int(case_specs["tokens"]["value"])
    top_k = int(case_specs["top_k"]["value"])
    dtype_name = str(case_specs["x"]["dtype"])
    dtype = DTYPE_MAP[dtype_name]
    if tokens not in route_cache:
        route_cache[tokens] = build_route(weights, tokens, top_k)
    sorted_hidden, sorted_experts = route_cache[tokens]

    def cached_tensor(name: str) -> torch.Tensor:
        key = (name, dtype)
        if key not in tensor_cache:
            tensor_cache[key] = getattr(weights, name).to(dtype).to(device)
        return tensor_cache[key]

    indices = sorted_experts.to(device)
    if phase == "w13":
        x = sorted_hidden.to(dtype).to(device)
        a = cached_tensor("w13_a")
        b = cached_tensor("w13_b")
    elif phase == "w2":
        cache_key = (tokens, dtype)
        if cache_key not in w2_input_cache:
            w13_state = {
                "x": sorted_hidden.to(dtype).to(device),
                "a": cached_tensor("w13_a"),
                "b": cached_tensor("w13_b"),
                "indices": indices,
                "shrink": torch.empty(
                    (tokens * top_k, 16), dtype=torch.float32, device=device
                ),
                "separate_y": torch.zeros(
                    (tokens * top_k, weights.metadata["expert_size"] * 2),
                    dtype=dtype,
                    device=device,
                ),
                "scale": 1.0,
                "output_size": weights.metadata["expert_size"] * 2,
            }
            gate_up = forward_separate(w13_state)
            torch.npu.synchronize()
            gate = gate_up[:, : weights.metadata["expert_size"]].float()
            up = gate_up[:, weights.metadata["expert_size"] :]
            w2_input_cache[cache_key] = (
                torch.nn.functional.silu(gate).to(dtype) * up
            ).contiguous()
        x = w2_input_cache[cache_key]
        a = cached_tensor("w2_a")
        b = cached_tensor("w2_b")
    else:
        raise ValueError(f"unsupported phase: {phase}")

    expected_x_shape = tuple(case_specs["x"]["shape"])
    expected_a_shape = tuple(case_specs["lora_a"]["shape"])
    expected_b_shape = tuple(case_specs["lora_b"]["shape"])
    if tuple(x.shape) != expected_x_shape:
        raise ValueError(f"x shape {tuple(x.shape)} != case {expected_x_shape}")
    if tuple(a.shape) != expected_a_shape or tuple(b.shape) != expected_b_shape:
        raise ValueError("real weight shape does not match performance case")
    output_size = int(b.shape[1])
    rows = int(x.shape[0])
    return {
        "x": x,
        "a": a,
        "b": b,
        "indices": indices,
        "shrink": torch.empty((rows, 16), dtype=torch.float32, device=device),
        "separate_y": torch.zeros((rows, output_size), dtype=dtype, device=device),
        "fused_y": torch.zeros((rows, output_size), dtype=dtype, device=device),
        "scale": float(case_specs["scale"]["value"]),
        "output_size": output_size,
        "phase": phase,
        "tokens": tokens,
        "top_k": top_k,
        "dtype": dtype_name,
    }


def median_dict(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.median(sample[key] for sample in samples)
        for key in samples[0]
    }


def render_report(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    case_file: Path,
    trace_root: Path,
) -> str:
    ratios = [result["speedup"] for result in results]
    lines = [
        "# `moe_lora_bgmv_fused` 真实权重性能评估",
        "",
        f"- Checkpoint：`{metadata['model_root']}`（{metadata['model_type']}）。",
        f"- 配置：hidden={metadata['hidden_size']}、expert intermediate={metadata['expert_size']}、experts={metadata['num_experts']}、top-k={metadata['top_k']}、rank={metadata['rank']}。",
        "- BF16 直接读取 checkpoint；FP16 用例由同一真实权重转换，用于覆盖算子支持 dtype。",
        f"- Router fingerprint：`{metadata['fingerprints']['router']}`；W13 A/B：`{metadata['fingerprints']['w13_a']}`/`{metadata['fingerprints']['w13_b']}`；W2 A/B：`{metadata['fingerprints']['w2_a']}`/`{metadata['fingerprints']['w2_b']}`。",
        "- 无标杆等价 API；标杆为 NPU 上 `bgmv_shrink + bgmv_expand` 小算子拼接。",
        "- 自定义路径为单次 `moe_lora_bgmv_fused`；路由、W2 激活和 buffer 构造均在 profiler 外。",
        f"- 每条路径独立采集 {metadata['trials']} 轮并交换顺序，表中取中位数。",
        "- 固定 schedule：warmup=5、active=5、repeat=1；指标为 op_statistic.csv 全部算子 Total Time(us) 求和 / 5。",
        f"- 用例：`{case_file}`",
        f"- Trace：`{trace_root}`",
        "",
        "## 性能对比",
        "",
        "| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |",
        "| ---- | ----- | ----- | ------------- | -------- | ------ |",
    ]
    for result in results:
        lines.append(
            f"| {result['case']} {result['phase']} | {result['shape']} | "
            f"{result['dtype']} | {result['fused_us']:.3f} | "
            f"{result['separate_us']:.3f} | {result['speedup']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 算子级拆分（每步中位数）",
            "",
            "| Case | Fused(us) | Shrink(us) | Expand(us) | Other fused/separate(us) |",
            "| ---- | --------: | ---------: | ---------: | -----------------------: |",
        ]
    )
    for result in results:
        fused = result["fused_breakdown"]
        separate = result["separate_breakdown"]
        lines.append(
            f"| {result['case']} {result['phase']} | {fused['fused']:.3f} | "
            f"{separate['shrink']:.3f} | {separate['expand']:.3f} | "
            f"{fused['other']:.3f}/{separate['other']:.3f} |"
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
            f"| 平均加速比（>1 表示融合更快） | {sum(ratios) / len(ratios):.3f} |",
            f"| 自定义算子更优 | {custom_better} |",
            f"| 标杆更优 | {len(results) - custom_better} |",
            "",
            "### 按数据类型汇总",
            "",
            "| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |",
            "| ----- | ------ | ---------- | -------------- | -------- |",
        ]
    )
    for dtype in ("bfloat16", "float16"):
        dtype_rows = [result for result in results if result["dtype"] == dtype]
        if not dtype_rows:
            continue
        dtype_ratios = [result["speedup"] for result in dtype_rows]
        better = sum(ratio > 1.0 for ratio in dtype_ratios)
        lines.append(
            f"| {dtype} | {len(dtype_rows)} | "
            f"{sum(dtype_ratios) / len(dtype_ratios):.3f} | "
            f"{better} | {len(dtype_rows) - better} |"
        )
    lines.extend(
        [
            "",
            "## 简短分析",
            "",
            "- 融合路径取消 FP32 rank 中间结果的 GM 往返和一次 kernel launch。",
            "- expert-sorted indices 可优先命中 8-row 权重复用快路径，并保留 4-row fallback；大 token case 用于验证带宽收益是否随分组长度增长。",
            "- W13 与 W2 分开统计，可区分 H=2048 shrink 压力和 O=2048 expand 压力。",
            "- checkpoint 没有对应 LoRA adapter；A/B 是真实 expert 权重的 rank-16 子块，适合评估 kernel 访存形态，但不代表训练 LoRA 精度。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--model-root", type=Path, default=Path("/home/models/Qwen3.5-35B-A3B")
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        default=script_dir / "moe_lora_bgmv_fused_perf_cases.jsonl",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=Path("/opt/wqy2/temp/moe_lora_bgmv_fused_profiler_trace"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=script_dir / "moe_lora_bgmv_fused_torch_npu_profiler_report.md",
    )
    parser.add_argument("--only-case", type=int, action="append")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    print(f"[INFO] custom lib: {load_custom_library()}")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    cases = load_cases(args.case_file)
    selected = set(args.only_case or range(len(cases)))
    if any(case_id < 0 or case_id >= len(cases) for case_id in selected):
        raise ValueError("--only-case is out of range")
    rank_values = {int(specs(case)["rank"]["value"]) for case in cases}
    if len(rank_values) != 1:
        raise ValueError(f"all cases must use one rank: {rank_values}")
    weights = load_real_weights(args.model_root, rank_values.pop())
    metadata = weights.metadata
    metadata["trials"] = args.trials

    route_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tensor_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    w2_input_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
    results = []
    for case_id, case in enumerate(cases):
        if case_id not in selected:
            continue
        state = make_state(
            case,
            weights,
            device,
            route_cache,
            tensor_cache,
            w2_input_cache,
        )
        forward_separate(state)
        forward_fused(state)
        torch.npu.synchronize()
        precision = compute_metrics(state["fused_y"], state["separate_y"])
        threshold = 2**-7 if state["dtype"] == "bfloat16" else 2**-10
        if precision["MERE"] >= threshold or precision["MARE"] >= 10 * threshold:
            raise AssertionError(
                f"case {case_id} precision mismatch: {precision} threshold={threshold}"
            )

        samples = {"fused": [], "separate": []}
        breakdown_samples = {"fused": [], "separate": []}
        forwards = {
            "fused": lambda: forward_fused(state),
            "separate": lambda: forward_separate(state),
        }
        for trial in range(args.trials):
            order = ("fused", "separate") if trial % 2 == 0 else ("separate", "fused")
            for mode in order:
                handler_dir = str(
                    args.trace_root
                    / f"trial_{trial:02d}"
                    / mode
                    / f"case_{case_id:03d}"
                )
                elapsed_us = profile_forward(forwards[mode], handler_dir)
                breakdown = summarize_breakdown(read_op_statistics(handler_dir))
                samples[mode].append(elapsed_us)
                breakdown_samples[mode].append(breakdown)
                print(
                    f"[INFO] case={case_id} {state['phase']} {state['dtype']} "
                    f"trial={trial} mode={mode} elapsed={elapsed_us:.3f}us"
                )
        fused_us = statistics.median(samples["fused"])
        separate_us = statistics.median(samples["separate"])
        result = {
            "case": case_id,
            "phase": state["phase"],
            "shape": list(state["x"].shape),
            "dtype": state["dtype"],
            "fused_us": fused_us,
            "separate_us": separate_us,
            "speedup": separate_us / fused_us,
            "precision": precision,
            "fused_breakdown": median_dict(breakdown_samples["fused"]),
            "separate_breakdown": median_dict(breakdown_samples["separate"]),
        }
        results.append(result)
        print(
            f"[RESULT] case={case_id} fused={fused_us:.3f}us "
            f"separate={separate_us:.3f}us speedup={result['speedup']:.3f}x"
        )

    args.report.write_text(
        render_report(results, metadata, args.case_file, args.trace_root),
        encoding="utf-8",
    )
    print(f"[INFO] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
