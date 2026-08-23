#!/usr/bin/env python3
"""Profile or screen fused BGMV with real Qwen3.5 weight proxies."""

from __future__ import annotations

import argparse
import statistics
from itertools import product
from pathlib import Path
from typing import Any

import torch

from moe_lora_bgmv_fused_profiler_common import (
    DTYPE_MAP,
    INDEX_DTYPE_MAP,
    SUPPORTED_RANKS,
    build_route,
    compute_metrics,
    forward_fused,
    forward_separate,
    load_cases,
    load_custom_library,
    load_real_weights,
    profile_forward,
    read_op_statistics,
    resize_real_tensor,
    screen_forward,
    specs,
    summarize_breakdown,
)

FUSED_BGMV_FAST_MAX_DIM = 4096
FUSED_BGMV_MAX_DIM = 16384
GROUPED_RANK8_MIN_ROWS = 512
GROUPED_RANK32_OR_64_MIN_ROWS = 384


def classify_kernel_route(rank: int, rows: int, hidden: int, output: int) -> str:
    if (
        rank == 16
        and hidden <= FUSED_BGMV_FAST_MAX_DIM
        and output <= FUSED_BGMV_FAST_MAX_DIM
    ):
        return "fast"
    grouped_min_rows = (
        GROUPED_RANK8_MIN_ROWS if rank == 8 else GROUPED_RANK32_OR_64_MIN_ROWS
    )
    if (
        rank in (8, 32, 64)
        and rows >= grouped_min_rows
        and hidden <= FUSED_BGMV_FAST_MAX_DIM
        and output <= FUSED_BGMV_FAST_MAX_DIM
    ):
        return "grouped"
    if (
        rank in SUPPORTED_RANKS
        and hidden <= FUSED_BGMV_MAX_DIM
        and output <= FUSED_BGMV_MAX_DIM
    ):
        return "generic"
    return "split"


def make_state(
    case: dict[str, Any],
    weights: Any,
    device: torch.device,
    route_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]],
    tensor_cache: dict[
        tuple[str, torch.dtype, tuple[int, ...]], torch.Tensor
    ],
    w2_input_cache: dict[
        tuple[int, int, int, int, torch.dtype, torch.dtype], torch.Tensor
    ],
) -> dict[str, Any]:
    case_specs = specs(case)
    phase = str(case_specs["phase"]["value"])
    tokens = int(case_specs["tokens"]["value"])
    top_k = int(case_specs["top_k"]["value"])
    rank = int(case_specs["rank"]["value"])
    dtype_name = str(case_specs["x"]["dtype"])
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"unsupported data dtype: {dtype_name}")
    dtype = DTYPE_MAP[dtype_name]
    index_dtype_name = str(case_specs["indices"]["dtype"])
    if index_dtype_name not in INDEX_DTYPE_MAP:
        raise ValueError(f"unsupported index dtype: {index_dtype_name}")
    index_dtype = INDEX_DTYPE_MAP[index_dtype_name]
    expected_x_shape = tuple(case_specs["x"]["shape"])
    expected_a_shape = tuple(case_specs["lora_a"]["shape"])
    expected_b_shape = tuple(case_specs["lora_b"]["shape"])
    expected_indices_shape = tuple(case_specs["indices"]["shape"])
    expected_y_shape = tuple(case_specs["y"]["shape"])
    if rank not in SUPPORTED_RANKS:
        raise ValueError(f"unsupported rank: {rank}")
    if any(
        str(case_specs[name]["dtype"]) != dtype_name
        for name in ("lora_a", "lora_b", "y")
    ):
        raise ValueError("x, lora_a, lora_b and y dtypes must match")
    if (
        len(expected_x_shape) != 2
        or len(expected_b_shape) != 3
        or expected_a_shape != (
            weights.metadata["num_experts"], rank, expected_x_shape[1]
        )
        or expected_b_shape[0] != weights.metadata["num_experts"]
        or expected_b_shape[2] != rank
        or expected_indices_shape != (expected_x_shape[0],)
        or expected_y_shape != (expected_x_shape[0], expected_b_shape[1])
    ):
        raise ValueError("inconsistent parameterized performance case shapes")
    if not (
        1 <= expected_x_shape[1] <= FUSED_BGMV_MAX_DIM
        and 1 <= expected_b_shape[1] <= FUSED_BGMV_MAX_DIM
    ):
        raise ValueError("H and O must both be in [1, 16384]")
    rows = int(expected_x_shape[0])
    if rows <= 0:
        raise ValueError("M must be positive for performance screening")
    route_key = (tokens, top_k, rows)
    if route_key not in route_cache:
        route_cache[route_key] = build_route(
            weights, tokens, top_k, routed_rows=rows
        )
    route_hidden, route_experts = route_cache[route_key]
    if route_hidden.shape[0] != rows:
        raise ValueError("route builder returned an unexpected row count")

    def cached_tensor(name: str, target_shape: tuple[int, ...]) -> torch.Tensor:
        key = (name, dtype, target_shape)
        if key not in tensor_cache:
            source = getattr(weights, name)
            tensor_cache[key] = resize_real_tensor(
                source, target_shape
            ).to(dtype).to(device)
        return tensor_cache[key]

    indices = route_experts[:rows].to(dtype=index_dtype, device=device)
    if phase == "w13":
        x_source = route_hidden
        a = cached_tensor("w13_a", expected_a_shape)
        b = cached_tensor("w13_b", expected_b_shape)
    elif phase == "w2":
        cache_key = (tokens, top_k, rows, rank, dtype, index_dtype)
        if cache_key not in w2_input_cache:
            full_indices = route_experts.to(dtype=index_dtype, device=device)
            full_rows = int(route_hidden.shape[0])
            w13_state = {
                "x": route_hidden.to(dtype=dtype, device=device),
                "a": cached_tensor(
                    "w13_a",
                    (
                        weights.metadata["num_experts"],
                        rank,
                        weights.metadata["hidden_size"],
                    ),
                ),
                "b": cached_tensor(
                    "w13_b",
                    (
                        weights.metadata["num_experts"],
                        2 * weights.metadata["expert_size"],
                        rank,
                    ),
                ),
                "indices": full_indices,
                "shrink": torch.empty(
                    (full_rows, rank), dtype=torch.float32, device=device
                ),
                "separate_y": torch.zeros(
                    (full_rows, weights.metadata["expert_size"] * 2),
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
        x_source = w2_input_cache[cache_key]
        a = cached_tensor("w2_a", expected_a_shape)
        b = cached_tensor("w2_b", expected_b_shape)
    else:
        raise ValueError(f"unsupported phase: {phase}")

    x_source = x_source[:rows].contiguous()
    x = resize_real_tensor(x_source, expected_x_shape).to(
        dtype=dtype, device=device
    )
    if tuple(x.shape) != expected_x_shape:
        raise ValueError(f"x shape {tuple(x.shape)} != case {expected_x_shape}")
    if tuple(a.shape) != expected_a_shape or tuple(b.shape) != expected_b_shape:
        raise ValueError("real weight shape does not match performance case")
    output_size = int(b.shape[1])
    kernel_route = classify_kernel_route(
        rank,
        rows,
        int(x.shape[1]),
        output_size,
    )
    return {
        "x": x,
        "a": a,
        "b": b,
        "indices": indices,
        "shrink": torch.empty((rows, rank), dtype=torch.float32, device=device),
        "separate_y": torch.zeros((rows, output_size), dtype=dtype, device=device),
        "fused_y": torch.zeros((rows, output_size), dtype=dtype, device=device),
        "scale": float(case_specs["scale"]["value"]),
        "output_size": output_size,
        "phase": phase,
        "tokens": tokens,
        "top_k": top_k,
        "rank": rank,
        "dtype": dtype_name,
        "index_dtype": index_dtype_name,
        "kernel_route": kernel_route,
        "derived_from_checkpoint": (
            expected_x_shape[1] != route_hidden.shape[1]
            or expected_a_shape != tuple(getattr(weights, f"{phase}_a").shape)
            or expected_b_shape != tuple(getattr(weights, f"{phase}_b").shape)
        ),
        "production_fused": kernel_route != "split",
    }


def build_parameterized_cases(
    ranks: list[int],
    rows_values: list[int],
    hidden_values: list[int],
    output_values: list[int],
    dtype_names: list[str],
    index_dtype_names: list[str],
    phases: list[str],
    top_k: int,
    num_experts: int,
) -> list[dict[str, Any]]:
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}]")
    if any(rank not in SUPPORTED_RANKS for rank in ranks):
        raise ValueError(f"rank must be one of {SUPPORTED_RANKS}")
    if any(rows <= 0 for rows in rows_values):
        raise ValueError("M must be positive for performance screening")
    if any(not 1 <= size <= FUSED_BGMV_MAX_DIM for size in hidden_values):
        raise ValueError("H must be in [1, 16384]")
    if any(not 1 <= size <= FUSED_BGMV_MAX_DIM for size in output_values):
        raise ValueError("O must be in [1, 16384]")

    cases = []
    combinations = product(
        ranks,
        rows_values,
        hidden_values,
        output_values,
        dtype_names,
        index_dtype_names,
        phases,
    )
    for rank, rows, hidden, output, dtype, index_dtype, phase in combinations:
        tokens = (rows + top_k - 1) // top_k
        cases.append(
            {
                "inputs": [
                    {
                        "name": "x",
                        "type": "tensor",
                        "required": True,
                        "dtype": dtype,
                        "shape": [rows, hidden],
                    },
                    {
                        "name": "lora_a",
                        "type": "tensor",
                        "required": True,
                        "dtype": dtype,
                        "shape": [num_experts, rank, hidden],
                    },
                    {
                        "name": "lora_b",
                        "type": "tensor",
                        "required": True,
                        "dtype": dtype,
                        "shape": [num_experts, output, rank],
                    },
                    {
                        "name": "indices",
                        "type": "tensor",
                        "required": True,
                        "dtype": index_dtype,
                        "shape": [rows],
                        "range": [0, num_experts],
                    },
                    {
                        "name": "y",
                        "type": "tensor",
                        "required": True,
                        "dtype": dtype,
                        "shape": [rows, output],
                    },
                    {
                        "name": "phase",
                        "type": "attr",
                        "required": True,
                        "dtype": "str",
                        "value": phase,
                    },
                    {
                        "name": "tokens",
                        "type": "attr",
                        "required": True,
                        "dtype": "int",
                        "value": tokens,
                    },
                    {
                        "name": "top_k",
                        "type": "attr",
                        "required": True,
                        "dtype": "int",
                        "value": top_k,
                    },
                    {
                        "name": "rank",
                        "type": "attr",
                        "required": True,
                        "dtype": "int",
                        "value": rank,
                    },
                    {
                        "name": "scale",
                        "type": "attr",
                        "required": True,
                        "dtype": "float",
                        "value": 1.0,
                    },
                ]
            }
        )
    return cases


def median_dict(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.median(sample[key] for sample in samples)
        for key in samples[0]
    }


def render_report(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    case_source: str,
    trace_root: Path,
    quick_screen: bool,
) -> str:
    if not results:
        raise ValueError("cannot render an empty performance report")
    ratios = [result["speedup"] for result in results]
    routed_ratios = [
        result["speedup"] if result["production_fused"] else 1.0
        for result in results
    ]
    fallback_results = [
        result for result in results if not result["production_fused"]
    ]
    title_suffix = "同步快速筛选（非 profiler）" if quick_screen else "性能评估"
    lines = [
        "# 性能评估结果",
        "",
        f"- 算子：`moe_lora_bgmv_fused`；模式：Qwen 权重 proxy {title_suffix}。",
        f"- Checkpoint：`{metadata['model_root']}`（{metadata['model_type']}）。",
        f"- Checkpoint 配置：hidden={metadata['hidden_size']}、"
        f"expert intermediate={metadata['expert_size']}、"
        f"experts={metadata['num_experts']}、原始 top-k={metadata['top_k']}；"
        f"case ranks={metadata['case_ranks']}，"
        f"加载最大 rank 子块={metadata['rank']}。",
        "- **Proxy 声明**：A/B 来自 Qwen expert 权重子块，并按目标 H/O/rank "
        "裁剪或周期重复；这是真实 checkpoint 数值驱动的 kernel proxy，不是 "
        "Qwen/DeepSeek 的真实 LoRA adapter，也不是 DeepSeek checkpoint 实测。",
        "- BF16 直接读取 checkpoint；FP16 用例由同一真实权重转换，用于覆盖算子支持 dtype。",
        f"- Router fingerprint：`{metadata['fingerprints']['router']}`；"
        f"W13 A/B：`{metadata['fingerprints']['w13_a']}`/"
        f"`{metadata['fingerprints']['w13_b']}`；"
        f"W2 A/B：`{metadata['fingerprints']['w2_a']}`/"
        f"`{metadata['fingerprints']['w2_b']}`。",
        "- 无标杆等价 API；标杆为 NPU 上 `bgmv_shrink + bgmv_expand` 小算子拼接。",
        "- 自定义路径为单次 `moe_lora_bgmv_fused`；路由、W2 激活和 buffer 构造均在计时区外。",
        f"- 每条路径独立采集 {metadata['trials']} 轮并交换顺序，表中取中位数。",
        f"- 用例来源：`{case_source}`",
    ]
    if quick_screen:
        lines.append(
            f"- 同步筛选口径：warmup={metadata['screen_warmup']}、"
            f"iterations={metadata['screen_iterations']}，使用 host wall time / "
            "iterations；只用于快速找 crossover，正式结论必须复跑 "
            "torch_npu.profiler。"
        )
    else:
        lines.extend(
            [
                "- 固定 profiler schedule：warmup=5、active=5、repeat=1；指标为 op_statistic.csv 全部算子 Total Time(us) 求和 / 5。",
                f"- Trace：`{trace_root}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 性能对比",
            "",
            "| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |",
            "| ---- | ----- | ----- | ------------- | -------- | ------ |",
        ]
    )
    for result in results:
        lines.append(
            f"| {result['case']} {result['phase']} | {result['shape']} | "
            f"{result['dtype_label']} | {result['fused_us']:.3f} | "
            f"{result['separate_us']:.3f} | {result['speedup']:.3f} |"
        )
    if not quick_screen:
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
    dtype_labels = sorted({result["dtype_label"] for result in results})
    for dtype_label in dtype_labels:
        dtype_rows = [
            result for result in results if result["dtype_label"] == dtype_label
        ]
        if not dtype_rows:
            continue
        dtype_ratios = [result["speedup"] for result in dtype_rows]
        better = sum(ratio > 1.0 for ratio in dtype_ratios)
        lines.append(
            f"| {dtype_label} | {len(dtype_rows)} | "
            f"{sum(dtype_ratios) / len(dtype_ratios):.3f} | "
            f"{better} | {len(dtype_rows) - better} |"
        )
    fallback_cases = ", ".join(
        f"{result['case']} ({result['shape']})" for result in fallback_results
    ) or "无"
    route_counts = {
        route: sum(result["kernel_route"] == route for result in results)
        for route in ("fast", "grouped", "generic", "split")
    }
    lines.extend(
        [
            "",
            "## 生产路由汇总",
            "",
            "直接 kernel 对比表按四级路由契约标记 fast、grouped、generic 与 split 兜底。",
            "",
            "| 指标 | 值 |",
            "| ---- | -- |",
            f"| 选择 fast kernel | {route_counts['fast']} |",
            f"| 选择 grouped kernel | {route_counts['grouped']} |",
            f"| 选择 generic kernel | {route_counts['generic']} |",
            f"| 选择 split BGMV 兜底 | {route_counts['split']} |",
            f"| 路由后平均有效加速比 | {sum(routed_ratios) / len(routed_ratios):.3f} |",
            f"| 兜底 case | {fallback_cases} |",
        ]
    )
    lines.extend(
        [
            "",
            "## 简短分析",
            "",
            "- 融合路径取消 FP32 rank 中间结果的 GM 往返和一次 kernel launch。",
            "- rank16 且 H/O<=4096 使用 fast；rank8 在 M>=512、rank32/64 在 M>=384 且 H/O<=4096 使用 grouped；其余受支持 shape 使用 group1 generic。",
            "- 数据和索引 dtype 单独参数化，表中 DType 使用 `data/index` 形式，便于观察 int32/int64 搬运差异。",
            "- 结果只代表 Qwen expert 权重 proxy 下 fused 对 split 的 kernel crossover；不能外推为 DeepSeek 或真实 LoRA adapter 的端到端收益。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile or synchronously screen fused MoE LoRA BGMV against "
            "split BGMV using Qwen checkpoint weight proxies."
        )
    )
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
        default=None,
    )
    parser.add_argument("--only-case", type=int, action="append")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--quick-screen",
        action="store_true",
        help="use synchronized wall time instead of torch_npu.profiler",
    )
    parser.add_argument(
        "--rank", type=int, nargs="+", choices=SUPPORTED_RANKS
    )
    parser.add_argument("--m", type=int, nargs="+", help="routed row counts")
    parser.add_argument("--h", type=int, nargs="+", help="input dimensions")
    parser.add_argument("--o", type=int, nargs="+", help="output dimensions")
    parser.add_argument(
        "--dtype", nargs="+", choices=tuple(DTYPE_MAP), help="data dtypes"
    )
    parser.add_argument(
        "--index-dtype",
        nargs="+",
        choices=tuple(INDEX_DTYPE_MAP),
        help="index dtypes",
    )
    parser.add_argument("--phase", nargs="+", choices=("w13", "w2"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--screen-warmup", type=int, default=5)
    parser.add_argument("--screen-iterations", type=int, default=20)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.screen_warmup < 0 or args.screen_iterations <= 0:
        raise ValueError("screen warmup must be non-negative and iterations positive")

    shape_values = (args.m, args.h, args.o)
    parameterized = all(value is not None for value in shape_values)
    if any(value is not None for value in shape_values) and not parameterized:
        raise ValueError("--m, --h and --o must be specified together")

    if parameterized:
        rank_values = args.rank or [16]
        weights = load_real_weights(args.model_root, max(rank_values))
        cases = build_parameterized_cases(
            ranks=rank_values,
            rows_values=args.m,
            hidden_values=args.h,
            output_values=args.o,
            dtype_names=args.dtype or ["bfloat16"],
            index_dtype_names=args.index_dtype or ["int32"],
            phases=args.phase or ["w13"],
            top_k=args.top_k,
            num_experts=weights.metadata["num_experts"],
        )
        case_source = (
            "CLI Cartesian grid "
            f"rank={rank_values}, M={args.m}, H={args.h}, O={args.o}, "
            f"dtype={args.dtype or ['bfloat16']}, "
            f"index_dtype={args.index_dtype or ['int32']}, "
            f"phase={args.phase or ['w13']}, top_k={args.top_k}"
        )
    else:
        cases = load_cases(args.case_file)
        filtered_cases = []
        for case in cases:
            case_specs = specs(case)
            if args.rank and int(case_specs["rank"]["value"]) not in args.rank:
                continue
            if args.dtype and str(case_specs["x"]["dtype"]) not in args.dtype:
                continue
            if (
                args.index_dtype
                and str(case_specs["indices"]["dtype"]) not in args.index_dtype
            ):
                continue
            if args.phase and str(case_specs["phase"]["value"]) not in args.phase:
                continue
            filtered_cases.append(case)
        cases = filtered_cases
        if not cases:
            raise ValueError(
                "case filters selected nothing; provide --m/--h/--o to generate new shapes"
            )
        rank_values = sorted(
            {int(specs(case)["rank"]["value"]) for case in cases}
        )
        weights = load_real_weights(args.model_root, max(rank_values))
        case_source = str(args.case_file)

    selected = set(args.only_case or range(len(cases)))
    if any(case_id < 0 or case_id >= len(cases) for case_id in selected):
        raise ValueError("--only-case is out of range")
    selected_rank_values = sorted(
        {
            int(specs(case)["rank"]["value"])
            for case_id, case in enumerate(cases)
            if case_id in selected
        }
    )
    report = args.report
    if report is None:
        report = (
            Path("/opt/wqy2/temp/moe_lora_bgmv_fused_screen_report.md")
            if args.quick_screen
            else script_dir / "moe_lora_bgmv_fused_torch_npu_profiler_report.md"
        )

    print(f"[INFO] custom lib: {load_custom_library()}")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    metadata = dict(weights.metadata)
    metadata["trials"] = args.trials
    metadata["case_ranks"] = selected_rank_values
    metadata["screen_warmup"] = args.screen_warmup
    metadata["screen_iterations"] = args.screen_iterations

    route_cache: dict[
        tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    tensor_cache: dict[
        tuple[str, torch.dtype, tuple[int, ...]], torch.Tensor
    ] = {}
    w2_input_cache: dict[
        tuple[int, int, int, int, torch.dtype, torch.dtype], torch.Tensor
    ] = {}
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
        state["separate_y"].zero_()
        state["fused_y"].zero_()
        forward_separate(state)
        forward_fused(state)
        torch.npu.synchronize()
        precision = compute_metrics(state["fused_y"], state["separate_y"])
        if state["dtype"] == "bfloat16":
            max_abs_limit = 2**-10
            relative_l2_limit = 2**-6
            cosine_limit = 0.999
        else:
            max_abs_limit = 2**-13
            relative_l2_limit = 2**-9
            cosine_limit = 0.9999
        reference_near_zero = precision["expected_abs_max"] < max_abs_limit
        relative_mismatch = not reference_near_zero and (
            precision["relative_l2"] >= relative_l2_limit
            or precision["cosine_sim"] <= cosine_limit
        )
        if precision["max_abs_err"] >= max_abs_limit or relative_mismatch:
            raise AssertionError(
                f"case {case_id} precision mismatch: {precision}; "
                f"limits=max_abs<{max_abs_limit}, "
                f"relative_l2<{relative_l2_limit}, cosine>{cosine_limit}, "
                f"reference_near_zero={reference_near_zero}"
            )

        samples = {"fused": [], "separate": []}
        breakdown_samples = {"fused": [], "separate": []}
        forwards = {
            "fused": lambda: forward_fused(state),
            "separate": lambda: forward_separate(state),
        }
        reset_outputs = {
            "fused": state["fused_y"].zero_,
            "separate": state["separate_y"].zero_,
        }
        for trial in range(args.trials):
            order = ("fused", "separate") if trial % 2 == 0 else ("separate", "fused")
            for mode in order:
                if args.quick_screen:
                    elapsed_us = screen_forward(
                        forwards[mode],
                        reset_outputs[mode],
                        args.screen_warmup,
                        args.screen_iterations,
                    )
                else:
                    reset_outputs[mode]()
                    torch.npu.synchronize()
                    handler_dir = str(
                        args.trace_root
                        / f"trial_{trial:02d}"
                        / mode
                        / f"case_{case_id:03d}"
                    )
                    elapsed_us = profile_forward(forwards[mode], handler_dir)
                    breakdown = summarize_breakdown(
                        read_op_statistics(handler_dir)
                    )
                    breakdown_samples[mode].append(breakdown)
                samples[mode].append(elapsed_us)
                print(
                    f"[INFO] case={case_id} {state['phase']} R={state['rank']} "
                    f"{state['dtype']}/{state['index_dtype']} "
                    f"trial={trial} mode={mode} elapsed={elapsed_us:.3f}us"
                )
        fused_us = statistics.median(samples["fused"])
        separate_us = statistics.median(samples["separate"])
        result = {
            "case": case_id,
            "phase": state["phase"],
            "shape": (
                f"R{state['rank']} {list(state['x'].shape)} -> {state['output_size']}"
            ),
            "dtype": state["dtype"],
            "index_dtype": state["index_dtype"],
            "dtype_label": f"{state['dtype']}/{state['index_dtype']}",
            "kernel_route": state["kernel_route"],
            "fused_us": fused_us,
            "separate_us": separate_us,
            "speedup": separate_us / fused_us,
            "precision": precision,
            "derived_from_checkpoint": state["derived_from_checkpoint"],
            "production_fused": state["production_fused"],
        }
        if not args.quick_screen:
            result["fused_breakdown"] = median_dict(
                breakdown_samples["fused"]
            )
            result["separate_breakdown"] = median_dict(
                breakdown_samples["separate"]
            )
        results.append(result)
        print(
            f"[RESULT] case={case_id} fused={fused_us:.3f}us "
            f"separate={separate_us:.3f}us speedup={result['speedup']:.3f}x"
        )

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_report(
            results,
            metadata,
            case_source,
            args.trace_root,
            args.quick_screen,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
