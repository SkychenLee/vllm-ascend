#!/usr/bin/env python3
"""Profile fused MoE-LoRA metadata with real Qwen3.5 MoE checkpoint data."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch_npu
from safetensors import safe_open

from moe_lora_build_combined_idx_profiler_common import (
    profile_forward,
    read_op_statistics,
)


DEFAULT_MODEL_ROOT = "/home/models/Qwen3.5-35B-A3B"
EXPECTED_MODEL_TYPE = "qwen3_5_moe_text"
ROUTER_KEY = "model.language_model.layers.0.mlp.gate.weight"
GATE_UP_KEY = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DOWN_KEY = "model.language_model.layers.0.mlp.experts.down_proj"


@dataclass
class RealWeights:
    router_cpu: torch.Tensor
    w13_a: torch.Tensor
    w13_b: torch.Tensor
    w2_a: torch.Tensor
    w2_b: torch.Tensor
    fingerprints: dict[str, str]
    hidden_size: int
    expert_size: int
    num_experts: int
    rank: int


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
    with path.open(encoding="utf-8") as case_file:
        cases = [json.loads(line) for line in case_file if line.strip()]
    if len(cases) < 8:
        raise ValueError("real-weight profiler requires at least 8 JSONL cases")
    return cases


def case_attrs(case: dict[str, Any]) -> dict[str, Any]:
    return {item["name"]: item["value"] for item in case["inputs"]}


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _resolve_shard(model_root: Path, weight_map: dict[str, str], key: str) -> Path:
    if key not in weight_map:
        raise KeyError(f"checkpoint does not contain {key}")
    return model_root / weight_map[key]


def load_real_weights(
    model_root: Path,
    rank: int,
    device: torch.device,
) -> tuple[RealWeights, dict[str, Any]]:
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    if text_config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"{model_root} is {text_config.get('model_type')}, expected {EXPECTED_MODEL_TYPE}"
        )
    hidden_size = int(text_config["hidden_size"])
    expert_size = int(text_config["moe_intermediate_size"])
    num_experts = int(text_config["num_experts"])
    top_k = int(text_config["num_experts_per_tok"])
    if rank <= 0 or rank > min(hidden_size, expert_size):
        raise ValueError(f"invalid rank {rank}")

    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    router_shard = _resolve_shard(model_root, weight_map, ROUTER_KEY)
    gate_up_shard = _resolve_shard(model_root, weight_map, GATE_UP_KEY)
    down_shard = _resolve_shard(model_root, weight_map, DOWN_KEY)

    with safe_open(router_shard, framework="pt", device="cpu") as checkpoint:
        router = checkpoint.get_tensor(ROUTER_KEY).contiguous()
    with safe_open(gate_up_shard, framework="pt", device="cpu") as checkpoint:
        gate_up = checkpoint.get_slice(GATE_UP_KEY)
        w13_a_cpu = gate_up[:, :rank, :].contiguous()
        w13_b_cpu = gate_up[:, :, :rank].contiguous()
    with safe_open(down_shard, framework="pt", device="cpu") as checkpoint:
        down = checkpoint.get_slice(DOWN_KEY)
        w2_a_cpu = down[:, :rank, :].contiguous()
        w2_b_cpu = down[:, :, :rank].contiguous()

    expected_shapes = {
        "router": (num_experts, hidden_size),
        "w13_a": (num_experts, rank, hidden_size),
        "w13_b": (num_experts, 2 * expert_size, rank),
        "w2_a": (num_experts, rank, expert_size),
        "w2_b": (num_experts, hidden_size, rank),
    }
    tensors = {
        "router": router,
        "w13_a": w13_a_cpu,
        "w13_b": w13_b_cpu,
        "w2_a": w2_a_cpu,
        "w2_b": w2_b_cpu,
    }
    for name, expected_shape in expected_shapes.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"unexpected {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )
    fingerprints = {
        name: tensor_fingerprint(tensor) for name, tensor in tensors.items()
    }
    metadata = {
        "model_root": str(model_root),
        "model_type": text_config["model_type"],
        "hidden_size": hidden_size,
        "expert_size": expert_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "rank": rank,
        "source_keys": {
            "router": ROUTER_KEY,
            "w13": GATE_UP_KEY,
            "w2": DOWN_KEY,
        },
        "source_shards": {
            "router": router_shard.name,
            "w13": gate_up_shard.name,
            "w2": down_shard.name,
        },
        "source_shapes": {
            name: list(tensor.shape) for name, tensor in tensors.items()
        },
        "fingerprints": fingerprints,
    }
    return RealWeights(
        router_cpu=router,
        w13_a=w13_a_cpu.to(device),
        w13_b=w13_b_cpu.to(device),
        w2_a=w2_a_cpu.to(device),
        w2_b=w2_b_cpu.to(device),
        fingerprints=fingerprints,
        hidden_size=hidden_size,
        expert_size=expert_size,
        num_experts=num_experts,
        rank=rank,
    ), metadata


def build_inputs(
    tokens: int,
    top_k: int,
    weights: RealWeights,
    device: torch.device,
) -> dict[str, Any]:
    repeats = (tokens + weights.num_experts - 1) // weights.num_experts
    hidden_cpu = weights.router_cpu.repeat((repeats, 1))[:tokens].contiguous()
    logits = hidden_cpu.float() @ weights.router_cpu.float().T
    topk_ids_cpu = torch.topk(logits, top_k, dim=1).indices.to(torch.int32)
    flat_experts = topk_ids_cpu.reshape(-1)
    inverse_permutation = torch.argsort(flat_experts, stable=True)
    expanded_cpu = torch.empty(flat_experts.numel(), dtype=torch.int32)
    expanded_cpu[inverse_permutation] = torch.arange(
        flat_experts.numel(), dtype=torch.int32
    )
    sorted_hidden_cpu = hidden_cpu.repeat_interleave(top_k, dim=0)[
        inverse_permutation
    ].contiguous()
    token_lora_cpu = torch.zeros(tokens, dtype=torch.int64)
    adapter_enabled_cpu = torch.ones(1, dtype=torch.bool)
    return {
        "expanded": expanded_cpu.to(device),
        "topk_ids": topk_ids_cpu.to(device),
        "token_lora": token_lora_cpu.to(device),
        "adapter_enabled": adapter_enabled_cpu.to(device),
        "x": sorted_hidden_cpu.to(device),
        "top_k": top_k,
        "num_experts": weights.num_experts,
        "tokens": tokens,
    }


def build_combined_idx_baseline(
    expert_per_row: torch.Tensor,
    lora_per_row: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    safe_lora = lora_per_row.clamp(min=0)
    enabled = (lora_per_row >= 0) & adapter_enabled[safe_lora].bool()
    return torch.where(
        enabled,
        safe_lora * num_experts + expert_per_row,
        torch.full_like(lora_per_row, -1),
    ).contiguous()


def recover_baseline(inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_permutation = torch.argsort(torch.abs(inputs["expanded"]))
    expert_per_row = inputs["topk_ids"].reshape(-1)[inverse_permutation].long()
    original_token = inverse_permutation // inputs["top_k"]
    lora_per_row = inputs["token_lora"][original_token]
    return expert_per_row, lora_per_row


def apply_bgmv(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    rows = x.shape[0]
    rank = a.shape[1]
    output_size = b.shape[1]
    shrink = torch.zeros((rows, rank), dtype=torch.float32, device=x.device)
    output = torch.zeros(
        (rows, output_size), dtype=torch.bfloat16, device=x.device
    )
    torch.ops._C_ascend.bgmv_shrink(x, a, indices, shrink, 1.0)
    return torch.ops._C_ascend.bgmv_expand(
        shrink, b, indices, output, 0, output_size
    )


def apply_w13_w2(
    x: torch.Tensor,
    combined_idx_w13: torch.Tensor,
    combined_idx_w2: torch.Tensor,
    weights: RealWeights,
) -> torch.Tensor:
    gate_up = apply_bgmv(x, weights.w13_a, weights.w13_b, combined_idx_w13)
    gate = gate_up[:, : weights.expert_size].float()
    up = gate_up[:, weights.expert_size :]
    activated = (torch.nn.functional.silu(gate).to(torch.bfloat16) * up).contiguous()
    return apply_bgmv(activated, weights.w2_a, weights.w2_b, combined_idx_w2)


def custom_forward(inputs: dict[str, Any], weights: RealWeights) -> torch.Tensor:
    combined_idx = torch.ops._C_ascend.moe_lora_build_combined_idx(
        inputs["expanded"],
        inputs["topk_ids"],
        inputs["token_lora"],
        inputs["adapter_enabled"],
        inputs["num_experts"],
    )
    return apply_w13_w2(inputs["x"], combined_idx, combined_idx, weights)


def baseline_forward(inputs: dict[str, Any], weights: RealWeights) -> torch.Tensor:
    expert_per_row, lora_per_row = recover_baseline(inputs)
    combined_idx_w13 = build_combined_idx_baseline(
        expert_per_row,
        lora_per_row,
        inputs["adapter_enabled"],
        inputs["num_experts"],
    )
    combined_idx_w2 = build_combined_idx_baseline(
        expert_per_row,
        lora_per_row,
        inputs["adapter_enabled"],
        inputs["num_experts"],
    )
    return apply_w13_w2(
        inputs["x"], combined_idx_w13, combined_idx_w2, weights
    )


def bind_forward(
    forward: Callable[[dict[str, Any], RealWeights], torch.Tensor],
    inputs: dict[str, Any],
    weights: RealWeights,
) -> Callable[[], torch.Tensor]:
    return lambda: forward(inputs, weights)


def summarize_breakdown(
    op_statistics: list[dict[str, float | int | str]],
) -> dict[str, float]:
    breakdown = {
        "routing": 0.0,
        "sort": 0.0,
        "shrink": 0.0,
        "expand": 0.0,
        "other": 0.0,
    }
    for row in op_statistics:
        op_type = str(row["op_type"])
        total_us = float(row["total_us"])
        if op_type == "moe_lora_build_combined_idx":
            category = "routing"
        elif op_type == "Sort":
            category = "sort"
        elif op_type.startswith("bgmv_shrink"):
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
        "# Qwen3.5 MoE 真实权重性能评估",
        "",
        f"- Checkpoint：`{metadata['model_root']}`",
        f"- Model type：`{metadata['model_type']}`",
        f"- 真实配置：hidden={metadata['hidden_size']}、expert intermediate={metadata['expert_size']}、experts={metadata['num_experts']}、top-k={metadata['top_k']}、BF16。",
        f"- 低秩测试 rank：{metadata['rank']}。A/B 来自 layer-0 真实 expert 权重的 rank 子块，不是训练得到的 LoRA adapter。",
        f"- Router key：`{metadata['source_keys']['router']}`，shape={metadata['source_shapes']['router']}，fingerprint=`{fingerprints['router']}`。",
        f"- W13 key：`{metadata['source_keys']['w13']}`，A/B shape={metadata['source_shapes']['w13_a']}/{metadata['source_shapes']['w13_b']}，fingerprint=`{fingerprints['w13_a']}`/`{fingerprints['w13_b']}`。",
        f"- W2 key：`{metadata['source_keys']['w2']}`，A/B shape={metadata['source_shapes']['w2_a']}/{metadata['source_shapes']['w2_b']}，fingerprint=`{fingerprints['w2_a']}`/`{fingerprints['w2_b']}`。",
        "- 标杆：AiCPU argsort 路由恢复 + W13/W2 各自生成 int64 combined_idx + 原 int64 BGMV。",
        "- 自定义路径：一次 AscendC fused routing + W13/W2 复用 int32 combined_idx + int32 BGMV。",
        f"- 独立采集：每条路径 {metadata['trials']} 轮；每轮 repeat=1，奇偶轮交换 custom/baseline 顺序，表中为中位数。",
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
            f"| {result['case']} | [{result['tokens']}, {metadata['top_k']}] "
            f"| bfloat16 | {result['custom_us']:.3f} | "
            f"{result['baseline_us']:.3f} | {result['speedup']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 算子级拆分（每步中位数）",
            "",
            "| Case | Shape | Path | Routing(us) | Sort(us) | BGMV shrink(us) | BGMV expand(us) | Other(us) |",
            "| ---- | ----- | ---- | ----------- | -------- | --------------- | --------------- | --------- |",
        ]
    )
    for result in results:
        for mode in ("custom", "baseline"):
            breakdown = result[f"{mode}_breakdown"]
            lines.append(
                f"| {result['case']} | [{result['tokens']}, {metadata['top_k']}] "
                f"| {mode} | {breakdown['routing']:.3f} | {breakdown['sort']:.3f} "
                f"| {breakdown['shrink']:.3f} | {breakdown['expand']:.3f} "
                f"| {breakdown['other']:.3f} |"
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
            "- 数据直接读取自 Qwen3.5-35B-A3B checkpoint；tensor key、shape、dtype 和内容指纹均在报告头记录。",
            "- Router top-k 由真实 router 权重与 checkpoint 权重行构造的确定性 hidden 输入计算，不是随机 top-k，也不是线上抓取的真实激活。",
            "- 每个 profiler step 均在 `prof.step()` 前同步 NPU，避免异步 warmup 工作泄漏到 active 统计。",
            "- 由于本机没有匹配的 LoRA adapter，rank-16 A/B 是 checkpoint expert 权重子块；结果用于评估 kernel 路径和内存形态，不代表某个训练 adapter 的模型精度。",
            "- Qwen3.5-27B 是 dense 模型，没有 expert/routing，因此不能用于验证 moe_lora_build_combined_idx 的真实模型端到端收益。",
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
        default=script_dir / "moe_lora_build_combined_idx_real_weight_perf_cases.jsonl",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=Path("/opt/wqy2/temp/moe_lora_real_weight_profiler_trace"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=script_dir
        / "moe_lora_build_combined_idx_real_weight_profiler_report.md",
    )
    parser.add_argument(
        "--only-tokens",
        type=int,
        action="append",
        help="只运行指定 token 数；可重复传入。",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="独立 profiler 采集轮数；每轮内部固定 warmup=5、active=5。",
    )
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
        raise ValueError(f"all real-weight cases must use one rank, got {ranks}")
    weights, metadata = load_real_weights(args.model_root, ranks.pop(), device)
    if any(int(case_attrs(case)["top_k"]) != metadata["top_k"] for case in cases):
        raise ValueError("case top_k does not match checkpoint")
    if any(
        int(case_attrs(case)["num_experts"]) != metadata["num_experts"]
        for case in cases
    ):
        raise ValueError("case num_experts does not match checkpoint")
    metadata["trials"] = args.trials
    metadata["visible_devices"] = os.environ.get(
        "ASCEND_RT_VISIBLE_DEVICES", "not-set"
    )

    results = []
    for case_id, case in enumerate(cases):
        attrs = case_attrs(case)
        tokens = int(attrs["tokens"])
        inputs = build_inputs(tokens, metadata["top_k"], weights, device)
        custom_output = custom_forward(inputs, weights)
        baseline_output = baseline_forward(inputs, weights)
        torch.npu.synchronize()
        if not torch.equal(custom_output.cpu(), baseline_output.cpu()):
            max_abs = (
                custom_output.float() - baseline_output.float()
            ).abs().max().item()
            raise AssertionError(
                f"real-weight precision mismatch tokens={tokens} max_abs={max_abs}"
            )
        timings: dict[str, list[float]] = {"custom": [], "baseline": []}
        breakdowns: dict[str, list[dict[str, float]]] = {
            "custom": [],
            "baseline": [],
        }
        forwards = {
            "custom": bind_forward(custom_forward, inputs, weights),
            "baseline": bind_forward(baseline_forward, inputs, weights),
        }
        for trial in range(args.trials):
            order = (
                ("custom", "baseline")
                if trial % 2 == 0
                else ("baseline", "custom")
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
        custom_us = statistics.median(timings["custom"])
        baseline_us = statistics.median(timings["baseline"])
        speedup = baseline_us / custom_us
        result = {
            "case": case_id,
            "tokens": tokens,
            "custom_us": custom_us,
            "baseline_us": baseline_us,
            "speedup": speedup,
            "custom_samples": timings["custom"],
            "baseline_samples": timings["baseline"],
            "custom_breakdown": median_breakdown(breakdowns["custom"]),
            "baseline_breakdown": median_breakdown(breakdowns["baseline"]),
        }
        results.append(result)
        print(
            f"[INFO] case={case_id} tokens={tokens} custom={custom_us:.3f}us "
            f"baseline={baseline_us:.3f}us speedup={speedup:.3f}x"
        )

    args.report.write_text(
        render_report(results, metadata, args.case_file, args.trace_root),
        encoding="utf-8",
    )
    print(f"[INFO] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
