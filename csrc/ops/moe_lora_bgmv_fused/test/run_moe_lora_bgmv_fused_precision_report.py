#!/usr/bin/env python3
"""Run fused BGMV precision cases and emit JSON and Markdown reports."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from test_moe_lora_bgmv_fused_precision import (
    PRECISION_CASES,
    SUPPORTED_DTYPES,
    dtype_name,
    evaluate_case,
    load_ops,
)


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _scientific(value: float) -> str:
    return f"{value:.3e}"


def render_markdown(results: list[dict]) -> str:
    passed = sum(result["passed"] for result in results)
    failed = len(results) - passed
    dtype_counts: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        dtype_counts[result["dtype"]]["total"] += 1
        dtype_counts[result["dtype"]]["passed"] += int(result["passed"])

    lines = [
        "# `moe_lora_bgmv_fused` 精度验证报告",
        "",
        "- 测试平台：Ascend 910B3。",
        f"- 测试时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- 参考实现：PyTorch CPU FP32 两阶段 BMM，最终 Cast 回输出 dtype。",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| 总用例数 | {len(results)} |",
        f"| 通过数 | {passed} |",
        f"| 失败数 | {failed} |",
        f"| 通过率 | {100.0 * passed / len(results):.2f}% |",
        "",
        "## 精度标准",
        "",
        "相对误差按 `abs(actual - golden) / (abs(golden) + 1e-7)` 计算；",
        "FP16 要求 MERE < 2^-10 且 MARE < 10 * 2^-10，BF16 要求",
        "MERE < 2^-7 且 MARE < 10 * 2^-7。返回 Tensor 必须与 y alias，",
        "slice 外数据必须逐元素不变。",
        "",
        "## 用例结果",
        "",
        "| # | 类别 | 描述 | Shape(M,H,O,Y) | dtype | index dtype | MERE | MARE | MaxAbs | alias/slice | 结果 |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        alias_slice = (
            f"{result['alias_preserved']}/{result['outside_slice_equal']}"
        )
        lines.append(
            f"| {result['case_id']} | {result['category']} | {result['description']} "
            f"| {result['shape']} | {result['dtype']} | {result['index_dtype']} "
            f"| {_scientific(result['MERE'])} | {_scientific(result['MARE'])} "
            f"| {_scientific(result['max_abs_err'])} | {alias_slice} "
            f"| {_status(result['passed'])} |"
        )

    lines.extend(
        [
            "",
            "## 按 dtype 汇总",
            "",
            "| dtype | 用例数 | 通过数 | 失败数 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in sorted(dtype_counts):
        count = dtype_counts[name]
        lines.append(
            f"| {name} | {count['total']} | {count['passed']} "
            f"| {count['total'] - count['passed']} |"
        )

    grouped = [result for result in results if result["index_pattern"] == "expert_sorted"]
    fallback = [result for result in results if result["index_pattern"] in {"alternating", "mixed_within4"}]
    lines.extend(
        [
            "",
            "## 关键发现",
            "",
            f"1. 共 {len(results)} 个 FP16/BF16 与 int32/int64 组合，覆盖 1-row fallback、4/8-row 复用和尾块。",
            f"2. expert-sorted 快速路径覆盖 {len(grouped)} 例，mixed/alternating fallback 覆盖 {len(fallback)} 例。",
            "3. 非 32B 对齐 H/O、slice offset、负/零 scale、全 -1 index 和最大合法 index 均纳入验证。",
            "4. 每个用例同时检查返回 alias 和 slice 外逐元素不变，避免只验证数值而漏掉 in-place 语义。",
            "",
        ]
    )
    failures = [result for result in results if not result["passed"]]
    if failures:
        lines.extend(["## 失败摘要", ""])
        for result in failures:
            lines.append(
                f"- Case {result['case_id']} {result['description']} "
                f"{result['dtype']}/{result['index_dtype']}: "
                f"MERE={result['MERE']:.3e}, MARE={result['MARE']:.3e}, "
                f"MaxAbs={result['max_abs_err']:.3e}."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    load_ops()
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    results = []
    for case_id, case in enumerate(PRECISION_CASES, start=1):
        for dtype in SUPPORTED_DTYPES:
            result = evaluate_case(case_id, case, dtype, device)
            results.append(result)
            print(
                f"[{_status(result['passed'])}] {case_id:02d} "
                f"{dtype_name(dtype)} {result['index_dtype']} "
                f"MERE={result['MERE']:.3e} MARE={result['MARE']:.3e}"
            )

    report_dir = Path(__file__).resolve().parent
    json_path = report_dir / "moe_lora_bgmv_fused_precision_report.json"
    markdown_path = report_dir / "moe_lora_bgmv_fused_precision_report.md"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(results), encoding="utf-8")

    passed = sum(result["passed"] for result in results)
    print(
        f"PRECISION_SUMMARY total={len(results)} passed={passed} "
        f"failed={len(results) - passed}"
    )
    print(f"JSON_REPORT {json_path}")
    print(f"MARKDOWN_REPORT {markdown_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
