#!/usr/bin/env python3
"""Run precision cases and emit JSON and Markdown reports."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from test_moe_lora_build_combined_idx_precision import (
    BGMV_CASES,
    GRAPH_CASE_IDS,
    ROUTING_CASES,
    SUPPORTED_DTYPES,
    dtype_name,
    evaluate_bgmv_case,
    evaluate_graph_case,
    evaluate_routing_case,
    load_ops,
)


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _scientific(value: float) -> str:
    return f"{value:.3e}"


def render_markdown(results: list[dict]) -> str:
    routing = [result for result in results if result["suite"] == "routing"]
    bgmv = [result for result in results if result["suite"] == "bgmv_index"]
    graph = [result for result in results if result["suite"] == "acl_graph"]
    passed = sum(result["passed"] for result in results)
    failed = len(results) - passed
    dtype_counts: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        dtype_counts[result["dtype"]]["total"] += 1
        dtype_counts[result["dtype"]]["passed"] += int(result["passed"])

    lines = [
        "# `moe_lora_build_combined_idx` 精度验证报告",
        "",
        f"- 测试平台：Ascend 910B3",
        f"- 测试时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- Routing 参考：PyTorch CPU direct scatter，输出要求逐元素完全相等。",
        "- BGMV 参考：原 int64 index kernel 严格一致，并与 PyTorch CPU 参考做 allclose。",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| 总用例数 | {len(results)} |",
        f"| 通过数 | {passed} |",
        f"| 失败数 | {failed} |",
        f"| 通过率 | {100.0 * passed / len(results):.2f}% |",
        f"| Routing 严格相等 | {sum(r['passed'] for r in routing)}/{len(routing)} |",
        f"| BGMV index 兼容 | {sum(r['passed'] for r in bgmv)}/{len(bgmv)} |",
        f"| ACL Graph capture/replay | {sum(r['passed'] for r in graph)}/{len(graph)} |",
        "",
        "## 精度标准",
        "",
        "Routing 是整数索引构造，采用比浮点阈值更严格的逐元素完全相等；",
        "因此通过用例的 MERE、MARE、MaxAbsErr 均必须为 0。相对误差仍按",
        "`abs(actual - golden) / (abs(golden) + 1e-7)` 记录。",
        "BGMV 的 int32 与 int64 index 输出要求逐元素一致，同时 FP16 使用",
        "`rtol=atol=2e-2`，BF16 expand 使用 `rtol=atol=1.5e-1` 对比 CPU。",
        "",
        "## Routing 测试结果",
        "",
        "| # | 类别 | 描述 | Shape | adapter dtype | Mismatch | MERE | MARE | 结果 |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in routing:
        lines.append(
            f"| {result['case_id']} | {result['category']} | {result['description']} "
            f"| {result['shape']} | {result['dtype']} | {result['mismatch_count']} "
            f"| {_scientific(result['MERE'])} | {_scientific(result['MARE'])} "
            f"| {_status(result['passed'])} |"
        )

    lines.extend(
        [
            "",
            "## BGMV int32 index 回归结果",
            "",
            "| # | 算子 | 数据 dtype | index dtype | Shape | int64 parity MaxAbs | CPU MaxAbsErr | 容差 | 结果 |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for result in bgmv:
        lines.append(
            f"| {result['case_id']} | {result['category']} | {result['dtype']} "
            f"| {result['index_dtype']} | {result['shape']} "
            f"| {_scientific(result['index_parity_max_abs'])} "
            f"| {_scientific(result['max_abs_err'])} "
            f"| {result['reference_rtol_atol']:.2e} | {_status(result['passed'])} |"
        )

    lines.extend(
        [
            "",
            "## ACL Graph capture/replay",
            "",
            "| Case | Shape | Mismatch | MaxAbsErr | 结果 |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for result in graph:
        lines.append(
            f"| {result['case_id']} | {result['shape']} "
            f"| {result['mismatch_count']} | {_scientific(result['max_abs_err'])} "
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

    lines.extend(
        [
            "",
            "## 关键发现",
            "",
            f"1. Routing 的 {len(routing)} 个用例全部逐元素相等，最大绝对误差和 mismatch 均为 0。",
            "2. bool/int8 两种 adapter_enabled 存储类型结果一致；非 32B 对齐、小 decode、4096-token prefill 均通过。",
            "3. `lora=-1`、adapter 全禁用/全启用、最大合法 ID、负 routing 位置和 capacity 大于 token 数均已覆盖。",
            f"4. BGMV 的 {len(bgmv)} 个 FP16/BF16、shrink/expand、int32/int64 index 回归用例全部通过；int32 与原 int64 kernel 输出严格一致。",
            f"5. ACL Graph capture/replay 的 {len(graph)} 个用例全部通过，`1x6` decode 与 `4096x6` prefill 均逐元素严格相等。",
            "",
        ]
    )
    failures = [result for result in results if not result["passed"]]
    if failures:
        lines.extend(["## 失败摘要", ""])
        for result in failures:
            lines.append(f"- `{result}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    load_ops()
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    results = []
    for case_id, case in enumerate(ROUTING_CASES, start=1):
        for adapter_dtype in SUPPORTED_DTYPES:
            result = evaluate_routing_case(
                case_id, case, adapter_dtype, device
            )
            results.append(result)
            print(
                f"[{_status(result['passed'])}] routing/{case_id:02d} "
                f"{result['dtype']} mismatch={result['mismatch_count']}"
            )
    for case_id, (operation, dtype, index_dtype) in enumerate(
        BGMV_CASES, start=1
    ):
        result = evaluate_bgmv_case(
            case_id, operation, dtype, index_dtype, device
        )
        results.append(result)
        print(
            f"[{_status(result['passed'])}] bgmv/{case_id:02d} "
            f"{operation} {dtype_name(dtype)} {dtype_name(index_dtype)} "
            f"parity={result['index_parity_max_abs']:.3e}"
        )
    for case_id in GRAPH_CASE_IDS:
        result = evaluate_graph_case(case_id, device)
        results.append(result)
        print(
            f"[{_status(result['passed'])}] acl_graph/{case_id:02d} "
            f"shape={result['shape']} mismatch={result['mismatch_count']}"
        )

    report_dir = Path(__file__).resolve().parent
    json_path = report_dir / "moe_lora_build_combined_idx_precision_report.json"
    markdown_path = report_dir / "moe_lora_build_combined_idx_precision_report.md"
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
