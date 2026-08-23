#!/usr/bin/env python3
"""torch_npu.profiler benchmark for fused MoE-LoRA routing."""

from __future__ import annotations

import argparse
import os

import torch
import torch_npu  # noqa: F401

from moe_lora_build_combined_idx_profiler_common import (
    baseline_forward,
    build_inputs,
    custom_forward,
    load_cases,
    load_custom_library,
    profile_forward,
    render_report,
)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-file",
        default=os.path.join(here, "moe_lora_build_combined_idx_perf_cases.jsonl"),
    )
    parser.add_argument("--trace-root", required=True)
    parser.add_argument(
        "--report-md",
        default=os.path.join(here, "moe_lora_build_combined_idx_torch_npu_profiler_report.md"),
    )
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    print(f"[INFO] custom library: {load_custom_library()}")
    device = torch.device(args.device)
    cases = load_cases(args.case_file)
    results = []
    for case_index, case in enumerate(cases):
        inputs = build_inputs(case, device, 20260822 + case_index)
        custom_value = custom_forward(inputs)
        baseline_value = baseline_forward(inputs)
        torch.npu.synchronize()
        if not torch.equal(custom_value.cpu(), baseline_value.cpu()):
            raise AssertionError(f"precision mismatch before profiling case {case_index}")

        custom_us = profile_forward(
            lambda: custom_forward(inputs),
            os.path.join(args.trace_root, "custom", f"case_{case_index:03d}"),
        )
        baseline_us = profile_forward(
            lambda: baseline_forward(inputs),
            os.path.join(args.trace_root, "baseline", f"case_{case_index:03d}"),
        )
        row = {
            "case": case_index,
            "tokens": inputs["tokens"],
            "top_k": inputs["top_k"],
            "dtype": inputs["dtype"],
            "custom_us": custom_us,
            "baseline_us": baseline_us,
            "speedup": baseline_us / custom_us,
        }
        results.append(row)
        print(
            f"[INFO] case={case_index} custom={custom_us:.3f}us "
            f"baseline={baseline_us:.3f}us speedup={row['speedup']:.3f}x"
        )

    report = render_report(
        results,
        os.path.abspath(args.case_file),
        os.path.abspath(args.trace_root),
    )
    with open(args.report_md, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"[INFO] wrote {args.report_md}")


if __name__ == "__main__":
    main()
