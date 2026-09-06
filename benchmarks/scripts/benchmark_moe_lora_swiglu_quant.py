#!/usr/bin/env python3
"""Benchmark DSV4 prefill clipped-SwiGLU activation before INT8 quant."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
import torch_npu


def _benchmark_functions(
    functions: dict[str, Callable[[], object]],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, float]:
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    torch.npu.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in functions}
    names = list(functions)
    for round_index in range(rounds):
        ordered_names = names if round_index % 2 == 0 else list(reversed(names))
        for name in ordered_names:
            start = time.perf_counter()
            for _ in range(iterations):
                functions[name]()
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e6 / iterations)
    return {name: statistics.median(values) for name, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[96, 384, 1536, 6144])
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--with-topk-scales", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    print("rows,baseline_us,clipped_swiglu_us,speedup,max_quant_diff,max_scale_abs_diff")

    for rows in args.rows:
        torch.manual_seed(0)
        x = (
            torch.randn(
                rows,
                args.intermediate_size * 2,
                dtype=torch.bfloat16,
                device=device,
            )
            * 4
        )
        topk_scales = None
        if args.with_topk_scales:
            topk_scales = torch.rand(rows, 1, dtype=torch.bfloat16, device=device)

        def baseline(
            x: torch.Tensor = x,
            topk_scales: torch.Tensor | None = topk_scales,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            gate, up = x.chunk(2, dim=-1)
            if args.swiglu_limit > 0:
                gate = gate.clamp(max=args.swiglu_limit)
                up = up.clamp(min=-args.swiglu_limit, max=args.swiglu_limit)
                activation_input = torch.cat((gate, up), dim=-1)
            else:
                activation_input = x
            activated = torch_npu.npu_swiglu(activation_input)
            if topk_scales is not None:
                activated *= topk_scales
            return torch_npu.npu_dynamic_quant(activated)

        def clipped_swiglu(
            x: torch.Tensor = x,
            topk_scales: torch.Tensor | None = topk_scales,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            activated = torch_npu.npu_clipped_swiglu(
                x,
                interleaved=False,
                alpha=1.0,
                limit=args.swiglu_limit,
                bias=0.0,
            )
            if topk_scales is not None:
                activated *= topk_scales
            return torch_npu.npu_dynamic_quant(activated)

        expected, expected_scale = baseline()
        clipped_actual, clipped_actual_scale = clipped_swiglu()
        torch.npu.synchronize()
        clipped_quant_diff = (clipped_actual.to(torch.int16) - expected.to(torch.int16)).abs()
        clipped_scale_diff = (clipped_actual_scale.reshape(-1) - expected_scale.reshape(-1)).abs()

        timings = _benchmark_functions(
            {"baseline": baseline, "clipped_swiglu": clipped_swiglu},
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        print(
            f"{rows},{timings['baseline']:.3f},{timings['clipped_swiglu']:.3f},"
            f"{timings['baseline'] / timings['clipped_swiglu']:.4f},{clipped_quant_diff.max().item()},"
            f"{clipped_scale_diff.max().item():.8f}"
        )


if __name__ == "__main__":
    main()
