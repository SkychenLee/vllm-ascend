#!/usr/bin/env python3
"""Benchmark GroupedMatmulV5 tuning_config for W8A8 MoE."""

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
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--active-experts", type=int, default=8)
    parser.add_argument("--expected-tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64, 256])
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    weight = torch.zeros(
        args.num_experts,
        args.k,
        args.n,
        dtype=torch.int8,
        device=device,
    )
    scale = torch.ones(
        args.num_experts,
        args.n,
        dtype=torch.bfloat16,
        device=device,
    )

    print("rows,active_experts,config,time_us,speedup")
    for rows in args.rows:
        active_experts = min(rows, args.num_experts, args.active_experts)
        group_list = torch.zeros(args.num_experts, dtype=torch.int64)
        base_count, remainder = divmod(rows, active_experts)
        group_list[:active_experts] = base_count
        group_list[:remainder].add_(1)
        group_list = group_list.to(device)
        inputs = torch.zeros(rows, args.k, dtype=torch.int8, device=device)
        per_token_scale = torch.ones(rows, dtype=torch.float32, device=device)

        def grouped_matmul(
            tuning_config: list[int] | None,
            inputs: torch.Tensor = inputs,
            per_token_scale: torch.Tensor = per_token_scale,
            group_list: torch.Tensor = group_list,
        ) -> torch.Tensor:
            kwargs = {
                "x": [inputs],
                "weight": [weight],
                "scale": [scale],
                "per_token_scale": [per_token_scale],
                "split_item": 2,
                "group_type": 0,
                "group_list": group_list,
                "group_list_type": 1,
                "output_dtype": torch.bfloat16,
            }
            if tuning_config is not None:
                kwargs["tuning_config"] = tuning_config
            return torch_npu.npu_grouped_matmul(**kwargs)[0]

        functions: dict[str, Callable[[], torch.Tensor]] = {
            "default": lambda: grouped_matmul(None),
        }
        for expected_tokens in args.expected_tokens:
            if expected_tokens > rows:
                continue
            tuning_config = [expected_tokens, 0, -1]
            functions[f"tokens={expected_tokens}"] = lambda tuning_config=tuning_config: grouped_matmul(tuning_config)

        expected = functions["default"]()
        for name, fn in functions.items():
            actual = fn()
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, msg=name)

        timings = _benchmark_functions(
            functions,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        default_us = timings["default"]
        for name, value in timings.items():
            print(f"{rows},{active_experts},{name},{value:.3f},{default_us / value:.4f}")


if __name__ == "__main__":
    main()
