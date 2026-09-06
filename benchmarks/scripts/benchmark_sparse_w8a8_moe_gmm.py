#!/usr/bin/env python3
"""Benchmark sparse GroupedMatmulV5 routing for W8A8 MoE decode."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: F401


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


def _make_group_list(
    rows: int,
    *,
    num_experts: int,
    active_experts: int,
    device: torch.device,
) -> torch.Tensor:
    active_experts = min(rows, num_experts, active_experts)
    counts = torch.zeros(num_experts, dtype=torch.int64)
    base_count, remainder = divmod(rows, active_experts)
    counts[:active_experts] = base_count
    counts[:remainder].add_(1)
    return counts.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--active-experts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
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

    print("rows,active_experts,count_us,sparse_gmm_us,sparse_e2e_us,gmm_speedup,e2e_speedup,prepare_us")
    for rows in args.rows:
        inputs = torch.zeros(rows, args.k, dtype=torch.int8, device=device)
        per_token_scale = torch.ones(rows, dtype=torch.float32, device=device)
        for active_experts in args.active_experts:
            if active_experts > min(rows, args.num_experts):
                continue
            group_list = _make_group_list(
                rows,
                num_experts=args.num_experts,
                active_experts=active_experts,
                device=device,
            )
            sparse_group_list = torch.empty(
                args.num_experts,
                2,
                dtype=torch.int64,
                device=device,
            )

            def prepare_sparse(
                group_list: torch.Tensor = group_list,
                sparse_group_list: torch.Tensor = sparse_group_list,
            ) -> None:
                torch.ops._C_ascend.moe_lora_prepare_sparse_group_list(
                    group_list,
                    sparse_group_list,
                )

            def grouped_matmul(
                groups: torch.Tensor,
                group_list_type: int,
                inputs: torch.Tensor = inputs,
                per_token_scale: torch.Tensor = per_token_scale,
            ) -> torch.Tensor:
                return torch_npu.npu_grouped_matmul(
                    x=[inputs],
                    weight=[weight],
                    scale=[scale],
                    per_token_scale=[per_token_scale],
                    split_item=2,
                    group_type=0,
                    group_list=groups,
                    group_list_type=group_list_type,
                    output_dtype=torch.bfloat16,
                )[0]

            def count_gmm(group_list: torch.Tensor = group_list) -> torch.Tensor:
                return grouped_matmul(group_list, 1)

            def sparse_gmm(
                sparse_group_list: torch.Tensor = sparse_group_list,
            ) -> torch.Tensor:
                return grouped_matmul(sparse_group_list, 2)

            def sparse_e2e() -> torch.Tensor:
                prepare_sparse()
                return sparse_gmm()

            prepare_sparse()
            expected = count_gmm()
            actual = sparse_gmm()
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)

            timings = _benchmark_functions(
                {
                    "count": count_gmm,
                    "sparse_gmm": sparse_gmm,
                    "sparse_e2e": sparse_e2e,
                    "prepare": prepare_sparse,
                },
                warmup=args.warmup,
                iterations=args.iterations,
                rounds=args.rounds,
            )
            count_us = timings["count"]
            sparse_gmm_us = timings["sparse_gmm"]
            sparse_e2e_us = timings["sparse_e2e"]
            prepare_us = timings["prepare"]
            print(
                f"{rows},{active_experts},{count_us:.3f},{sparse_gmm_us:.3f},"
                f"{sparse_e2e_us:.3f},{count_us / sparse_gmm_us:.4f},"
                f"{count_us / sparse_e2e_us:.4f},{prepare_us:.3f}"
            )


if __name__ == "__main__":
    main()
