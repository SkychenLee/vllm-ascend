#!/usr/bin/env python3
"""Compare mixed-LoRA GMM with the ordinary per-row BGMV baseline.

Rows are already expert-major, matching the output of MoE init-routing.  The
baseline therefore applies the same ``(LoRA slot, expert)`` BGMV index used by
the production fallback.  The candidate path additionally groups rows by the
composite key and uses two grouped matmuls, matching quant_moe.py.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: F401


def _benchmark(fn: Callable[[], None], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1_000_000 / iterations


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> str:
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    absolute = (actual_cpu.float() - expected_cpu.float()).abs()
    exact = actual_cpu == expected_cpu
    return (
        f"exact={exact.float().mean().item():.8f} "
        f"mismatch={torch.count_nonzero(~exact).item()} "
        f"max_abs={absolute.max().item():.8f} "
        f"mean_abs={absolute.mean().item():.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--loras", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=7168)
    parser.add_argument("--output-size", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    dtype = torch.bfloat16

    base_count, remainder = divmod(args.rows, args.experts)
    group_counts = torch.full((args.experts,), base_count, dtype=torch.int64)
    group_counts[:remainder].add_(1)
    expert_ids = torch.repeat_interleave(
        torch.arange(args.experts, dtype=torch.int64),
        group_counts,
        output_size=args.rows,
    ).to(device)
    slots = (torch.arange(args.rows, device=device) % args.loras).to(torch.float32)
    adapter_enabled = torch.ones(args.loras, dtype=torch.int32, device=device)
    group_counts = group_counts.to(device)

    composite_ids = torch.empty(args.rows, dtype=torch.int32, device=device)
    composite_counts = torch.empty(args.loras * args.experts, dtype=torch.int64, device=device)
    enabled = torch.empty(args.rows, dtype=torch.bool, device=device)
    torch.ops._C_ascend.moe_lora_prepare_composite_gmm_routing(
        slots,
        group_counts,
        adapter_enabled,
        composite_ids,
        composite_counts,
        enabled,
    )
    bgmv_indices = slots.to(torch.int64) * args.experts + expert_ids

    inputs = torch.randn(args.rows, args.input_size, dtype=dtype, device=device)
    weights_a = torch.randn(
        args.loras * args.experts,
        args.rank,
        args.input_size,
        dtype=dtype,
        device=device,
    )
    weights_b = torch.randn(
        args.loras * args.experts,
        args.output_size,
        args.rank,
        dtype=dtype,
        device=device,
    )
    base = torch.randn(args.rows, args.output_size, dtype=dtype, device=device)
    bgmv_output = torch.empty_like(base)
    gmm_output = torch.empty_like(base)
    fp32_gmm_output = torch.empty_like(base)
    bgmv_shrink = torch.empty(args.rows, args.rank, dtype=torch.float32, device=device)

    grouped_inputs, reverse_mapping = torch_npu.npu_moe_token_permute(
        tokens=inputs,
        indices=composite_ids,
        num_out_tokens=args.rows,
    )
    weights_a_t = weights_a.transpose(-1, -2)
    weights_b_t = weights_b.transpose(-1, -2)
    weights_a_fp32_t = weights_a.float().transpose(-1, -2)
    weights_b_fp32_t = weights_b.float().transpose(-1, -2)

    def run_bgmv() -> None:
        bgmv_output.copy_(base)
        torch.ops._C_ascend.bgmv_shrink(inputs, weights_a, bgmv_indices, bgmv_shrink, 1.0)
        torch.ops._C_ascend.bgmv_expand(
            bgmv_shrink,
            weights_b,
            bgmv_indices,
            bgmv_output,
            0,
            args.output_size,
            True,
        )

    def run_gmm() -> None:
        shrink = torch_npu.npu_grouped_matmul(
            x=[grouped_inputs],
            weight=[weights_a_t],
            split_item=2,
            group_type=0,
            group_list=composite_counts,
            group_list_type=1,
        )[0]
        delta = torch_npu.npu_grouped_matmul(
            x=[shrink],
            weight=[weights_b_t],
            split_item=2,
            group_type=0,
            group_list=composite_counts,
            group_list_type=1,
        )[0]
        delta = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=delta,
            sorted_indices=reverse_mapping,
        )
        gmm_output.copy_(base).add_(delta)

    def run_fp32_gmm() -> None:
        shrink = torch_npu.npu_grouped_matmul(
            x=[grouped_inputs.float()],
            weight=[weights_a_fp32_t],
            split_item=2,
            group_type=0,
            group_list=composite_counts,
            group_list_type=1,
            output_dtype=torch.float32,
        )[0]
        delta = torch_npu.npu_grouped_matmul(
            x=[shrink],
            weight=[weights_b_fp32_t],
            split_item=2,
            group_type=0,
            group_list=composite_counts,
            group_list_type=1,
            output_dtype=torch.float32,
        )[0]
        delta = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=delta,
            sorted_indices=reverse_mapping,
        )
        fp32_gmm_output.copy_(base).add_(delta)

    run_bgmv()
    run_gmm()
    run_fp32_gmm()
    torch.npu.synchronize()
    print(f"bf16_gmm_vs_bgmv {_difference(gmm_output, bgmv_output)}")
    print(f"fp32_gmm_vs_bgmv {_difference(fp32_gmm_output, bgmv_output)}")
    bgmv_us = _benchmark(run_bgmv, warmup=args.warmup, iterations=args.iterations)
    gmm_us = _benchmark(run_gmm, warmup=args.warmup, iterations=args.iterations)
    fp32_gmm_us = _benchmark(run_fp32_gmm, warmup=args.warmup, iterations=args.iterations)
    print(
        f"rows={args.rows} experts={args.experts} loras={args.loras} "
        f"K={args.input_size} N={args.output_size} rank={args.rank}"
    )
    print(f"bgmv_us={bgmv_us:.3f}")
    print(f"bf16_gmm_us={gmm_us:.3f} speedup={bgmv_us / gmm_us:.3f}x")
    print(f"fp32_gmm_us={fp32_gmm_us:.3f} speedup={bgmv_us / fp32_gmm_us:.3f}x")


if __name__ == "__main__":
    main()
