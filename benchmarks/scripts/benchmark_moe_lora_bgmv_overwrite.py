"""Benchmark zero-fill fusion for decode MoE LoRA BGMV kernels."""

import argparse
import json
import time
from collections.abc import Callable

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _run_projection(
    inputs: torch.Tensor,
    a_weights: tuple[torch.Tensor, ...],
    b_weights: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    output: torch.Tensor,
    *,
    add_inputs: bool,
) -> None:
    if add_inputs:
        output.zero_()
    offset = 0
    shrink_factory = torch.zeros if add_inputs else torch.empty
    for a_weight, b_weight in zip(a_weights, b_weights, strict=True):
        rank = a_weight.shape[-2]
        shrink = shrink_factory((inputs.shape[0], rank), dtype=torch.float32, device=inputs.device)
        torch.ops._C_ascend.bgmv_shrink(inputs, a_weight, indices, shrink, 1.0)
        output_size = b_weight.shape[-2]
        torch.ops._C_ascend.bgmv_expand(
            shrink,
            b_weight,
            indices,
            output,
            offset,
            output_size,
            add_inputs,
        )
        offset += output_size


def _benchmark(fn: Callable[[], None], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1_000_000 / iterations


def _weights(
    num_weight_sets: int,
    input_size: int,
    output_sizes: tuple[int, ...],
    rank: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    a_weights = tuple(
        torch.randn(num_weight_sets, rank, input_size, dtype=torch.bfloat16, device="npu") for _ in output_sizes
    )
    b_weights = tuple(
        torch.randn(num_weight_sets, output_size, rank, dtype=torch.bfloat16, device="npu")
        for output_size in output_sizes
    )
    return a_weights, b_weights


def _difference_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int | bool]:
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    difference = (actual_cpu.float() - expected_cpu.float()).abs()
    return {
        "bitwise_equal": torch.equal(actual_cpu, expected_cpu),
        "mismatched_elements": int(torch.count_nonzero(actual_cpu != expected_cpu)),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--num-weight-sets", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    hidden_size = 4096
    intermediate_size = 2048
    rank = 16
    indices = torch.arange(args.rows, dtype=torch.int64, device="npu").remainder(args.num_weight_sets)
    indices[2::4] = -1

    w13_input = torch.randn(args.rows, hidden_size, dtype=torch.bfloat16, device="npu")
    w13_a, w13_b = _weights(args.num_weight_sets, hidden_size, (intermediate_size, intermediate_size), rank)
    w13_baseline = torch.empty(args.rows, intermediate_size * 2, dtype=torch.bfloat16, device="npu")
    w13_fused = torch.empty_like(w13_baseline)

    w2_input = torch.randn(args.rows, intermediate_size, dtype=torch.bfloat16, device="npu")
    w2_a, w2_b = _weights(args.num_weight_sets, intermediate_size, (hidden_size,), rank)
    w2_baseline = torch.empty(args.rows, hidden_size, dtype=torch.bfloat16, device="npu")
    w2_fused = torch.empty_like(w2_baseline)

    def baseline() -> None:
        _run_projection(w13_input, w13_a, w13_b, indices, w13_baseline, add_inputs=True)
        _run_projection(w2_input, w2_a, w2_b, indices, w2_baseline, add_inputs=True)

    def fused() -> None:
        _run_projection(w13_input, w13_a, w13_b, indices, w13_fused, add_inputs=False)
        _run_projection(w2_input, w2_a, w2_b, indices, w2_fused, add_inputs=False)

    baseline()
    fused()
    torch.npu.synchronize()
    w13_difference = _difference_stats(w13_fused, w13_baseline)
    w2_difference = _difference_stats(w2_fused, w2_baseline)
    torch.testing.assert_close(w13_fused.cpu(), w13_baseline.cpu(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w2_fused.cpu(), w2_baseline.cpu(), atol=2e-2, rtol=2e-2)

    baseline_us = _benchmark(baseline, warmup=args.warmup, iterations=args.iterations)
    fused_us = _benchmark(fused, warmup=args.warmup, iterations=args.iterations)
    result = {
        "rows": args.rows,
        "num_weight_sets": args.num_weight_sets,
        "rank": rank,
        "baseline_us": baseline_us,
        "fused_us": fused_us,
        "speedup_percent": (baseline_us - fused_us) / baseline_us * 100,
        "removed_zero_fill_launches_per_layer": 5,
        "w13_difference": w13_difference,
        "w2_difference": w2_difference,
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            output_file.write(payload + "\n")


if __name__ == "__main__":
    main()
