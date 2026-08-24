"""Phase-0 benchmark for the large-prefill MoE LoRA grouped path.

This benchmark validates the existing A2 MoeGroupedMatmul kernel in the
low-rank shrink shape before the routing kernels are enabled.  It compares
the ND, transposed-weight GMM binding with the current BGMV implementation.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op


def _measure(
    fn: Callable[[], Any], warmup: int, iterations: int, repeats: int
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    samples.sort()
    return {
        "p50_us": statistics.median(samples),
        "p95_us": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        "best_us": samples[0],
    }


def _group_counts(m: int, groups: int, distribution: str) -> list[int]:
    if distribution == "hotspot":
        return [m] + [0] * (groups - 1)
    if distribution != "fragmented":
        raise ValueError(f"unsupported distribution: {distribution}")
    quotient, remainder = divmod(m, groups)
    return [quotient + (group < remainder) for group in range(groups)]


def _make_case(
    m: int,
    k: int,
    groups: int,
    distribution: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = _group_counts(m, groups, distribution)
    active_groups = [(group, count) for group, count in enumerate(counts) if count]
    # The custom kernel expects compact [group_id, row_count] entries followed
    # by zero-count rows.  Input rows are concatenated in the same order.
    group_list_cpu = torch.zeros((groups, 2), dtype=torch.int32)
    indices_cpu = torch.empty(m, dtype=torch.int64)
    row = 0
    for entry, (group, count) in enumerate(active_groups):
        group_list_cpu[entry] = torch.tensor([group, count], dtype=torch.int32)
        indices_cpu[row : row + count] = group
        row += count

    x = torch.randn((m, k), dtype=dtype, device="npu") * 0.05
    weight = torch.randn((groups, 16, k), dtype=dtype, device="npu") * 0.05
    return x, weight, group_list_cpu.npu(), indices_cpu.npu()


def _reference(
    x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor
) -> torch.Tensor:
    group_list_cpu = group_list.cpu()
    pieces = []
    offset = 0
    for group, count in group_list_cpu.tolist():
        if count <= 0:
            break
        pieces.append(torch.matmul(x[offset : offset + count], weight[group].transpose(0, 1)))
        offset += count
    if not pieces:
        return torch.empty((0, 16), dtype=x.dtype, device=x.device)
    return torch.cat(pieces, dim=0)


def _aclgraph_check(
    x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor
) -> dict[str, Any]:
    graph = torch.npu.NPUGraph()
    try:
        # Warm up executor/cache creation outside graph capture.
        eager = torch.ops._C_ascend.moe_lora_grouped_matmul(x, weight, group_list)
        torch.npu.synchronize()
        with torch.npu.graph(
            graph, capture_error_mode="thread_local", auto_dispatch_capture=True
        ):
            captured = torch.ops._C_ascend.moe_lora_grouped_matmul(x, weight, group_list)
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        max_abs = float((captured.float() - eager.float()).abs().max().cpu())
        return {"passed": max_abs == 0.0, "replays": 10, "max_abs": max_abs}
    except Exception as error:  # Report the gate failure in benchmark JSON.
        return {"passed": False, "error": str(error)}


def _run_case(
    m: int,
    k: int,
    groups: int,
    distribution: str,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
    check_graph: bool,
) -> dict[str, Any]:
    x, weight, group_list, indices = _make_case(
        m, k, groups, distribution, dtype
    )
    gmm_output = torch.ops._C_ascend.moe_lora_grouped_matmul(x, weight, group_list)
    bgmv_output = torch.empty((m, 16), dtype=torch.float32, device="npu")
    torch.ops._C_ascend.bgmv_shrink(x, weight, indices, bgmv_output, 1.0)
    expected = _reference(x, weight, group_list)
    torch.npu.synchronize()

    difference = (gmm_output.float() - expected.float()).abs()
    tolerance = 2**-9 if dtype == torch.float16 else 2**-6
    matched = torch.isclose(
        gmm_output.float(), expected.float(), rtol=tolerance, atol=tolerance
    )
    gmm_perf = _measure(
        lambda: torch.ops._C_ascend.moe_lora_grouped_matmul(x, weight, group_list),
        warmup,
        iterations,
        repeats,
    )
    bgmv_perf = _measure(
        lambda: torch.ops._C_ascend.bgmv_shrink(
            x, weight, indices, bgmv_output, 1.0
        ),
        warmup,
        iterations,
        repeats,
    )
    result: dict[str, Any] = {
        "dtype": str(dtype).removeprefix("torch."),
        "m": m,
        "n": 16,
        "k": k,
        "groups": groups,
        "distribution": distribution,
        "accuracy": {
            "max_abs": float(difference.max().cpu()),
            "matched_ratio": float(matched.float().mean().cpu()),
        },
        "gmm": gmm_perf,
        "bgmv": bgmv_perf,
        "gmm_speedup_p50": bgmv_perf["p50_us"] / gmm_perf["p50_us"],
        "gmm_speedup_p95": bgmv_perf["p95_us"] / gmm_perf["p95_us"],
    }
    if check_graph:
        result["aclgraph"] = _aclgraph_check(x, weight, group_list)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--m", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--k", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument(
        "--dtype", nargs="+", choices=("float16", "bfloat16"), default=["float16", "bfloat16"]
    )
    parser.add_argument(
        "--distribution", nargs="+", choices=("hotspot", "fragmented"), default=["hotspot", "fragmented"]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    enable_custom_op()
    torch.npu.set_device(args.device)
    torch.manual_seed(17)
    cases = []
    for dtype_name in args.dtype:
        for m in args.m:
            for k in args.k:
                for distribution in args.distribution:
                    cases.append(
                        _run_case(
                            m,
                            k,
                            args.groups,
                            distribution,
                            getattr(torch, dtype_name),
                            args.warmup,
                            args.iterations,
                            args.repeats,
                            not args.skip_graph,
                        )
                    )
    print(json.dumps({"phase": 0, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
