# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time

import torch
import torch_npu


def _gmm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    return torch_npu.npu_grouped_matmul(
        x=[inputs],
        weight=[weight],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=1,
    )[0]


def _benchmark(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=7168)
    parser.add_argument("--output-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device("npu:0")
    inputs = torch.randn(
        args.rows,
        args.input_size,
        dtype=torch.bfloat16,
        device=device,
    )
    bgmv_weight = torch.randn(
        args.experts,
        args.output_size,
        args.input_size,
        dtype=torch.bfloat16,
        device=device,
    )
    transposed_view = bgmv_weight.transpose(-1, -2)
    transposed_contiguous = transposed_view.contiguous()
    counts = torch.zeros(args.experts, dtype=torch.int64, device=device)
    counts[: args.rows % args.experts] += 1
    counts += args.rows // args.experts

    eager_view = _gmm(inputs, transposed_view, counts)
    eager_contiguous = _gmm(inputs, transposed_contiguous, counts)
    torch.npu.synchronize()
    torch.testing.assert_close(eager_view.cpu(), eager_contiguous.cpu(), rtol=0, atol=0)

    view_us = _benchmark(
        lambda: _gmm(inputs, transposed_view, counts),
        args.warmup,
        args.iterations,
    )
    contiguous_us = _benchmark(
        lambda: _gmm(inputs, transposed_contiguous, counts),
        args.warmup,
        args.iterations,
    )

    view_graph = torch.npu.NPUGraph()
    with torch.npu.graph(view_graph):
        view_graph_output = _gmm(inputs, transposed_view, counts)
    contiguous_graph = torch.npu.NPUGraph()
    with torch.npu.graph(contiguous_graph):
        contiguous_graph_output = _gmm(inputs, transposed_contiguous, counts)
    view_graph.replay()
    contiguous_graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(
        view_graph_output.cpu(),
        contiguous_graph_output.cpu(),
        rtol=0,
        atol=0,
    )

    view_graph_us = _benchmark(view_graph.replay, args.warmup, args.iterations)
    contiguous_graph_us = _benchmark(
        contiguous_graph.replay,
        args.warmup,
        args.iterations,
    )
    print(f"rows={args.rows},experts={args.experts},K={args.input_size},N={args.output_size}")
    print(f"eager_view_us={view_us:.3f}")
    print(f"eager_contiguous_us={contiguous_us:.3f}")
    print(f"graph_view_us={view_graph_us:.3f}")
    print(f"graph_contiguous_us={contiguous_graph_us:.3f}")
    print("bitwise_equal=true")


if __name__ == "__main__":
    main()
