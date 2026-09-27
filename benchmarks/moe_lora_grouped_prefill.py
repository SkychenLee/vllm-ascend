# SPDX-License-Identifier: Apache-2.0
"""Alternate legacy/grouped local W8A8 LoRA timings, including independent sort.

Example:
  ASCEND_RT_VISIBLE_DEVICES=1 python benchmarks/moe_lora_grouped_prefill.py \
      --rows 48 2053 49152 --output /tmp/grouped_prefill_event.json

These are local-kernel measurements. Synthetic rank-major inputs replace TP
communication; base GMM, token dispatch/combine and serving latency are excluded.
Each w13 call includes routing, A, inverse permutation, fused B/activation/quant,
and output inverse permutation. w2 includes legacy A and grouped B.
The shared-routing variant builds the route once for the combined w13+w2 call.
Small-row cases deliberately force the candidate kernels to characterize the
cutoff; the production dispatcher keeps those cases on the legacy path.
The historical mixed mode uses about 9% disabled rows. mixed50 and invalid90
stress inactive tails; base_expert layout models the base dispatch row order.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.lora.grouped_prefill import prepare_grouped_moe_lora_routing
from vllm_ascend.utils import enable_custom_op


def alternating_measure(functions, *, graph_mode, repeats, iterations):
    calls, graphs = {}, []
    for name, function in functions.items():
        for _ in range(5):
            function()
        if graph_mode:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                function()
            graphs.append(graph)
            calls[name] = graph.replay
        else:
            calls[name] = function
        for _ in range(5):
            calls[name]()
    torch.npu.synchronize()
    samples = {name: [] for name in calls}
    orders = []
    for repeat in range(repeats):
        order = list(calls)
        if repeat % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            for _ in range(iterations):
                calls[name]()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / iterations)
    return {
        "order": orders,
        "measurements": {
            name: {
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
                "samples_us": values,
            }
            for name, values in samples.items()
        },
    }


@torch.inference_mode()
def benchmark(args, rows, adapter_mode):
    torch.manual_seed(20260926)
    hidden, width, experts = (4096, 256, 256) if args.shape == "deepseek" else (2048, 96, 128)
    local_rank, full_rank, shards = 2, 16, 8
    adapters, groups = 3, 3 * experts
    row = torch.arange(rows, device="npu")
    expert_ids = row % max(1, experts - 1)
    if adapter_mode in ("mixed50", "invalid90"):
        expert_ids = row % experts
        token = row // (6 if args.shape == "deepseek" else 8)
        if adapter_mode == "mixed50":
            slots = token % 4 - 1
            active = (slots == 0) | (slots == 2)
        else:
            slots = token // 10 % 2 * 2
            active = token % 10 == 0
        ids = torch.where(active, slots * experts + expert_ids, -1)
    else:
        ids = expert_ids.clone()
        if adapter_mode == "single":
            ids += (adapters - 1) * experts
        else:
            ids += (row // experts % adapters) * experts
        ids[::11] = -1
    order = torch.argsort(expert_ids) if args.routing_layout == "base_expert" else torch.randperm(rows, device="npu")
    ids = ids.index_select(0, order).contiguous()
    q = torch.randint(-128, 128, (rows, hidden), device="npu", dtype=torch.int8)
    scales = torch.rand(rows, device="npu") * 0.0173
    weights = torch.randn(2, groups, local_rank, hidden, device="npu", dtype=torch.bfloat16) * 0.01
    base = torch.randn(rows, width * 2, device="npu", dtype=torch.bfloat16)
    bg = torch.randn(groups, width, full_rank, device="npu", dtype=torch.bfloat16) * 0.01
    bu = torch.randn_like(bg) * 0.02
    pair = torch.randn(shards, rows, 2 * local_rank, device="npu") * 0.01
    local = torch.empty(rows, 2 * local_rank, device="npu")
    down_q = torch.randint(-128, 128, (rows, width), device="npu", dtype=torch.int8)
    down_scale = torch.rand(rows, device="npu") * 0.0187
    down_a = torch.randn(groups, full_rank, width, device="npu", dtype=torch.bfloat16) * 0.01
    # Zero B yields a zero delta and keeps repeated in-place timing stable.
    # Kernel dimensions and routing are unchanged; correctness uses nonzero B
    # in the NPU test suite and the full MLP comparison.
    down_b = torch.zeros(groups, hidden // shards, full_rank, device="npu", dtype=torch.bfloat16)
    down_base = torch.randn(rows, hidden, device="npu", dtype=torch.bfloat16)
    down_local = torch.empty(rows, full_rank, device="npu")

    def legacy_w13():
        torch.ops._C_ascend.bgmv_shrink_int8_pair(q, weights, ids, scales, local)
        pair[0].copy_(local)
        return torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(base, pair, bg, bu, ids, None, 10.0)

    def grouped_w13(route=None):
        if route is None:
            route = prepare_grouped_moe_lora_routing(ids, groups)
        torch.ops._C_ascend.bgmv_shrink_int8_pair_grouped(q, weights, route.sorted_indices, route.order, scales, local)
        pair[0].copy_(local.index_select(0, route.inverse))
        out, scale = torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair_grouped(
            base, pair, bg, bu, route.sorted_indices, route.order, None, 10.0
        )
        return out.index_select(0, route.inverse), scale.index_select(0, route.inverse)

    def legacy_w2():
        torch.ops._C_ascend.bgmv_shrink_int8(down_q, down_a, ids, down_scale, down_local)
        torch.ops._C_ascend.bgmv_expand(down_local, down_b, ids, down_base, 0, hidden // shards)

    def grouped_w2(route=None):
        if route is None:
            route = prepare_grouped_moe_lora_routing(ids, groups)
        # The grouped A experiment loses against the existing short-K kernel;
        # production retains legacy A and only groups this stage's B.
        torch.ops._C_ascend.bgmv_shrink_int8(down_q, down_a, ids, down_scale, down_local)
        projected = down_local
        torch.ops._C_ascend.bgmv_expand_grouped(projected, down_b, route.sorted_indices, route.order, down_base, 0)

    def legacy_both():
        legacy_w13()
        legacy_w2()

    def grouped_both():
        route = prepare_grouped_moe_lora_routing(ids, groups)
        grouped_w13(route)
        grouped_w2(route)

    expected_q, expected_scale = legacy_w13()
    actual_q, actual_scale = grouped_w13()
    torch.testing.assert_close(actual_q.cpu().float(), expected_q.cpu().float(), atol=2, rtol=0)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale.cpu(), atol=1e-5, rtol=8e-3)
    result = {
        "rows": rows,
        "hidden": hidden,
        "intermediate_per_partition": width,
        "experts": experts,
        "adapters": adapters,
        "adapter_mode": adapter_mode,
        "active_fraction": (ids >= 0).float().mean().item(),
        "routing_layout": args.routing_layout,
        "local_rank": local_rank,
        "full_rank": full_rank,
        "graph": args.graph,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "timing_includes_independent_grouping": True,
        "timing_includes_row_restoration": True,
        "real_tp_communication": False,
        "force_grouped_kernels": True,
        "w2_uses_zero_b_for_stable_repetition": True,
        "w2_a_backend": "legacy_int8",
    }
    for name, old, new in (
        ("w13", legacy_w13, grouped_w13),
        ("w2", legacy_w2, grouped_w2),
        ("combined_shared_route", legacy_both, grouped_both),
    ):
        result[name] = alternating_measure(
            {"legacy": old, "grouped": new}, graph_mode=args.graph, repeats=args.repeats, iterations=args.iterations
        )
    result["routing_only"] = alternating_measure(
        {"grouping": lambda: prepare_grouped_moe_lora_routing(ids, groups)},
        graph_mode=args.graph,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[48, 2053, 49152])
    parser.add_argument("--shape", choices=["deepseek", "qwen"], default="deepseek")
    parser.add_argument(
        "--adapters", choices=["single", "mixed", "mixed50", "invalid90"], nargs="+", default=["single", "mixed"]
    )
    parser.add_argument("--routing-layout", choices=["random", "base_expert"], default="random")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.rows, args.iterations) <= 0 or args.repeats < 5:
        parser.error("positive rows/iterations and at least five alternating repeats are required")
    if args.output.exists():
        parser.error("output exists; preserve earlier measurements")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert enable_custom_op()
    results = []
    for rows in args.rows:
        for mode in args.adapters:
            results.append(benchmark(args, rows, mode))
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results[-1]), flush=True)
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
