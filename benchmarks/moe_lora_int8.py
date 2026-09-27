# SPDX-License-Identifier: Apache-2.0
"""Compare MoE LoRA shrink and B/add/clamp/SwiGLU/quant on one NPU.

Example (rows are routed rows, before dividing by TP):
    ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/moe_lora_int8.py --graph --output perf.json

This measures local kernels; TP communication, routing, base GMMs and service
latency must be measured separately. Expand timing uses zero low-rank inputs
to keep repeated in-place baseline calls numerically stable.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def measure(fn, graph_mode, repeats=5, iterations=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            fn()
        fn = graph.replay
    values = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(values), "samples_us": values}


@torch.inference_mode()
def benchmark(
    rows,
    graph_mode,
    profile_dir=None,
    *,
    hidden=4096,
    intermediate=256,
    local_rank=2,
    full_rank=16,
    groups=8,
    dtype=torch.bfloat16,
    inactive=False,
    inactive_fraction=0.0,
):
    indices = (torch.arange(rows, device="npu") % groups).long()
    if inactive:
        indices.fill_(-1)
    elif inactive_fraction:
        indices.masked_fill_(torch.arange(rows, device="npu") % 4 < int(4 * inactive_fraction), -1)
    x = torch.randn(rows, hidden, device="npu", dtype=dtype)
    q, scale = torch_npu.npu_dynamic_quant(x)
    a = torch.randn(groups, local_rank, hidden, device="npu", dtype=dtype)
    shrink = torch.empty(rows, local_rank, device="npu")
    base = torch.zeros(rows, 2 * intermediate, device="npu", dtype=dtype)
    b = torch.randn(groups, intermediate, full_rank, device="npu", dtype=dtype) * 0.01
    projected = torch.zeros(rows, full_rank, device="npu")
    down_x = torch.randn(rows, intermediate, device="npu", dtype=dtype)
    down_q, down_scale = torch_npu.npu_dynamic_quant(down_x)
    down_a = torch.randn(groups, full_rank, intermediate, device="npu", dtype=dtype)
    down_shrink = torch.empty(rows, full_rank, device="npu")

    def unfused():
        torch.ops._C_ascend.bgmv_expand(projected, b, indices, base, 0, intermediate)
        torch.ops._C_ascend.bgmv_expand(projected, b, indices, base, intermediate, intermediate)
        gate, up = base.chunk(2, -1)
        clipped = torch.cat((gate.clamp(max=10), up.clamp(-10, 10)), -1)
        return torch_npu.npu_dynamic_quant(torch_npu.npu_swiglu(clipped))

    def fused():
        return torch.ops._C_ascend.moe_lora_expand_swiglu_quant(base, projected, projected, b, b, indices, None, 10.0)

    result = {
        "routed_rows": rows,
        "graph": graph_mode,
        "hidden": hidden,
        "intermediate": intermediate,
        "local_rank": local_rank,
        "full_rank": full_rank,
        "groups": groups,
        "dtype": str(dtype),
        "inactive": inactive,
        "inactive_fraction": 1.0 if inactive else inactive_fraction,
    }
    for name, fn in (
        ("float_shrink", lambda: torch.ops._C_ascend.bgmv_shrink(x, a, indices, shrink, 1.0)),
        ("int8_shrink", lambda: torch.ops._C_ascend.bgmv_shrink_int8(q, a, indices, scale, shrink)),
        ("float_w2_shrink", lambda: torch.ops._C_ascend.bgmv_shrink(down_x, down_a, indices, down_shrink, 1.0)),
        (
            "int8_w2_shrink",
            lambda: torch.ops._C_ascend.bgmv_shrink_int8(down_q, down_a, indices, down_scale, down_shrink),
        ),
        ("unfused_expand_activation_quant", unfused),
        ("fused_expand_activation_quant", fused),
    ):
        result[name] = measure(fn, graph_mode)
    for name, fn in (("unfused", unfused), ("fused", fused)):
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        allocated = torch.npu.memory_allocated()
        output = fn()
        torch.npu.synchronize()
        result[f"{name}_peak_extra_bytes"] = torch.npu.max_memory_allocated() - allocated
        del output
        if profile_dir is not None:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir / str(rows) / name)),
            ) as profiler:
                for _ in range(10):
                    fn()
                    torch.npu.synchronize()
                    profiler.step()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[6, 48, 384, 4096])
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Sweep both dtypes, ranks, widths and inactive adapters.")
    parser.add_argument("--adapter-mix", action="store_true", help="Sweep 0/25/50/75/100 percent inactive rows.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, help="Also collect separate fused/unfused NPU traces.")
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows) or args.groups <= 0:
        parser.error("rows and groups must be positive")
    if args.sweep and args.adapter_mix:
        parser.error("choose either --sweep or --adapter-mix")
    if args.output.exists():
        parser.error("output already exists; preserve earlier measurements")
    assert enable_custom_op()
    torch.manual_seed(20260921)
    results = []
    cases = [(rows, {"groups": args.groups}) for rows in args.rows]
    if args.sweep:
        # Include rank/width boundaries and a generic (non-vectorized) width.
        cases = [
            (
                rows,
                {
                    "intermediate": width,
                    "full_rank": rank,
                    "local_rank": max(1, rank // 8),
                    "dtype": dtype,
                    "groups": args.groups,
                },
            )
            for dtype in (torch.bfloat16, torch.float16)
            for rows in (6, 48, 384, 4096)
            for width, rank in (
                (64, 8),
                (128, 16),
                (256, 16),
                (256, 32),
                (512, 16),
                (128, 64),
                (512, 32),
                (1024, 64),
                (256, 128),
            )
        ]
        cases += [(rows, {"inactive": True, "groups": args.groups}) for rows in (6, 48, 4096)]
    elif args.adapter_mix:
        cases = [
            (rows, {"inactive_fraction": fraction, "groups": args.groups})
            for rows in args.rows
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
    for rows, options in cases:
        results.append(benchmark(rows, args.graph, args.profile_dir, **options))
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
