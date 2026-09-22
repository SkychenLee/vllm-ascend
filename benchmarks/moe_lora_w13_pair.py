# SPDX-License-Identifier: Apache-2.0
"""Compare separate and paired w13 kernels (excluding TP communication).

ASCEND_RT_VISIBLE_DEVICES=8 python benchmarks/moe_lora_w13_pair.py --output pair.json
The expand comparison uses identical values in split vs rank-major TP layouts.
Use the TP MLP/service benchmark separately to measure communication savings.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def measure(fn, iterations, repeats):
    for _ in range(5):
        fn()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(samples), "samples_us": samples}


@torch.inference_mode()
def benchmark(args, rows):
    torch.manual_seed(87)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    groups, rank, hidden = args.groups, args.local_rank, args.hidden
    full_rank, width = args.tp_size * rank, args.width
    x = torch.randint(-128, 128, (rows, hidden), device="npu", dtype=torch.int8)
    scales = torch.rand(rows, device="npu") * 0.01
    ids = (torch.arange(rows, device="npu") % groups).long()
    inactive_rows = int(rows * args.inactive_fraction)
    ids[torch.randperm(rows, device="npu")[:inactive_rows]] = -1
    weights = torch.randn(2, groups, rank, hidden, device="npu", dtype=dtype) * 0.01
    out = torch.empty(rows, 2 * rank, device="npu")
    gate, up = torch.empty(rows, rank, device="npu"), torch.empty(rows, rank, device="npu")

    def shrink_separate():
        torch.ops._C_ascend.bgmv_shrink_int8(x, weights[0], ids, scales, gate)
        torch.ops._C_ascend.bgmv_shrink_int8(x, weights[1], ids, scales, up)

    def shrink_pair():
        torch.ops._C_ascend.bgmv_shrink_int8_pair(x, weights, ids, scales, out)

    shrink_separate()
    shrink_pair()
    torch.testing.assert_close(out[:, :rank], gate, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(out[:, rank:], up, atol=2e-5, rtol=2e-4)
    paired = torch.randn(args.tp_size, rows, 2 * rank, device="npu")
    g = paired[:, :, :rank].permute(1, 0, 2).reshape(rows, full_rank).contiguous()
    u = paired[:, :, rank:].permute(1, 0, 2).reshape(rows, full_rank).contiguous()
    base = torch.randn(rows, width * 2, device="npu", dtype=dtype)
    bg = torch.randn(groups, width, full_rank, device="npu", dtype=dtype) * 0.01
    bu = torch.randn_like(bg) * 0.01

    def expand_split_layout():
        return torch.ops._C_ascend.moe_lora_expand_swiglu_quant(base, g, u, bg, bu, ids, None, 10.0)

    def expand_rank_major():
        return torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair(base, paired, bg, bu, ids, None, 10.0)

    ref, refscale = expand_split_layout()
    q, scale = expand_rank_major()
    torch.testing.assert_close(q, ref, atol=0, rtol=0)
    torch.testing.assert_close(scale, refscale, atol=0, rtol=0)
    result = {k: v for k, v in vars(args).items() if k not in ("rows", "output", "profile_dir")}
    result["rows"] = rows
    result["inactive_rows"] = inactive_rows
    for name, fn in (
        ("shrink_separate", shrink_separate),
        ("shrink_pair", shrink_pair),
        ("expand_split_layout", expand_split_layout),
        ("expand_rank_major", expand_rank_major),
    ):
        result[name] = measure(fn, args.iterations, args.repeats)
        if args.profile_dir:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                ),
                schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.profile_dir / str(rows) / name)),
            ) as profiler:
                for _ in range(11):
                    fn()
                    torch.npu.synchronize()
                    profiler.step()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[6, 48, 384, 4096])
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--local-rank", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--inactive-fraction", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if (
        min(
            *args.rows,
            args.hidden,
            args.width,
            args.local_rank,
            args.tp_size,
            args.groups,
            args.iterations,
            args.repeats,
        )
        <= 0
    ):
        parser.error("dimensions and measurement counts must be positive")
    if args.local_rank * args.tp_size > 512 or args.width > 8192 or not 0 <= args.inactive_fraction <= 1:
        parser.error("unsupported rank, width or inactive fraction")
    if args.output.exists():
        parser.error("output exists; preserve prior measurements")
    assert enable_custom_op()
    results = []
    for rows in args.rows:
        results.append(benchmark(args, rows))
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
