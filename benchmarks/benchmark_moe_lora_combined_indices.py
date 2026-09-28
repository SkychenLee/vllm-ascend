# SPDX-License-Identifier: Apache-2.0
"""NPU precision/changed-input graph/microbenchmark gate; never run on shared cards.

Example (after a device is explicitly available):
python benchmarks/benchmark_moe_lora_combined_indices.py --device 0 --library native.so --output results.json
Calls the native op directly; Python dispatch boundaries have separate CPU tests.
"""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def reference(experts, slots, mask, count):
    safe = slots.clamp(0, mask.numel() - 1)
    active = (slots >= 0) & (slots < mask.numel()) & mask[safe].bool()
    return torch.where(active, safe * count + experts.long(), -1)


def old_path(experts, slots, mask, count):
    safe = slots.clamp_min(0)
    enabled = (slots >= 0) & mask[safe].bool()
    return torch.where(enabled, safe * count + experts.long(), torch.full_like(slots, -1)).contiguous()


def timed(fn, iterations):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {"samples_us": samples, "median_us": statistics.median(samples)}


def capture(fn):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            fn()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = fn()
    return graph, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    op = torch.ops._C_ascend.moe_lora_combined_indices
    results = {
        "library": str(args.library.resolve()),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": args.device,
        "time": time.time(),
        "production_max_rows": 98304,
        "large_shapes": "direct native testing through target maximum",
        "correctness": [],
        "graphs": [],
        "benchmarks": [],
    }
    for dtype in (torch.int32, torch.int64):
        for rows in (0, 1, 7, 8, 9, 36, 288, 511, 512, 513, 49152, 98304):
            cpu_e = torch.arange(rows, dtype=dtype) % 256
            cpu_s = torch.arange(rows, dtype=torch.int64) % 7 - 2
            # Includes negative nonzero enabled values and positive invalid slots.
            cpu_m = torch.tensor([0, 1, -7, 0], dtype=torch.int32)
            e, s, m = [x.to("npu") for x in (cpu_e, cpu_s, cpu_m)]
            expected = reference(cpu_e, cpu_s, cpu_m, 256)
            assert torch.equal(op(e, s, m, 256).cpu(), expected)
            results["correctness"].append({"rows": rows, "dtype": str(dtype), "case": "mixed_valid_disabled_invalid"})
    for rows in (36, 288, 98304):
        cpu_e = torch.arange(rows, dtype=torch.int64) % 256
        for case in ("all_disabled", "all_negative_slots", "all_active_single_adapter"):
            cpu_s = torch.full((rows,), -1 if case == "all_negative_slots" else 0, dtype=torch.int64)
            cpu_m = torch.full((4,), 0 if case == "all_disabled" else 1, dtype=torch.int32)
            actual = op(cpu_e.npu(), cpu_s.npu(), cpu_m.npu(), 256)
            assert torch.equal(actual.cpu(), reference(cpu_e, cpu_s, cpu_m, 256))
            results["correctness"].append({"rows": rows, "case": case})
    cpu_e = torch.tensor([2**63 - 1, -(2**63), 2**60 + 3, -(2**60) + 5, 0], dtype=torch.int64)
    cpu_s = torch.tensor([1, 2, 3, 1, 2**63 - 1], dtype=torch.int64)
    cpu_m = torch.tensor([0, -1, 2, 1], dtype=torch.int32)
    for count in (256, 2**62 + 1):
        expected = reference(cpu_e, cpu_s, cpu_m, count)
        actual = op(cpu_e.npu(), cpu_s.npu(), cpu_m.npu(), count)
        assert torch.equal(actual.cpu(), expected)
        results["correctness"].append({"case": "int64_bits_and_wrap", "num_experts": count})
    for rows in (36, 288, 98304):
        e = torch.arange(rows, dtype=torch.int64).npu()
        s = (torch.arange(rows, dtype=torch.int64) % 4).npu()
        m = torch.tensor([1, 1, 0, -3], dtype=torch.int32).npu()
        graph, output = capture(partial(op, e, s, m, 256))
        for replay in range(4):
            cpu_e = (torch.arange(rows, dtype=torch.int64) + replay * 17) % 256
            cpu_s = (torch.arange(rows, dtype=torch.int64) + replay) % 6 - 1
            cpu_m = torch.tensor([replay % 2, 1, -2, 0], dtype=torch.int32)
            e.copy_(cpu_e)
            s.copy_(cpu_s)
            m.copy_(cpu_m)
            graph.replay()
            assert torch.equal(output.cpu(), reference(cpu_e, cpu_s, cpu_m, 256))
        results["graphs"].append({"rows": rows, "changed_input_replays": 4})
    for rows in (36, 288, 512, 49152, 98304):
        e = (torch.arange(rows, dtype=torch.int64) % 256).npu()
        s = (torch.arange(rows, dtype=torch.int64) % 5 - 1).npu()
        m = torch.tensor([1, 1, 0, -3], dtype=torch.int32).npu()
        record = {"rows": rows, "production_dispatch": rows <= 98304}
        for name, fn in (("old", partial(old_path, e, s, m, 256)), ("fused", partial(op, e, s, m, 256))):
            assert torch.equal(fn().cpu(), reference(e.cpu(), s.cpu(), m.cpu(), 256))
            record[name + "_eager"] = timed(fn, args.iterations)
            graph, output = capture(fn)
            graph.replay()
            assert torch.equal(output.cpu(), reference(e.cpu(), s.cpu(), m.cpu(), 256))
            record[name + "_graph"] = timed(graph.replay, args.iterations)
        results["benchmarks"].append(record)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
