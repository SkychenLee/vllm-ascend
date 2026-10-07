# SPDX-License-Identifier: Apache-2.0
"""Exact metadata/replay checks and the fully-sharded MoE routing microbenchmark.

The owner must reserve a device before running this script. --device is the
logical device within ASCEND_RT_VISIBLE_DEVICES, and has no default. No model
service is started. The script loads only the explicitly supplied native .so.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch_npu

TOP_K = 6
NUM_EXPERTS = 256
MAX_ROWS = 98304


def make_inputs(tokens, index_dtype, expert_dtype, pattern="single-0", seed=0, short_slots=False):
    generator = torch.Generator().manual_seed(seed)
    expanded = torch.randperm(tokens * TOP_K, generator=generator).to(index_dtype)
    expanded[seed % 2 :: 2].neg_()
    experts = torch.randint(0, NUM_EXPERTS, (tokens, TOP_K), generator=generator, dtype=expert_dtype)
    slots = torch.zeros(min(tokens, 3) if short_slots else tokens, dtype=torch.int64)
    enabled = torch.tensor([1, -7, 1], dtype=torch.int32)
    if pattern.startswith("single-"):
        slots.fill_(int(pattern[-1]))
    elif pattern == "negative-padding":
        slots.fill_(1)
        slots[1::3] = -1
    elif pattern == "mixed":
        slots.copy_(torch.arange(slots.numel(), dtype=torch.int64) % 3)
    elif pattern == "disabled":
        slots.fill_(1)
        enabled[1] = 0
    elif pattern == "all-negative":
        slots.fill_(-(2**63))
    elif pattern == "invalid-positive":
        slots.fill_(3)
        slots[1::2] = 2**63 - 1
    else:
        raise ValueError(pattern)
    return expanded, experts, slots, enabled


def integer_reference(inputs, count=NUM_EXPERTS):
    expanded, experts, slots, enabled = (value.tolist() for value in (inputs[0], inputs[1].flatten(), *inputs[2:]))
    result = [-1] * len(expanded)
    for original, destination in enumerate(expanded):
        slot = slots[min(original // TOP_K, len(slots) - 1)]
        if 0 <= slot < len(enabled) and enabled[slot] != 0:
            bits = (slot * count + experts[original]) & ((1 << 64) - 1)
            result[abs(destination)] = bits if bits < 1 << 63 else bits - (1 << 64)
    return torch.tensor(result, dtype=torch.int64)


def old_pipeline(inputs, count=NUM_EXPERTS, consumers=1):
    expanded, experts, slots, enabled = inputs
    recovered_experts, recovered_slots = torch.ops._C_ascend.moe_lora_recover(expanded, experts, slots, TOP_K)
    combined = None
    for _ in range(consumers):
        combined = torch.ops._C_ascend.moe_lora_combined_indices(recovered_experts, recovered_slots, enabled, count)
    return combined


def assert_output(output, inputs, cpu_inputs, count=NUM_EXPERTS):
    assert output.shape == inputs[0].shape and output.dtype == torch.int64
    assert output.device == inputs[0].device and output.is_contiguous()
    assert torch.equal(output.cpu(), integer_reference(cpu_inputs, count))
    assert torch.equal(output, old_pipeline(inputs, count))
    for tensor, original in zip(inputs, cpu_inputs):
        assert torch.equal(tensor.cpu(), original)
        assert not torch._C._is_alias_of(output, tensor)


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


def timed(fn, iterations, rounds):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    samples = []
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    for _ in range(rounds):
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def queued_contiguous_temporaries(op):
    """Exercise queued handlers after Python releases materialized inputs."""
    queued = []
    for seed in range(16):
        tokens = 48 if seed < 12 else 1366
        cpu_inputs = make_inputs(tokens, torch.int32, torch.int64, "mixed", seed)
        strided = []
        for source in cpu_inputs:
            holder = torch.full((source.numel() * 2 + 32,), -73, dtype=source.dtype, device="npu")
            view = holder[16 : 16 + source.numel() * 2 : 2].reshape(source.shape)
            view.copy_(source)
            assert not view.is_contiguous()
            strided.append(view)
        # Do not retain the contiguous copies in Python. Enqueue completes
        # before their release; current-stream allocator reuse and FIFO kernel
        # submission order subsequent writes after their metadata consumption.
        output = op(*(value.contiguous() for value in strided), TOP_K, NUM_EXPERTS)
        del strided, holder, view
        # Bounded allocator reuse pressure; no synchronize before validation.
        for source in cpu_inputs:
            torch.full(source.shape, seed + 37, dtype=source.dtype, device="npu")
        queued.append((output, cpu_inputs))
    for output, cpu_inputs in queued:
        assert torch.equal(output.cpu(), integer_reference(cpu_inputs))
        permanent_inputs = tuple(value.npu() for value in cpu_inputs)
        assert torch.equal(output, old_pipeline(permanent_inputs))
    return {
        "case": "queued-contiguous-input-temporaries",
        "queued_calls": len(queued),
        "rows": [288, 8196],
        "synchronized_only_for_validation": True,
    }


def meta_checks(op):
    results = []
    for tokens in (0, 6, 1366, 16384):
        values = make_inputs(tokens, torch.int32, torch.int64)
        values = tuple(torch.empty(value.shape, dtype=value.dtype, device="meta") for value in values)
        output = op(*values, TOP_K, NUM_EXPERTS)
        assert output.shape == values[0].shape and output.dtype == torch.int64 and output.device.type == "meta"
        results.append({"rows": tokens * TOP_K, "case": "valid-meta"})
    for invalid in (
        "top-k",
        "num-experts",
        "rows",
        "slots-empty",
        "mask-empty",
        "mask-too-long",
        "noncontiguous",
        "index-dtype",
        "expert-dtype",
        "slot-dtype",
        "mask-dtype",
        "middle",
        "over-limit",
    ):
        tokens = 147 if invalid == "middle" else 16385 if invalid == "over-limit" else 6
        values = [
            torch.empty(value.shape, dtype=value.dtype, device="meta")
            for value in make_inputs(tokens, torch.int32, torch.int64)
        ]
        top_k, count = TOP_K, NUM_EXPERTS
        if invalid == "top-k":
            top_k = 0
        elif invalid == "num-experts":
            count = 0
        elif invalid == "rows":
            values[0] = values[0][:-1]
        elif invalid == "slots-empty":
            values[2] = values[2][:0]
        elif invalid == "mask-empty":
            values[3] = values[3][:0]
        elif invalid == "mask-too-long":
            values[3] = torch.empty(65, dtype=torch.int32, device="meta")
        elif invalid == "noncontiguous":
            values[0] = torch.empty(72, dtype=torch.int32, device="meta")[::2]
        elif invalid.endswith("-dtype"):
            index = {"index-dtype": 0, "expert-dtype": 1, "slot-dtype": 2, "mask-dtype": 3}[invalid]
            values[index] = values[index].float()
        try:
            op(*values, top_k, count)
        except RuntimeError as error:
            assert "moe_lora_single_adapter_indices" in str(error)
        else:
            raise AssertionError(f"accepted invalid metadata: {invalid}")
        results.append({"case": "invalid-meta-" + invalid})
    return results


def run(args, results):
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    op = torch.ops._C_ascend.moe_lora_single_adapter_indices
    results["meta"] = meta_checks(op)
    tokens_to_test = (0, 6, 48, 1366, 16384) if args.smoke else (0, 1, 6, 8, 24, 48, 64, 146, 1366, 2731, 16384)
    patterns = (
        "single-0",
        "single-1",
        "single-2",
        "negative-padding",
        "mixed",
        "disabled",
        "all-negative",
        "invalid-positive",
    )
    for index_dtype in (torch.int32, torch.int64):
        for expert_dtype in (torch.int32, torch.int64):
            for tokens in tokens_to_test:
                for pattern in patterns:
                    cpu_inputs = make_inputs(tokens, index_dtype, expert_dtype, pattern)
                    inputs = tuple(value.npu() for value in cpu_inputs)
                    output = op(*inputs, TOP_K, NUM_EXPERTS)
                    assert_output(output, inputs, cpu_inputs)
                    results["correctness"].append(
                        {
                            "rows": tokens * TOP_K,
                            "index_dtype": str(index_dtype),
                            "expert_dtype": str(expert_dtype),
                            "case": pattern,
                        }
                    )
                if tokens:
                    cpu_inputs = make_inputs(tokens, index_dtype, expert_dtype, "mixed", short_slots=True)
                    inputs = tuple(value.npu() for value in cpu_inputs)
                    assert_output(op(*inputs, TOP_K, NUM_EXPERTS), inputs, cpu_inputs)
                    results["correctness"].append(
                        {
                            "rows": tokens * TOP_K,
                            "index_dtype": str(index_dtype),
                            "expert_dtype": str(expert_dtype),
                            "case": "short-slots",
                        }
                    )
    # Confirm signed INT32/INT64 payload bits and overflow semantics, independently.
    for index_dtype in (torch.int32, torch.int64):
        for expert_dtype in (torch.int32, torch.int64):
            limits = torch.iinfo(expert_dtype)
            large = 2 ** (limits.bits - 4)
            cpu_inputs = (
                torch.tensor([5, -2, 0, -3, 1, -4], dtype=index_dtype),
                torch.tensor([limits.max, limits.min, large + 3, -large + 5, -1, 0], dtype=expert_dtype).reshape(
                    1, TOP_K
                ),
                torch.tensor([2], dtype=torch.int64),
                torch.tensor([0, 1, -2], dtype=torch.int32),
            )
            inputs = tuple(value.npu() for value in cpu_inputs)
            for count in (NUM_EXPERTS, 2**62 + 1):
                assert_output(op(*inputs, TOP_K, count), inputs, cpu_inputs, count)
                results["correctness"].append(
                    {
                        "case": "integer-bits-and-wrap",
                        "index_dtype": str(index_dtype),
                        "expert_dtype": str(expert_dtype),
                        "num_experts": count,
                    }
                )
    # Check the real storage descriptor, even when the logical view is contiguous.
    inputs = list(value.npu() for value in make_inputs(6, torch.int32, torch.int32))
    # This TorchNPU version exposes a setter-only config property. Query the
    # registered raw option so even its unset b"" state is restored exactly.
    allow_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    assert isinstance(allow_internal_format, bytes), "cannot preserve the internal-format option state"
    try:
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": "enable"})
        inputs[1] = torch_npu.npu_format_cast(inputs[1], 29)
    finally:
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": allow_internal_format.decode()})
    assert torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == allow_internal_format
    assert torch_npu.get_npu_format(inputs[1]) == 29
    try:
        op(*inputs, TOP_K, NUM_EXPERTS)
    except RuntimeError as error:
        assert "moe_lora_single_adapter_indices" in str(error)
    else:
        raise AssertionError("native accepted FRACTAL_NZ routing metadata")
    results["correctness"].append({"case": "blocked-storage-rejected"})
    results["correctness"].append(queued_contiguous_temporaries(op))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Exact eager checks passed: {len(results['correctness'])}; Meta checks: {len(results['meta'])}", flush=True)

    graph_dtypes = (
        ((torch.int32, torch.int32), (torch.int64, torch.int64))
        if args.smoke
        else (
            (torch.int32, torch.int32),
            (torch.int32, torch.int64),
            (torch.int64, torch.int32),
            (torch.int64, torch.int64),
        )
    )
    for index_dtype, expert_dtype in graph_dtypes:
        for tokens in (6, 48, 1366, 16384):
            cpu_inputs = make_inputs(tokens, index_dtype, expert_dtype)
            inputs = tuple(value.npu() for value in cpu_inputs)
            graph, output = capture(partial(op, *inputs, TOP_K, NUM_EXPERTS))
            for seed, pattern in enumerate(
                ("single-1", "disabled", "single-2", "mixed", "negative-padding", "single-0"), 1
            ):
                current = make_inputs(tokens, index_dtype, expert_dtype, pattern, seed)
                for destination, source in zip(inputs, current):
                    destination.copy_(source)
                graph.replay()
                assert_output(output, inputs, current)
            results["graphs"].append(
                {
                    "rows": tokens * TOP_K,
                    "index_dtype": str(index_dtype),
                    "expert_dtype": str(expert_dtype),
                    "changed_input_replays": 6,
                }
            )
    print(f"Changed-input graphs passed: {len(results['graphs'])}", flush=True)
    if args.validation_only:
        return

    for tokens in (6, 8, 24, 48, 64, 1366, 16384):
        for pattern in ("single-1", "mixed"):
            cpu_inputs = make_inputs(tokens, torch.int32, torch.int32, pattern)
            inputs = tuple(value.npu() for value in cpu_inputs)
            record = {
                "rows": tokens * TOP_K,
                "tokens": tokens,
                "top_k": TOP_K,
                "num_experts": NUM_EXPERTS,
                "enabled_slots": 3,
                "mapping": pattern,
                "production_dispatch": True,
            }
            methods = (
                ("recover_plus_one_combined", partial(old_pipeline, inputs, consumers=1)),
                ("recover_plus_two_combined", partial(old_pipeline, inputs, consumers=2)),
                ("single_adapter", partial(op, *inputs, TOP_K, NUM_EXPERTS)),
            )
            for name, fn in methods:
                assert_output(fn(), inputs, cpu_inputs)
                record[name + "_eager"] = timed(fn, args.iterations, args.rounds)
                graph, output = capture(fn)
                graph.replay()
                assert_output(output, inputs, cpu_inputs)
                record[name + "_graph"] = timed(graph.replay, args.iterations, args.rounds)
            for mode in ("eager", "graph"):
                new_us = record["single_adapter_" + mode]["median_us"]
                for consumers in ("one", "two"):
                    old_us = record["recover_plus_" + consumers + "_combined_" + mode]["median_us"]
                    record["speedup_vs_" + consumers + "_combined_" + mode] = old_us / new_us
            results["benchmarks"].append(record)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(f"Metadata benchmark done: rows={tokens * TOP_K}, mapping={pattern}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="Fewer dtype/shape combinations, retaining target maximum")
    parser.add_argument("--validation-only", action="store_true", help="Skip timing after eager and replay validation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.iterations <= 0 or args.rounds <= 0:
        parser.error("--iterations and --rounds must be positive")
    results = {
        "status": "RUNNING",
        "library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "logical_device": args.device,
        "ascend_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE"),
        "started_at": time.time(),
        "production_max_rows": MAX_ROWS,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "smoke": args.smoke,
        "validation_only": args.validation_only,
        "correctness": [],
        "meta": [],
        "graphs": [],
        "benchmarks": [],
    }
    try:
        run(args, results)
        results["status"] = "PASS"
    except Exception as error:
        results["status"] = "FAIL"
        results["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        results["finished_at"] = time.time()
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: results[key]
                    for key in (
                        "status",
                        "logical_device",
                        "ascend_visible_devices",
                        "task_queue_enable",
                        "library_sha256",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
