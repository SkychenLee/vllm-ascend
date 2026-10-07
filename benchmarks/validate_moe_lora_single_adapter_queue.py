"""Bounded queue-ring, native-consumer and graph lifetime regression checks.

The device owner must run this with an external process timeout, for example
timeout 180s python validate_single_adapter_queue_lifetime.py --device 0 ... .
The owner must set ASCEND_RT_VISIBLE_DEVICES and TASK_QUEUE_ENABLE=1. This
script starts no service or collective group. It does not retain metadata
outputs or large scratch tensors. A RUNNING artifact is not a passing result.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path

TOP_K = 6
EXPERTS = 256
ADAPTERS = 3
RANK = 16
HIDDEN = 128
OUTPUT_WIDTH = 64


def run(args, report):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    torch.ops.load_library(str(args.library.resolve()))
    op = torch.ops._C_ascend.moe_lora_single_adapter_indices
    shrink = torch.ops._C_ascend.bgmv_shrink
    expand = torch.ops._C_ascend.bgmv_expand
    report["torch"] = torch.__version__
    report["torch_npu"] = torch_npu.__version__

    def save(phase):
        report["phase"] = phase
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"phase": phase, "status": report["status"]}), flush=True)

    ids = torch.arange(ADAPTERS * EXPERTS)
    weights_a = torch.zeros(ADAPTERS * EXPERTS, RANK, HIDDEN, dtype=torch.float16)
    weights_b = torch.zeros(ADAPTERS * EXPERTS, OUTPUT_WIDTH, RANK, dtype=torch.float16)
    for rank in range(RANK):
        weights_a[:, rank, rank] = (ids % 7 + 1).half()
        weights_b[:, rank, rank] = (ids % 5 + 1).half() / 2
    weights_a, weights_b = weights_a.to(device), weights_b.to(device)

    def make_case(tokens, index_dtype, expert_dtype, pattern, seed):
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(tokens * TOP_K, generator=generator).to(index_dtype)
        permutation[1::2].neg_()
        experts = torch.randint(EXPERTS, (tokens, TOP_K), generator=generator, dtype=expert_dtype)
        slots = torch.ones(tokens, dtype=torch.int64)
        mask = torch.tensor([1, -7, 1], dtype=torch.int32)
        if pattern == "mixed":
            slots.copy_(torch.arange(tokens) % ADAPTERS)
        elif pattern == "disabled":
            mask[1] = 0
        elif pattern == "negative-padding":
            slots[::3] = -1
        combined = torch.full((tokens * TOP_K,), -1, dtype=torch.int64)
        for original, destination in enumerate(permutation.tolist()):
            slot = int(slots[original // TOP_K])
            if slot >= 0 and int(mask[slot]) != 0:
                combined[abs(destination)] = slot * EXPERTS + int(experts.flatten()[original])
        rows = combined.numel()
        x_cpu = ((torch.arange(rows)[:, None] % 4 + torch.arange(HIDDEN)[None, :] % 8 + 1) / 8).half()
        expected = torch.zeros(rows, OUTPUT_WIDTH, dtype=torch.float16)
        for row, index in enumerate(combined.tolist()):
            if index >= 0:
                expected[row, :RANK] = x_cpu[row, :RANK] * (index % 7 + 1) * ((index % 5 + 1) / 2)
        sources = []
        for value in (permutation, experts, slots):
            holder = torch.full((2 * value.numel() + 32,), -73, dtype=value.dtype, device=device)
            view = holder[16 : 16 + 2 * value.numel() : 2].reshape(value.shape)
            view.copy_(value)
            assert not view.is_contiguous()
            sources.append(view)
        # Reproduce the real max_loras=3, enabled[4] context. Each native call
        # receives a fresh view of the first three entries, with no Python
        # reference retained to the contiguous routing copies or mask view.
        sources.append(torch.cat((mask, torch.zeros(1, dtype=mask.dtype))).to(device))
        return {
            "sources": sources,
            "cpu_inputs": (permutation, experts, slots, mask),
            "combined": combined,
            "x": x_cpu.to(device),
            "expected": expected,
            "rows": rows,
            "pattern": pattern,
            "index_dtype": str(index_dtype),
            "expert_dtype": str(expert_dtype),
        }

    def metadata(case):
        source = case["sources"]
        return op(source[0].contiguous(), source[1].contiguous(), source[2].contiguous(), source[3][:3], TOP_K, EXPERTS)

    def consume(case, combined, destination=None):
        # Existing native W13 shrink -> W2 expand consumers, sharing exactly
        # the same combined routing tensor. Both result temporaries are freed
        # after their consuming launches have been enqueued on this stream.
        shrink_out = torch.zeros(case["rows"], RANK, dtype=torch.float32, device=device)
        shrink(case["x"], weights_a, combined, shrink_out, 1.0)
        if destination is None:
            destination = torch.zeros(case["rows"], OUTPUT_WIDTH, dtype=torch.float16, device=device)
        else:
            destination.zero_()
        expand(shrink_out, weights_b, combined, destination, 0, OUTPUT_WIDTH)
        return destination

    def new_chain(case, destination=None):
        combined = metadata(case)
        result = consume(case, combined, destination)
        del combined
        return result

    def old_chain(case):
        cpu = case["cpu_inputs"]
        inputs = tuple(value.to(device) for value in cpu)
        experts, slots = torch.ops._C_ascend.moe_lora_recover(*inputs[:3], TOP_K)
        combined = torch.ops._C_ascend.moe_lora_combined_indices(experts, slots, inputs[3], EXPERTS)
        assert torch.equal(combined.cpu(), case["combined"])
        return consume(case, combined)

    dtypes = [
        (torch.int32, torch.int32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int64, torch.int64),
    ]
    patterns = ("single", "mixed", "disabled", "negative-padding")
    small_cases = [
        make_case(48, index_dtype, expert_dtype, pattern, 31 + seed)
        for seed, (index_dtype, expert_dtype) in enumerate(dtypes)
        for pattern in patterns
    ]
    large_cases = [make_case(1366, a, b, "mixed", 73 + seed) for seed, (a, b) in enumerate(dtypes)]
    save("consumer-baselines")
    for case in small_cases + large_cases:
        assert torch.equal(old_chain(case).cpu(), case["expected"])
        assert torch.equal(new_chain(case).cpu(), case["expected"])
    report["consumer_baselines"] = {
        "status": "PASS",
        "cases": len(small_cases) + len(large_cases),
        "consumer_ops": ["bgmv_shrink", "bgmv_expand"],
        "independent_reference": "FP16 exact dyadic one-hot weights; unchanged old native routing also checked",
    }

    save("queued-ring-pressure")
    stream = torch.npu.current_stream()
    event = torch.npu.Event(enable_timing=False)
    event.record(stream)
    event.synchronize()
    producer_event = torch.npu.Event(enable_timing=False)
    producer_event.record(stream)
    producer_event.synchronize()
    producer_stop = threading.Event()
    producer = {"record_calls": 0, "errors": []}

    def event_producer():
        try:
            torch.npu.set_device(args.device)
            while not producer_stop.wait(0.0005) and producer["record_calls"] < 2 * args.iterations:
                producer_event.record(stream)
                producer["record_calls"] += 1
        except Exception as error:
            producer["errors"].append(f"{type(error).__name__}: {error}")

    worker = threading.Thread(target=event_producer, name="single-adapter-event-producer", daemon=True)
    worker.start()
    samples = []
    large_calls = 0
    try:
        for index in range(args.iterations):
            large = (index + 1) % 1024 == 0
            case = large_cases[(index // 1024) % len(large_cases)] if large else small_cases[index % len(small_cases)]
            large_calls += int(large)
            output = new_chain(case)
            # Retain only sparse consumer-output snapshots, never native
            # combined outputs or scratch. Their clones are enqueued before
            # allocation churn; all other consumer outputs are dropped.
            if large or index % 257 == 0 or index >= args.iterations - len(small_cases):
                samples.append((index, case, output.clone()))
            del output
            torch.full((case["rows"],), index % 97, dtype=torch.int64, device=device)
            if large:
                torch.full((case["rows"], 8), -17, dtype=torch.int64, device=device)
        event.record(stream)
        event.synchronize()
    finally:
        producer_stop.set()
        worker.join(timeout=5)
    assert not worker.is_alive(), "Event.record producer did not stop; external process timeout is required"
    assert not producer["errors"] and producer["record_calls"] > 0, producer
    sample_checksums = []
    for index, case, snapshot in samples:
        actual = snapshot.cpu()
        assert torch.equal(actual, case["expected"]), f"consumer result changed after queue reuse at {index}"
        sample_checksums.append(
            {"index": index, "rows": case["rows"], "sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest()}
        )
    torch.npu.synchronize()
    report["queue_pressure"] = {
        "status": "PASS",
        "native_calls": args.iterations,
        "execution_ring_capacity_from_pinned_source": 4096,
        "small_rows": 288,
        "large_rows": 8196,
        "large_calls": large_calls,
        "discarded_native_outputs": args.iterations,
        "sampled_consumer_outputs": len(samples),
        "sampled_consumer_output_checksums": sample_checksums,
        "event_producer": producer,
        "scope": "Concurrent Python Event.record producer; real HCCL watchdog interaction requires whole-model TP test",
    }
    del samples
    save("drop-output-capture-replay")
    report["graphs"] = []
    for case in (small_cases[0], small_cases[12], large_cases[0], large_cases[3]):
        capture_stream = torch.npu.Stream(device=args.device)
        capture_stream.wait_stream(torch.npu.current_stream())
        final = torch.zeros(case["rows"], OUTPUT_WIDTH, dtype=torch.float16, device=device)
        with torch.npu.stream(capture_stream):
            for _ in range(3):
                new_chain(case, final)
        torch.npu.current_stream().wait_stream(capture_stream)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture_stream):
            for _ in range(args.capture_calls):
                new_chain(case, final)
        for replay in range(6):
            mask = case["sources"][3]
            mask[1] = 0 if replay % 2 else 1
            graph.replay()
            event.record(torch.npu.current_stream())
            event.synchronize()
            if replay % 2:
                expected = torch.zeros_like(case["expected"])
                if case["pattern"] == "mixed":
                    valid = (case["combined"] >= 0) & (case["combined"] // EXPERTS != 1)
                    expected[valid] = case["expected"][valid]
            else:
                expected = case["expected"]
            assert torch.equal(final.cpu(), expected), (
                f"drop-output replay mismatch at R={case['rows']} replay={replay}"
            )
            # Recorded graph allocations remain graph-owned after Python
            # metadata intermediates have gone. Same-stream churn must not
            # alter their contents or replay destinations.
            torch.full((case["rows"], 8), replay, dtype=torch.int64, device=device)
        mask[1] = -7
        report["graphs"].append(
            {
                "status": "PASS",
                "rows": case["rows"],
                "index_dtype": case["index_dtype"],
                "expert_dtype": case["expert_dtype"],
                "captured_calls": args.capture_calls,
                "replays": 6,
                "dropped_combined_and_shrink_outputs": True,
                "changed_mask_disable_reenable": True,
            }
        )
        del graph, final
    torch.npu.synchronize()
    save("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4105)
    parser.add_argument("--capture-calls", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 4096 < args.iterations <= 8192 or not 1 <= args.capture_calls <= 32:
        parser.error("iterations must be 4097..8192, capture-calls must be 1..32")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible or args.device < 0 or args.device >= len(visible.split(",")):
        parser.error("explicit visible devices and a valid logical --device are required")
    if os.environ.get("TASK_QUEUE_ENABLE") != "1":
        parser.error("TASK_QUEUE_ENABLE=1 is required to exercise the production execution queue")
    report = {
        "status": "RUNNING",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "logical_device": args.device,
        "ascend_visible_devices": visible,
        "task_queue_enable": os.environ["TASK_QUEUE_ENABLE"],
        "started_unix": time.time(),
        "whole_model_or_collective_validation": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        run(args, report)
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_unix"] = time.time()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "library_sha256", "script_sha256")}), flush=True)


if __name__ == "__main__":
    main()
