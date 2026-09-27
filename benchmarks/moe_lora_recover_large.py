# SPDX-License-Identifier: Apache-2.0
"""Measure the complete large routing recovery with eager and graph NPU Events.

Example:
  ASCEND_RT_VISIBLE_DEVICES=8 python benchmarks/moe_lora_recover_large.py \
      --tokens 8192 --top-k 6 8 --output /tmp/recover_large.json

Every call includes output allocation, inverse construction, integer payload
gathers, dtype conversion and the token floor/clamp. Producer dispatch is
excluded. The Python sort reference has the established C++ fallback's tensor
operations but a different host launch path: its timing is explicitly labelled
and must not be presented as a same-binding old/new speedup. A diagnostic C++
copy of the original fallback can be supplied with --baseline-library and
--baseline-op namespace::operator for matched native launch paths. No expert
grouping is reused: recovery only reconstructs the payload per row.
Outside the supplied large-row interval, the native baseline uses the identical
production dispatch so existing small/midsize kernels are not counted as gains.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import vllm_ascend
from vllm_ascend.utils import enable_custom_op


def tensor_inputs(tokens, top_k, layout, index_dtype, expert_dtype):
    rows = tokens * top_k
    generator = torch.Generator().manual_seed(20260926)
    experts = torch.randint(0, 256, (tokens, top_k), generator=generator, dtype=expert_dtype)
    if layout == "random":
        expanded = torch.randperm(rows, generator=generator).to(index_dtype)
    elif layout == "base_expert":
        original_rows = experts.reshape(-1).argsort(stable=True)
        expanded = torch.empty(rows, dtype=index_dtype)
        expanded[original_rows] = torch.arange(rows, dtype=index_dtype)
    else:
        owners = 48
        chunk = (rows + owners - 1) // owners
        sources = [owner * chunk + offset for offset in range(chunk) for owner in range(owners)]
        sources = [source for source in sources if source < rows]
        expanded = torch.empty(rows, dtype=index_dtype)
        expanded[torch.tensor(sources)] = torch.arange(rows, dtype=index_dtype)
    expanded[::3].neg_()
    slots = torch.arange(tokens, dtype=torch.int64).remainder(4) - 1
    return expanded, experts, slots


def sort_reference(expanded, experts, slots, top_k):
    inverse = expanded.abs().float().argsort()
    token = torch.div(inverse, top_k, rounding_mode="floor")
    token.clamp_max_(slots.numel() - 1)
    return experts.reshape(-1).index_select(0, inverse).long(), slots.index_select(0, token)


def measure(functions, expected, graph_mode, repeats, iterations):
    calls, graphs, captured_outputs = {}, [], {}

    def check_replay(name):
        calls[name]()
        for actual, golden in zip(captured_outputs[name], expected):
            assert actual.dtype == torch.int64 and actual.is_contiguous()
            assert torch.equal(actual.cpu(), golden), f"graph replay mismatch: {name}"

    for name, function in functions.items():
        for _ in range(5):
            function()
        if graph_mode:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured_outputs[name] = function()
            graphs.append(graph)
            calls[name] = graph.replay
        else:
            calls[name] = function
        for _ in range(5):
            calls[name]()
        if graph_mode:
            check_replay(name)
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
    if graph_mode:
        # Keep graph output tensors alive for all replays and check the actual
        # captured buffers. Eager correctness alone does not validate a graph.
        # These replays and host copies are outside every measured interval.
        for name in calls:
            check_replay(name)
    return {
        "mode": "graph" if graph_mode else "eager",
        "graph_outputs_exact_before_and_after_timing": graph_mode,
        "orders": orders,
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
def run_case(args, tokens, top_k, layout):
    dtypes = {"int32": torch.int32, "int64": torch.int64}
    cpu_inputs = tensor_inputs(tokens, top_k, layout, dtypes[args.index_dtype], dtypes[args.expert_dtype])
    inputs = tuple(value.npu() for value in cpu_inputs)

    def native():
        return torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)

    def framework():
        return sort_reference(*inputs, top_k)

    functions = {"native": native}
    changed_shape = args.large_min_rows <= tokens * top_k <= args.large_max_rows
    if args.baseline_op:
        namespace, operator = args.baseline_op.split("::")
        baseline = getattr(getattr(torch.ops, namespace), operator)
        # Small/midsize production already has native kernels. Its existing
        # optimization must not be credited to the new large-row path.
        functions["native_original_dispatch"] = (lambda: baseline(*inputs, top_k)) if changed_shape else native
    if not args.skip_python_reference:
        functions["python_sort_reference"] = framework
    expected_experts, expected_slots = [None] * (tokens * top_k), [None] * (tokens * top_k)
    source_experts = cpu_inputs[1].reshape(-1).tolist()
    source_slots = cpu_inputs[2].tolist()
    for source, destination in enumerate(cpu_inputs[0].abs().tolist()):
        expected_experts[destination] = source_experts[source]
        expected_slots[destination] = source_slots[source // top_k]
    expected = (torch.tensor(expected_experts), torch.tensor(expected_slots))
    for function in functions.values():
        for actual, golden in zip(function(), expected):
            assert actual.dtype == torch.int64 and actual.is_contiguous()
            assert torch.equal(actual.cpu(), golden)
    result = {
        "tokens": tokens,
        "top_k": top_k,
        "rows": tokens * top_k,
        "routing_layout": layout,
        "index_dtype": args.index_dtype,
        "expert_dtype": args.expert_dtype,
        "slot_dtype": "int64",
        "exact_integer_oracle_passed": True,
        "timing_includes_inverse_allocation_cast_gather_floor_clamp": True,
        "timing_excludes_producer_dispatch": True,
        "matched_native_launch_paths": bool(args.baseline_op),
        "baseline_op": args.baseline_op,
        "large_recovery_interval": [args.large_min_rows, args.large_max_rows],
        "same_native_recovery_for_both_variants": bool(args.baseline_op) and not changed_shape,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "native_label": args.native_label,
    }
    for mode in args.modes:
        result[mode] = measure(functions, expected, mode == "graph", args.repeats, args.iterations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 8192])
    parser.add_argument("--top-k", type=int, nargs="+", default=[6, 8])
    parser.add_argument(
        "--layouts", choices=["random", "base_expert", "interleaved"], nargs="+", default=["base_expert"]
    )
    parser.add_argument("--modes", choices=["eager", "graph"], nargs="+", default=["eager", "graph"])
    parser.add_argument("--index-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--expert-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--native-label", default="current_snapshot")
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--baseline-op", help="Diagnostic native baseline schema, as namespace::operator")
    parser.add_argument("--skip-python-reference", action="store_true")
    parser.add_argument("--large-min-rows", type=int, default=8161)
    parser.add_argument("--large-max-rows", type=int, default=262144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.tokens, *args.top_k, args.iterations) <= 0 or args.repeats < 5:
        parser.error("positive dimensions/iterations and at least five alternating rounds are required")
    if max(args.tokens) * max(args.top_k) > 1 << 24:
        parser.error("complete-permutation contract requires rows <= 2**24")
    if args.output.exists():
        parser.error("output exists; preserve previous measurements")
    if bool(args.baseline_library) != bool(args.baseline_op):
        parser.error("--baseline-library and --baseline-op must be supplied together")
    if args.baseline_op and len(args.baseline_op.split("::")) != 2:
        parser.error("--baseline-op must use namespace::operator")
    if args.skip_python_reference and not args.baseline_op:
        parser.error("--skip-python-reference requires a native baseline")
    if not 0 < args.large_min_rows <= args.large_max_rows:
        parser.error("invalid large recovery row interval")
    torch.set_num_threads(1)
    assert enable_custom_op()
    if args.baseline_library:
        torch.ops.load_library(str(args.baseline_library.resolve()))
    native_paths = sorted(Path(vllm_ascend.__file__).parent.glob("*.so"))
    if args.baseline_library:
        native_paths.append(args.baseline_library.resolve())
    loaded_paths = {
        Path(line.split()[-1])
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "vllm_ascend" in line and ".so" in line
    }
    if args.baseline_library:
        loaded_paths.add(args.baseline_library.resolve())
    report = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in native_paths},
        "loaded_libraries_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(loaded_paths)
        },
        "device": torch.npu.get_device_name(),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        for top_k in args.top_k:
            for layout in args.layouts:
                case = run_case(args, tokens, top_k, layout)
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
