# SPDX-License-Identifier: Apache-2.0
"""Compare one real-size routed W8A8 MoE/LoRA layer without model weights.

Run in an isolated process with ASCEND_RT_VISIBLE_DEVICES set to an available
physical device. Synthetic weights are nonzero. This measures local compute;
there is no attention, TP/EP collective, scheduler, or full-model throughput.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    torch.npu.current_stream().wait_stream(stream)
    return graph, output


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


def pair_metadata(tokens, phase, local_experts=32, total_experts=256, topk=6):
    ids = ((torch.arange(tokens * topk) * 37 + phase * 17) % total_experts).reshape(tokens, topk).int()
    slots = (torch.arange(tokens).long() + phase) % 5 - 1
    pairs = sorted((int(ids[t, k]), t, k) for t in range(tokens) for k in range(topk) if ids[t, k] < local_experts)
    expanded = torch.full((tokens * topk,), -1, dtype=torch.int32)
    counts = torch.zeros(local_experts, dtype=torch.int64)
    for row, (expert, token, k) in enumerate(pairs):
        expanded[token * topk + k] = row
        counts[expert] += 1
    return ids, expanded, slots, counts, len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, action="append", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--limit", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.npu.set_device(args.device)
    for library in args.library:
        torch.ops.load_library(str(library.resolve()))

    from vllm_ascend import envs
    from vllm_ascend.utils import adapt_patch, register_ascend_customop

    # Match worker bootstrap order; importing device_op first creates an ops
    # package cycle before the platform has registered its implementations.
    adapt_patch()
    adapt_patch(True)
    register_ascend_customop()

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.lora import lora_ops, quant_moe, routing
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
    from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput, MoEQuantParams, MoEWeights
    from vllm_ascend.quantization.quant_type import QuantType

    baseline_quant = load_module(args.baseline_dir / "quant_moe.baseline.py", "decode_baseline_quant")
    baseline_routing = load_module(args.baseline_dir / "routing.baseline.py", "decode_baseline_routing")
    _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
    torch.manual_seed(20261006)
    device = torch.device(f"npu:{args.device}")
    hidden, intermediate, rank, experts, adapters = 4096, 2048, 16, 32, 4
    weights = MoEWeights(
        w1=[torch.randint(-3, 4, (experts, hidden, intermediate * 2), dtype=torch.int8, device=device)],
        w2=[torch.randint(-3, 4, (experts, intermediate, hidden), dtype=torch.int8, device=device)],
        w1_scale=[torch.full((experts, intermediate * 2), 0.02, dtype=torch.bfloat16, device=device)],
        w2_scale=[torch.full((experts, hidden), 0.02, dtype=torch.bfloat16, device=device)],
    )
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.bgmv_shrink = lora_ops.bgmv_shrink
    wrapper.bgmv_expand_slice = lora_ops.bgmv_expand_slice
    wrapper.is_prefill = False
    context = SimpleNamespace(
        use_ep=True,
        fully_sharded=False,
        punica_wrapper=wrapper,
        adapter_enabled=torch.tensor([1, 0, 1, 1], dtype=torch.int32, device=device),
        w13_lora_a_stacked=[
            (torch.randn(adapters, experts, rank, hidden, device=device) * 0.01).bfloat16() for _ in range(2)
        ],
        w13_lora_b_stacked=[
            (torch.randn(adapters, experts, intermediate, rank, device=device) * 0.02).bfloat16() for _ in range(2)
        ],
        w2_lora_a_stacked=[(torch.randn(adapters, experts, rank, intermediate, device=device) * 0.01).bfloat16()],
        w2_lora_b_stacked=[(torch.randn(adapters, experts, hidden, rank, device=device) * 0.02).bfloat16()],
    )
    expert_map = torch.full((256,), -1, dtype=torch.int32, device=device)
    expert_map[:experts] = torch.arange(experts, dtype=torch.int32, device=device)
    results = {
        "scope": "one routed MoE layer; no model/attention/communication/scheduler",
        "physical_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "local_device": args.device,
        "libraries": [str(path.resolve()) for path in args.library],
        "library_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.library},
        "shape": {"H": hidden, "I": intermediate, "rank": rank, "experts": experts, "adapters": adapters, "topk": 6},
        "nonzero_synthetic_lora": True,
        "routing_distribution": "deterministic global expert selections; rank0 owns 32/256 experts",
        "swiglu_limit": args.limit,
        "flags": {
            "fused_routing": envs.VLLM_ASCEND_MOE_LORA_FUSED_ROUTING,
            "fused_activation": getattr(envs, "VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION", False),
        },
        "cases": [],
    }
    for tokens in args.tokens:
        context.adapter_enabled.copy_(torch.tensor([1, 0, 1, 1], dtype=torch.int32))
        cpu_ids, cpu_expanded, cpu_slots, cpu_counts, active_rows = pair_metadata(tokens, 0)
        ids, expanded, slots, counts = [x.to(device) for x in (cpu_ids, cpu_expanded, cpu_slots, cpu_counts)]
        rows = tokens * 6
        x = (torch.randn(rows, hidden, device=device) * 2).bfloat16()

        def chain(
            quant_module, routing_module, *, expanded=expanded, ids=ids, slots=slots, rows=rows, x=x, counts=counts
        ):
            metadata = routing_module.recover_allgather_ep_lora_routing(
                expanded_row_idx=expanded,
                topk_ids=ids,
                prepared_lora_indices=slots,
                expert_map=expert_map,
                num_dispatched_rows=rows,
            )
            routed_x = x.masked_fill((metadata.expert_ids < 0).unsqueeze(-1), 0)
            payload = MoEMlpComputeInput(
                hidden_states=routed_x,
                group_list=counts,
                group_list_type=1,
                dynamic_scale=None,
                topk_scales=None,
                fusion=True,
                weights=weights,
                quant=MoEQuantParams(quant_type=QuantType.W8A8),
                lora_context=context,
                lora_routing=metadata,
                activation="silu",
                swiglu_limit=args.limit,
            )
            output, _ = quant_module.quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
            return output.masked_fill((metadata.expert_ids < 0).unsqueeze(-1), 0)

        baseline_fn = lambda: chain(baseline_quant, baseline_routing)
        optimized_fn = lambda: chain(quant_moe, routing)
        baseline_output = baseline_fn().cpu()
        activation_dispatch = []
        original_activation_guard = quant_moe.can_fuse_moe_lora_activation

        def inspect_activation(
            tensor,
            payload,
            comm_type,
            *,
            original_activation_guard=original_activation_guard,
            activation_dispatch=activation_dispatch,
        ):
            selected = original_activation_guard(tensor, payload, comm_type)
            activation_dispatch.append(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "storage_format": torch_npu.get_npu_format(tensor),
                    "fused": selected,
                }
            )
            return selected

        # Inspect one untimed call, then restore the production guard before
        # capture/timing. This proves the native path was selected at runtime.
        quant_moe.can_fuse_moe_lora_activation = inspect_activation
        try:
            optimized_output = optimized_fn().cpu()
        finally:
            quant_moe.can_fuse_moe_lora_activation = original_activation_guard
        difference = (baseline_output.float() - optimized_output.float()).abs()
        torch.testing.assert_close(optimized_output, baseline_output, rtol=0, atol=0)
        case = {
            "tokens": tokens,
            "capacity": rows,
            "active_rows": active_rows,
            "initial_max_abs": difference.max().item(),
            "activation_dispatch": activation_dispatch,
        }
        graphs = {}
        outputs = {}
        for name, fn in (("baseline", baseline_fn), ("optimized", optimized_fn)):
            case[name + "_eager"] = timed(fn, args.iterations)
            graphs[name], outputs[name] = capture(fn)
            case[name + "_graph"] = timed(graphs[name].replay, args.iterations)
        case["changed_input_replays"] = []
        for phase in (1, 2, 3):
            new_ids, new_expanded, new_slots, new_counts, active_rows = pair_metadata(tokens, phase)
            for target, source in zip((ids, expanded, slots, counts), (new_ids, new_expanded, new_slots, new_counts)):
                target.copy_(source)
            x.copy_((torch.randn(rows, hidden, device=device) * (phase + 1)).bfloat16())
            context.adapter_enabled.copy_(torch.tensor([phase % 2, 1, 0, 1], dtype=torch.int32))
            expected = baseline_fn().cpu()
            graphs["baseline"].replay()
            graphs["optimized"].replay()
            baseline_actual, actual = outputs["baseline"].cpu(), outputs["optimized"].cpu()
            torch.testing.assert_close(baseline_actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            case["changed_input_replays"].append({"phase": phase, "active_rows": active_rows, "max_abs": 0.0})
        case["graph_speedup"] = case["baseline_graph"]["median_us"] / case["optimized_graph"]["median_us"]
        results["cases"].append(case)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(case), flush=True)
        del graphs, outputs
    print("PASS", flush=True)


if __name__ == "__main__":
    from vllm.forward_context import ForwardContext, override_forward_context

    with override_forward_context(ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})):
        main()
