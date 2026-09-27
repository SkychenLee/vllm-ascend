# SPDX-License-Identifier: Apache-2.0
"""TP MoE MLP correctness and alternating grouped/legacy Event measurements.

Run under torchrun with TP2 or TP8. Includes real HCCL low-rank collectives,
routing recovery, the independent adapter/expert grouping and both base GMMs.
Initial token dispatch and final token combine/output AllReduce are excluded.
This is a synthetic layer benchmark, not a TTFT/TPOT service measurement.

  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 \
      --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29667 \
      benchmarks/moe_lora_grouped_prefill_tp.py \
      --tokens 8 1024 8192 --output-dir /tmp/grouped_tp8

The benchmark respects inherited CPU affinity and does not assume device8-15
NUMA placement. Rank files retain every alternating measurement round.
"""

import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu

# Register the MoE package before importing Punica: the plugin's registration
# imports the LoRA context, so reversing this order creates a module cycle.
import vllm_ascend.ops  # noqa: F401  # isort: skip

import vllm_ascend.lora.grouped_prefill as grouped
import vllm_ascend.lora.punica_npu as punica
import vllm_ascend.lora.quant_moe as quant
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.lora.lora_ops import bgmv_expand_slice, bgmv_shrink
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEMlpComputeInput, MoEWeights
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op


def all_gather(inputs, dim=-1):
    world = dist.get_world_size()
    output = torch.empty((world * inputs.shape[0], *inputs.shape[1:]), dtype=inputs.dtype, device=inputs.device)
    dist.all_gather_into_tensor(output, inputs)
    if dim == 0:
        return output
    if dim != -1:
        raise ValueError("The layer benchmark only expects leading-axis or rank-axis AllGather")
    return (
        output.view(world, *inputs.shape)
        .movedim(0, inputs.ndim - 1)
        .reshape(*inputs.shape[:-1], world * inputs.shape[-1])
    )


def all_reduce(inputs):
    dist.all_reduce(inputs)
    return inputs


def set_grouped(enabled):
    # A module-level gate is changed outside capture, never from graph data.
    # The production value is initialized from the centralized env registry.
    grouped.ENABLE_GROUPED_PREFILL = enabled


def measure(call, iterations):
    dist.barrier()
    torch.npu.synchronize()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


@torch.inference_mode()
def run_case(args, tokens, adapter_mode, rank, world, weights, context, wrapper):
    hidden, width, experts, top_k = (
        (4096, 2048 // world, 256, 6) if args.shape == "deepseek" else (2048, 768 // world, 128, 8)
    )
    torch.manual_seed(923)
    x = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16) * 0.05
    topk_ids = (torch.arange(tokens * top_k, device="npu", dtype=torch.int32) % experts).reshape(tokens, top_k)
    slots = torch.arange(tokens, device="npu") % 4 - 1
    if adapter_mode == "single":
        slots.fill_(2)
        slots[::11] = -1
    elif adapter_mode == "invalid90":
        token = torch.arange(tokens, device="npu")
        slots = torch.where(token % 10 == 0, token // 10 % 2 * 2, -1)
    wrapper._token_lora_indices = slots
    wrapper.indices_len = [tokens, 0, 0, 0]
    routed, permutation, counts, scales = torch_npu.npu_moe_init_routing_v2(
        x,
        topk_ids,
        active_num=tokens * top_k,
        expert_num=experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, experts],
        quant_mode=1,
    )
    payload = MoEMlpComputeInput(
        hidden_states=routed,
        dynamic_scale=scales,
        output_dtype=torch.bfloat16,
        group_list=counts,
        group_list_type=1,
        topk_scales=None,
        weights=weights,
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
        fusion=True,
        swiglu_limit=10.0,
        expanded_row_idx=permutation,
        topk_ids=topk_ids,
        lora_context=context,
    )
    graph_mode = args.graph or tokens <= 8
    set_grouped(True)
    selected = grouped.can_use_grouped_moe_lora(context, routed, width)
    functions, captured, outputs, references = {}, {}, {}, {}
    for name, enabled in (("legacy", False), ("grouped", True)):
        set_grouped(enabled)

        def eager(enabled=enabled):
            set_grouped(enabled)
            return quant.quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)[0]

        for _ in range(3):
            result = eager()
        torch.npu.synchronize()
        # Keep untimed precision references off device so large Prefill
        # comparisons do not retain additional full-width NPU outputs.
        references[name] = result.cpu()
        if graph_mode:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = eager()
            captured[name], outputs[name] = graph, result
            functions[name] = graph.replay
        else:
            functions[name] = eager
        for _ in range(5):
            functions[name]()
    old = references["legacy"].float()
    new = references["grouped"].float()
    errors = {
        "relative_l2": ((old - new).norm() / old.norm().clamp_min(1e-12)).item(),
        "max_abs": (old - new).abs().max().item(),
    }
    torch.testing.assert_close(new.cpu(), old.cpu(), atol=1e-5, rtol=0.01)
    del references, old, new
    samples = {name: [] for name in functions}
    order_by_round = []
    for repeat in range(args.repeats):
        order = ["legacy", "grouped"] if repeat % 2 == 0 else ["grouped", "legacy"]
        order_by_round.append(order)
        for name in order:
            samples[name].append(measure(functions[name], args.iterations))
    replay_checked = False
    if graph_mode:
        saved_slots = slots.clone()
        slots.copy_((slots + 2) % 4 - 1)
        # In-place adapter replacement must update both packed A and B inputs.
        context.w13_lora_a_packed[:, 0].mul_(0.75)
        context.w13_lora_b_stacked[0][0].mul_(1.25)
        for name, enabled in (("legacy", False), ("grouped", True)):
            functions[name]()
            actual = outputs[name].clone()
            set_grouped(enabled)
            expected = quant.quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)[0]
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-5, rtol=0.01)
        slots.copy_(saved_slots)
        replay_checked = True
    return {
        "tokens": tokens,
        "routed_rows": tokens * top_k,
        "tp_size": world,
        "hidden": hidden,
        "intermediate_per_partition": width,
        "experts": experts,
        "rank": rank,
        "adapter_mode": adapter_mode,
        "active_fraction": ((slots == 0) | (slots == 2)).float().mean().item(),
        "grouped_path_selected": selected,
        "graph": graph_mode,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "order": order_by_round,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "precision": errors,
        "graph_mapping_and_weight_update_checked": replay_checked,
        "measurements": {
            name: {
                "median_us": statistics.median(values),
                "samples_us": values,
                "min_us": min(values),
                "max_us": max(values),
            }
            for name, values in samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 1024, 8192])
    parser.add_argument("--shape", choices=["deepseek", "qwen"], default="deepseek")
    parser.add_argument("--adapters", nargs="+", choices=["single", "mixed", "invalid90"], default=["single", "mixed"])
    parser.add_argument(
        "--graph", action="store_true", help="Also capture large Prefill layers for an operator graph test"
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.tokens, args.iterations) <= 0 or args.repeats < 5:
        parser.error("positive token/iteration counts and at least five rounds are required")
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world not in (2, 8):
        parser.error("this benchmark expects TP2 or TP8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"rank{rank}.json"
    if output_file.exists():
        parser.error("rank output exists; preserve previous measurements")
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    assert enable_custom_op()
    torch.manual_seed(87 + rank)
    hidden, width, experts, top_k = (
        (4096, 2048 // world, 256, 6) if args.shape == "deepseek" else (2048, 768 // world, 128, 8)
    )

    def weight(*shape):
        return torch.randn(*shape, device="npu", dtype=torch.bfloat16) * 0.01

    wrapper = object.__new__(punica.PunicaWrapperNPU)
    wrapper.bgmv_shrink, wrapper.bgmv_expand_slice = bgmv_shrink, bgmv_expand_slice
    packed = weight(2, 3, experts, 16 // world, hidden)
    context = SimpleNamespace(
        use_ep=False,
        top_k=top_k,
        tp_rank=rank,
        tp_size=world,
        fully_sharded=True,
        punica_wrapper=wrapper,
        adapter_enabled=torch.tensor([1, 0, 1], device="npu", dtype=torch.int32),
        w13_lora_a_stacked=packed.unbind(0),
        w13_lora_a_packed=packed,
        w13_lora_b_stacked=[weight(3, experts, width, 16) for _ in range(2)],
        w2_lora_a_stacked=[weight(3, experts, 16, width)],
        w2_lora_b_stacked=[weight(3, experts, hidden // world, 16)],
    )
    weights = MoEWeights(
        w1=[torch.randint(-10, 11, (experts, hidden, width * 2), device="npu", dtype=torch.int8)],
        w2=[torch.randint(-10, 11, (experts, width, hidden), device="npu", dtype=torch.int8)],
        w1_scale=[torch.full((experts, width * 2), 0.01, device="npu", dtype=torch.bfloat16)],
        w2_scale=[torch.full((experts, hidden), 0.01, device="npu", dtype=torch.bfloat16)],
    )
    results = []
    with (
        patch.object(punica, "tensor_model_parallel_all_gather", all_gather),
        patch.object(punica, "tensor_model_parallel_all_reduce", all_reduce),
        patch.object(quant, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
    ):
        for tokens in args.tokens:
            for mode in args.adapters:
                result = run_case(args, tokens, mode, rank, world, weights, context, wrapper)
                results.append(result)
                output_file.write_text(json.dumps(results, indent=2) + "\n")
                if rank == 0:
                    summary = {key: value for key, value in result.items() if key != "cpu_affinity"}
                    print(json.dumps(summary), flush=True)
                torch.npu.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
