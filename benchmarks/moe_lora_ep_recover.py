# SPDX-License-Identifier: Apache-2.0
"""Measure AllGather EP MoE LoRA route recovery on one Ascend NPU.

Example: ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/moe_lora_ep_recover.py --output ep_recover.json
The service benchmark is needed to decide whether the local kernel gain improves TPOT.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm_ascend.utils import enable_custom_op


def framework_recover(expanded, topk, slots, top_k, expert_start, local_experts):
    rows = expanded.numel()
    source = torch.arange(rows, device=expanded.device, dtype=torch.long)
    valid = expanded >= 0
    keys = torch.where(valid, expanded.long(), rows + source)
    inverse = torch.argsort(keys.float() if rows <= (1 << 23) else keys)
    experts = topk.reshape(-1)[inverse].long() - expert_start
    lora_slots = slots[(inverse // top_k).clamp(max=slots.numel() - 1)]
    active = valid[inverse] & (experts >= 0) & (experts < local_experts)
    return torch.where(active, experts, -1), torch.where(active, lora_slots, -1)


def measure(fn, *, iterations, repeats):
    for _ in range(5):
        fn()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fn()
    for _ in range(5):
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
def benchmark(tokens, top_k, num_experts, ep_size, ep_rank, iterations, repeats):
    first = ep_rank * (num_experts // ep_size)
    local = num_experts // ep_size
    rows = tokens * top_k
    topk = torch.arange(rows, dtype=torch.int32).remainder(num_experts).view(tokens, top_k)
    topk[:, 0] = first + torch.arange(tokens, dtype=torch.int32).remainder(local)
    expanded = torch.full((rows,), -1, dtype=torch.int32)
    valid_sources = [source for source, expert in enumerate(topk.flatten().tolist()) if first <= expert < first + local]
    for destination, source in enumerate(reversed(valid_sources)):
        expanded[source] = destination
    slots = torch.arange(tokens, dtype=torch.int64).remainder(3) - 1
    args = (expanded.npu(), topk.npu(), slots.npu(), top_k, first, local)

    reference = framework_recover(*args)
    native = torch.ops._C_ascend.moe_lora_recover_ep(*args)
    for actual, expected in zip(native, reference):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    return {
        "tokens": tokens,
        "top_k": top_k,
        "num_experts": num_experts,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "active_rows": len(valid_sources),
        "native": measure(
            lambda: torch.ops._C_ascend.moe_lora_recover_ep(*args), iterations=iterations, repeats=repeats
        ),
        "framework": measure(lambda: framework_recover(*args), iterations=iterations, repeats=repeats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve prior measurements")
    if min(*args.tokens, args.iterations, args.repeats) <= 0:
        parser.error("tokens, iterations and repeats must be positive")
    assert enable_custom_op()
    results = []
    for num_experts, top_k in ((256, 6), (128, 8)):
        for tokens in args.tokens:
            result = benchmark(tokens, top_k, num_experts, 8, 3, args.iterations, args.repeats)
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
