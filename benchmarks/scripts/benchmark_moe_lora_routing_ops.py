#!/usr/bin/env python3
"""Microbenchmark MoE LoRA decode routing candidates on one Ascend NPU.

This intentionally avoids importing or starting vLLM.  It compares static-
shape metadata paths that turn expert-major LoRA slots plus per-expert counts
into the flattened ``(LoRA slot, expert)`` index consumed by BGMV.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch
import torch_npu  # noqa: F401 -- registers torch.npu and NPU operators
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: F401 -- registers direct kernels

TensorFn = Callable[[], torch.Tensor]


def custom_fused_aiv(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """One AIV launch that writes the final BGMV index in expert order."""
    torch.ops._C_ascend.moe_lora_prepare_bgmv_indices(
        routed_slots,
        group_counts,
        adapter_enabled,
        output,
    )
    return output


def _finish_combined_index(
    expert_ids: torch.Tensor,
    routed_slots: torch.Tensor,
    valid_rows: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    safe_slots = routed_slots.clamp(min=0, max=adapter_enabled.numel() - 1)
    enabled = valid_rows & (routed_slots >= 0) & adapter_enabled[safe_slots].bool()
    return torch.where(
        enabled,
        safe_slots * num_experts + expert_ids,
        torch.full_like(routed_slots, -1),
    ).contiguous()


def current_path(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
) -> torch.Tensor:
    """Exact current from-slots recovery followed by Punica preparation."""
    counts = group_counts.to(torch.long)
    expert_ends = torch.cumsum(counts, dim=0)
    valid_count = counts.sum()
    row_ids = torch.arange(routed_slots.numel(), dtype=torch.long, device=routed_slots.device)
    expert_ids = torch.searchsorted(expert_ends, row_ids, right=True)
    valid_rows = row_ids < valid_count
    expert_per_row = torch.where(
        valid_rows,
        expert_ids.clamp_max_(counts.numel() - 1),
        torch.zeros_like(row_ids),
    )
    lora_per_row = torch.where(
        valid_rows,
        routed_slots,
        torch.full_like(routed_slots, -1),
    )
    return _finish_combined_index(
        expert_per_row,
        lora_per_row,
        torch.ones_like(valid_rows),
        adapter_enabled,
        counts.numel(),
    )


def current_cached_rows(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    counts = group_counts.to(torch.long)
    expert_ends = torch.cumsum(counts, dim=0)
    valid_count = counts.sum()
    expert_ids = torch.searchsorted(expert_ends, row_ids, right=True)
    valid_rows = row_ids < valid_count
    expert_per_row = torch.where(
        valid_rows,
        expert_ids.clamp_max_(counts.numel() - 1),
        torch.zeros_like(row_ids),
    )
    lora_per_row = torch.where(valid_rows, routed_slots, torch.full_like(routed_slots, -1))
    return _finish_combined_index(
        expert_per_row,
        lora_per_row,
        torch.ones_like(valid_rows),
        adapter_enabled,
        counts.numel(),
    )


def fused_searchsorted(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    """Keep searchsorted but materialize only the final BGMV index."""
    expert_ends = torch.cumsum(group_counts, dim=0)
    valid_count = group_counts.sum()
    expert_ids = torch.searchsorted(expert_ends, row_ids, right=True).clamp_max_(group_counts.numel() - 1)
    return _finish_combined_index(
        expert_ids,
        routed_slots,
        row_ids < valid_count,
        adapter_enabled,
        group_counts.numel(),
    )


def broadcast_sum(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    """Replace searchsorted with one broadcast comparison and reduction."""
    expert_ends = torch.cumsum(group_counts, dim=0)
    valid_count = group_counts.sum()
    expert_ids = (row_ids[:, None] >= expert_ends[None, :]).sum(dim=1)
    expert_ids.clamp_max_(group_counts.numel() - 1)
    return _finish_combined_index(
        expert_ids,
        routed_slots,
        row_ids < valid_count,
        adapter_enabled,
        group_counts.numel(),
    )


def broadcast_argmax(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    """Find the first expert end above each row instead of searchsorted."""
    expert_ends = torch.cumsum(group_counts, dim=0)
    valid_count = group_counts.sum()
    expert_ids = torch.argmax((row_ids[:, None] < expert_ends[None, :]).to(torch.int32), dim=1)
    return _finish_combined_index(
        expert_ids,
        routed_slots,
        row_ids < valid_count,
        adapter_enabled,
        group_counts.numel(),
    )


def cube_prefix_broadcast(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    row_ids: torch.Tensor,
    prefix_matrix: torch.Tensor,
) -> torch.Tensor:
    """Replace cumsum with a fixed triangular BF16 matmul candidate."""
    expert_ends = torch.matmul(group_counts.to(torch.bfloat16).view(1, -1), prefix_matrix).view(-1).to(torch.long)
    valid_count = expert_ends[-1]
    expert_ids = (row_ids[:, None] >= expert_ends[None, :]).sum(dim=1)
    expert_ids.clamp_max_(group_counts.numel() - 1)
    return _finish_combined_index(
        expert_ids,
        routed_slots,
        row_ids < valid_count,
        adapter_enabled,
        group_counts.numel(),
    )


def repeat_interleave_path(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    row_ids: torch.Tensor,
    expert_values_with_tail: torch.Tensor,
) -> torch.Tensor:
    """Use one repeat-interleave expansion and an explicit static tail."""
    valid_count = group_counts.sum()
    tail_count = (routed_slots.numel() - valid_count).view(1)
    counts_with_tail = torch.cat((group_counts, tail_count))
    expert_ids = torch.repeat_interleave(
        expert_values_with_tail,
        counts_with_tail,
        output_size=routed_slots.numel(),
    )
    return _finish_combined_index(
        expert_ids,
        routed_slots,
        row_ids < valid_count,
        adapter_enabled,
        group_counts.numel(),
    )


def repeat_interleave_sentinel(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    expert_values_with_sentinel: torch.Tensor,
) -> torch.Tensor:
    """Encode the static tail as expert -1 and avoid a row-id vector."""
    tail_count = (routed_slots.numel() - group_counts.sum()).view(1)
    expert_ids = torch.repeat_interleave(
        expert_values_with_sentinel,
        torch.cat((group_counts, tail_count)),
        output_size=routed_slots.numel(),
    )
    return _finish_combined_index(
        expert_ids,
        routed_slots,
        expert_ids >= 0,
        adapter_enabled,
        group_counts.numel(),
    )


def repeat_interleave_float_slots(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    expert_values_with_sentinel: torch.Tensor,
) -> torch.Tensor:
    """Consume init-routing's FP32 sideband without a dispatcher-side cast."""
    tail_count = (routed_slots.numel() - group_counts.sum()).view(1)
    expert_ids = torch.repeat_interleave(
        expert_values_with_sentinel,
        torch.cat((group_counts, tail_count)),
        output_size=routed_slots.numel(),
    )
    valid_rows = expert_ids >= 0
    local_slots = torch.where(valid_rows, routed_slots, torch.zeros_like(routed_slots))
    safe_slots = local_slots.clamp(min=0, max=adapter_enabled.numel() - 1).to(torch.long)
    enabled = valid_rows & (local_slots >= 0) & adapter_enabled[safe_slots].bool()
    return torch.where(
        enabled,
        safe_slots * group_counts.numel() + expert_ids,
        torch.full_like(expert_ids, -1),
    ).contiguous()


def scatter_preencoded(
    encoded_per_pair: torch.Tensor,
    source_to_destination: torch.Tensor,
) -> torch.Tensor:
    """Single unique-destination scatter; invalid pairs contribute zero."""
    output = torch.zeros_like(encoded_per_pair)
    output.scatter_add_(0, source_to_destination, encoded_per_pair)
    return output.sub_(1)


def _combined_per_pair(
    source_slots: torch.Tensor,
    source_experts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    safe_slots = source_slots.clamp(min=0, max=adapter_enabled.numel() - 1)
    enabled = (source_experts >= 0) & (source_slots >= 0) & adapter_enabled[safe_slots].bool()
    return torch.where(
        enabled,
        safe_slots * num_experts + source_experts.clamp_min(0),
        torch.full_like(source_slots, -1),
    )


def scatter_full(
    source_slots: torch.Tensor,
    source_experts: torch.Tensor,
    source_to_destination: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    combined = _combined_per_pair(source_slots, source_experts, adapter_enabled, num_experts)
    encoded = torch.where(combined >= 0, combined + 1, torch.zeros_like(combined))
    return scatter_preencoded(encoded, source_to_destination)


def gather_reverse_mapping(
    combined_per_pair: torch.Tensor,
    destination_to_source: torch.Tensor,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    """Ideal row_idx_type=1 path; not directly compatible with finalize."""
    safe_source = destination_to_source.clamp(min=0, max=combined_per_pair.numel() - 1)
    gathered = combined_per_pair[safe_source]
    return torch.where(valid_rows, gathered, torch.full_like(gathered, -1))


def gather_reverse_mapping_full(
    source_slots: torch.Tensor,
    source_experts: torch.Tensor,
    destination_to_source: torch.Tensor,
    valid_rows: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    combined = _combined_per_pair(source_slots, source_experts, adapter_enabled, num_experts)
    return gather_reverse_mapping(combined, destination_to_source, valid_rows)


def _make_case(
    batch_size: int,
    *,
    top_k: int,
    num_local_experts: int,
    max_loras: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    num_rows = batch_size * top_k
    valid_count = batch_size

    # Force empty experts and uneven groups, which are the difficult cases
    # for boundary recovery.  The valid prefix is already expert-major.
    valid_experts = (torch.arange(valid_count, dtype=torch.long) * 7 + batch_size) % num_local_experts
    group_counts_cpu = torch.bincount(valid_experts, minlength=num_local_experts).to(torch.long)
    valid_slots_cpu = (torch.arange(valid_count, dtype=torch.long) + batch_size) % max_loras
    if valid_count > 1:
        valid_slots_cpu[1::3] = -1
    expert_major_order = torch.argsort(valid_experts, stable=True)
    routed_slots_cpu = torch.full((num_rows,), 99, dtype=torch.long)
    routed_slots_cpu[:valid_count] = valid_slots_cpu[expert_major_order]

    # Build a source-pair -> expert-sorted-destination mapping for the
    # scatter candidate. Remote pairs are harmless zero contributions.
    source_experts = torch.full((num_rows,), -1, dtype=torch.long)
    local_source_positions = torch.arange(valid_count) * top_k % num_rows
    source_experts[local_source_positions] = valid_experts
    source_slots = torch.full((num_rows,), -1, dtype=torch.long)
    source_slots[local_source_positions] = valid_slots_cpu
    source_to_destination = torch.zeros(num_rows, dtype=torch.long)
    destination_to_source = torch.zeros(num_rows, dtype=torch.long)
    cursor = 0
    for expert in range(num_local_experts):
        source_positions = torch.nonzero(source_experts == expert, as_tuple=False).view(-1)
        for source_position in source_positions:
            source_to_destination[source_position] = cursor
            destination_to_source[cursor] = source_position
            cursor += 1

    safe_source_slots = source_slots.clamp(min=0)
    adapter_enabled_cpu = torch.ones(max_loras, dtype=torch.int32)
    if max_loras > 2:
        adapter_enabled_cpu[-1] = False
    source_enabled = (source_experts >= 0) & (source_slots >= 0) & adapter_enabled_cpu[safe_source_slots].bool()
    combined_per_pair_cpu = torch.where(
        source_enabled,
        safe_source_slots * num_local_experts + source_experts.clamp_min(0),
        torch.full_like(source_slots, -1),
    )
    encoded_per_pair_cpu = torch.where(
        source_enabled,
        combined_per_pair_cpu + 1,
        torch.zeros_like(combined_per_pair_cpu),
    )

    return {
        "group_counts": group_counts_cpu.to(device),
        "routed_slots": routed_slots_cpu.to(device),
        "routed_slots_float": torch.where(
            torch.arange(num_rows) < valid_count,
            routed_slots_cpu.to(torch.float32),
            torch.full((num_rows,), float("nan"), dtype=torch.float32),
        ).to(device),
        "adapter_enabled": adapter_enabled_cpu.to(device),
        "row_ids": torch.arange(num_rows, dtype=torch.long, device=device),
        "expert_values_with_tail": torch.cat(
            (torch.arange(num_local_experts, dtype=torch.long), torch.zeros(1, dtype=torch.long))
        ).to(device),
        "expert_values_with_sentinel": torch.cat(
            (torch.arange(num_local_experts, dtype=torch.long), -torch.ones(1, dtype=torch.long))
        ).to(device),
        "prefix_matrix": torch.triu(torch.ones(num_local_experts, num_local_experts, dtype=torch.bfloat16)).to(device),
        "encoded_per_pair": encoded_per_pair_cpu.to(device),
        "combined_per_pair": combined_per_pair_cpu.to(device),
        "source_to_destination": source_to_destination.to(device),
        "destination_to_source": destination_to_source.to(device),
        "source_experts": source_experts.to(device),
        "source_slots": source_slots.to(device),
        "valid_rows": (torch.arange(num_rows) < valid_count).to(device),
        "custom_output": torch.empty(num_rows, dtype=torch.long, device=device),
    }


def _benchmark(name: str, fn: TensorFn, *, warmup: int, iterations: int) -> tuple[str, float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1_000_000 / iterations
    return name, elapsed_us


def _benchmark_graph(name: str, fn: TensorFn, *, warmup: int, iterations: int) -> tuple[str, float]:
    fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fn()
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.npu.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1_000_000 / iterations
    # Force one host-visible use so an invalid capture cannot pass silently.
    if output.numel() == 0:
        raise AssertionError(f"{name} produced an empty graph output")
    return name, elapsed_us


def _validate_repeat_graph_updates(case: dict[str, torch.Tensor]) -> None:
    candidates = _make_candidates(case)
    repeat_fn = candidates["repeat_interleave"]
    repeat_fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = repeat_fn()

    original_counts = case["group_counts"].clone()
    original_slots = case["routed_slots"].clone()
    count_variants = (
        torch.roll(original_counts, shifts=1),
        torch.roll(original_counts, shifts=7),
        torch.zeros_like(original_counts),
    )
    for counts in count_variants:
        case["group_counts"].copy_(counts)
        valid_count = int(counts.cpu().sum())
        slots = torch.full_like(original_slots, 99)
        if valid_count:
            slots[:valid_count].copy_(original_slots[:valid_count])
        case["routed_slots"].copy_(slots)
        graph.replay()
        torch.npu.synchronize()
        actual = graph_output.cpu().clone()
        expected = current_path(
            case["routed_slots"],
            case["group_counts"],
            case["adapter_enabled"],
        ).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    case["group_counts"].copy_(original_counts)
    case["routed_slots"].copy_(original_slots)
    torch.npu.synchronize()


def _validate_custom_graph_updates(case: dict[str, torch.Tensor]) -> None:
    candidates = _make_candidates(case)
    custom_fn = candidates["custom_fused_aiv"]
    custom_fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = custom_fn()

    original_counts = case["group_counts"].clone()
    original_slots = case["routed_slots_float"].clone()
    original_enabled = case["adapter_enabled"].clone()
    count_variants = (
        torch.roll(original_counts, shifts=1),
        torch.roll(original_counts, shifts=7),
        torch.zeros_like(original_counts),
    )
    for variant_index, counts in enumerate(count_variants):
        case["group_counts"].copy_(counts)
        valid_count = int(counts.cpu().sum())
        slots = torch.full_like(original_slots, float("nan"))
        if valid_count:
            slots[:valid_count].copy_(original_slots[:valid_count])
        if valid_count < slots.numel() and variant_index % 2:
            slots[valid_count] = float("inf")
        case["routed_slots_float"].copy_(slots)
        case["adapter_enabled"].copy_(original_enabled)
        if variant_index == 1 and case["adapter_enabled"].numel() > 1:
            case["adapter_enabled"][1] = 0
        graph.replay()
        torch.npu.synchronize()
        actual = graph_output.cpu().clone()
        expected = repeat_interleave_float_slots(
            case["routed_slots_float"],
            case["group_counts"],
            case["adapter_enabled"],
            case["expert_values_with_sentinel"],
        ).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    case["group_counts"].copy_(original_counts)
    case["routed_slots_float"].copy_(original_slots)
    case["adapter_enabled"].copy_(original_enabled)
    torch.npu.synchronize()


def _make_candidates(case: dict[str, torch.Tensor]) -> dict[str, TensorFn]:
    slots = case["routed_slots"]
    slots_float = case["routed_slots_float"]
    counts = case["group_counts"]
    enabled = case["adapter_enabled"]
    rows = case["row_ids"]
    return {
        "current": lambda: current_path(slots, counts, enabled),
        "current_cached_rows": lambda: current_cached_rows(slots, counts, enabled, rows),
        "fused_searchsorted": lambda: fused_searchsorted(slots, counts, enabled, rows),
        "broadcast_sum": lambda: broadcast_sum(slots, counts, enabled, rows),
        "broadcast_argmax": lambda: broadcast_argmax(slots, counts, enabled, rows),
        "cube_prefix_broadcast": lambda: cube_prefix_broadcast(
            slots,
            counts,
            enabled,
            rows,
            case["prefix_matrix"],
        ),
        "repeat_interleave": lambda: repeat_interleave_path(
            slots,
            counts,
            enabled,
            rows,
            case["expert_values_with_tail"],
        ),
        "repeat_interleave_sentinel": lambda: repeat_interleave_sentinel(
            slots,
            counts,
            enabled,
            case["expert_values_with_sentinel"],
        ),
        "repeat_interleave_float_slots": lambda: repeat_interleave_float_slots(
            slots_float,
            counts,
            enabled,
            case["expert_values_with_sentinel"],
        ),
        "custom_fused_aiv": lambda: custom_fused_aiv(
            slots_float,
            counts,
            enabled,
            case["custom_output"],
        ),
        "scatter_preencoded": lambda: scatter_preencoded(
            case["encoded_per_pair"],
            case["source_to_destination"],
        ),
        "scatter_full": lambda: scatter_full(
            case["source_slots"],
            case["source_experts"],
            case["source_to_destination"],
            enabled,
            counts.numel(),
        ),
        "rowidx1_gather": lambda: gather_reverse_mapping(
            case["combined_per_pair"],
            case["destination_to_source"],
            case["valid_rows"],
        ),
        "rowidx1_gather_full": lambda: gather_reverse_mapping_full(
            case["source_slots"],
            case["source_experts"],
            case["destination_to_source"],
            case["valid_rows"],
            enabled,
            counts.numel(),
        ),
    }


def _inspect_init_routing(device: torch.device) -> None:
    x = torch.arange(4 * 8, dtype=torch.float32).view(4, 8).to(torch.bfloat16).to(device)
    topk_ids = torch.tensor(
        [[5, 1], [0, 7], [2, 1], [6, 3]],
        dtype=torch.int32,
        device=device,
    )
    slots = torch.tensor([0, 1, -1, 2], dtype=torch.float32, device=device)
    print("\nMoeInitRoutingV2 row_idx_type semantics (active experts [0, 4)):")
    for row_idx_type in (0, 1):
        expanded_x, row_idx, counts, expanded_slots = torch_npu.npu_moe_init_routing_v2(
            x,
            topk_ids,
            scale=slots,
            active_num=topk_ids.numel(),
            expert_num=8,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, 4],
            quant_mode=-1,
            row_idx_type=row_idx_type,
        )
        torch.npu.synchronize()
        valid_count = 5
        unpermute_input = expanded_x.clone()
        unpermute_input[valid_count:].zero_()
        probs = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]],
            dtype=torch.bfloat16,
            device=device,
        )
        unpermuted = torch_npu.npu_moe_token_unpermute(unpermute_input, row_idx, probs=probs)
        torch.npu.synchronize()
        print(
            f"  type={row_idx_type}: row_idx={row_idx.cpu().tolist()} "
            f"counts={counts.cpu().tolist()} slots={expanded_slots.cpu().tolist()} "
            f"unpermute_row0={unpermuted[0].cpu().tolist()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-local-experts", type=int, default=32)
    parser.add_argument("--max-loras", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--graph-iterations", type=int, default=1000)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    _inspect_init_routing(device)

    for batch_size in args.batch_sizes:
        case = _make_case(
            batch_size,
            top_k=args.top_k,
            num_local_experts=args.num_local_experts,
            max_loras=args.max_loras,
            device=device,
        )
        candidates = _make_candidates(case)

        reference = candidates["current"]()
        torch.npu.synchronize()
        reference_cpu = reference.cpu()
        for name, fn in candidates.items():
            actual = fn()
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), reference_cpu, rtol=0, atol=0, msg=f"{name} mismatch")

        timings = [
            _benchmark(name, fn, warmup=args.warmup, iterations=args.iterations) for name, fn in candidates.items()
        ]
        timings.sort(key=lambda item: item[1])
        baseline_us = dict(timings)["current"]
        print(
            f"\nbatch={batch_size:2d} routed_rows={batch_size * args.top_k:3d} "
            f"valid_rows={batch_size:2d} local_experts={args.num_local_experts}"
        )
        for name, latency_us in timings:
            print(f"  {name:24s} {latency_us:9.3f} us  speedup={baseline_us / latency_us:5.2f}x")

        if args.graph:
            _validate_repeat_graph_updates(case)
            print("  repeat_interleave dynamic graph replay: PASS")
            _validate_custom_graph_updates(case)
            print("  custom_fused_aiv dynamic graph replay: PASS")
            graph_names = (
                "current",
                "fused_searchsorted",
                "repeat_interleave",
                "repeat_interleave_sentinel",
                "repeat_interleave_float_slots",
                "custom_fused_aiv",
                "cube_prefix_broadcast",
                "scatter_full",
                "rowidx1_gather_full",
            )
            graph_timings: list[tuple[str, float]] = []
            print("  ACLGraph replay:")
            for name in graph_names:
                try:
                    graph_timings.append(
                        _benchmark_graph(
                            name,
                            candidates[name],
                            warmup=args.warmup,
                            iterations=args.graph_iterations,
                        )
                    )
                except Exception as error:  # noqa: BLE001 -- report unsupported capture candidates
                    print(f"    {name:22s} CAPTURE_FAILED: {type(error).__name__}: {error}")
            if graph_timings:
                graph_timings.sort(key=lambda item: item[1])
                graph_baseline = dict(graph_timings).get("current")
                for name, latency_us in graph_timings:
                    speedup = graph_baseline / latency_us if graph_baseline is not None else float("nan")
                    print(f"    {name:22s} {latency_us:9.3f} us  speedup={speedup:5.2f}x")


if __name__ == "__main__":
    main()
