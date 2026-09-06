#!/usr/bin/env python3
"""Benchmark fused EP AllGather MoE LoRA BGMV-index preparation."""

from __future__ import annotations

import argparse
import time

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: F401

from vllm_ascend.lora.lora_ops import moe_lora_prepare_allgather_bgmv_indices


def _make_case(
    num_tokens: int,
    *,
    top_k: int,
    num_global_experts: int,
    num_local_experts: int,
    num_loras: int,
    device: torch.device,
) -> dict[str, object]:
    local_start = num_global_experts // 2
    topk_ids_cpu = (
        torch.arange(num_tokens * top_k, dtype=torch.int32).view(num_tokens, top_k) * 29 + 7
    ) % num_global_experts
    # Every token has at least one local route; other routes retain a realistic
    # global spread and exercise the non-local destination mask.
    topk_ids_cpu[:, 0] = local_start + torch.arange(num_tokens, dtype=torch.int32) % num_local_experts
    topk_ids = topk_ids_cpu.to(device)
    hidden_states = torch.zeros(num_tokens, 16, dtype=torch.bfloat16, device=device)
    _, expanded_row_idx, _, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids,
        active_num=topk_ids.numel(),
        expert_num=num_global_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[local_start, local_start + num_local_experts],
        quant_mode=-1,
        row_idx_type=0,
    )
    expert_map = torch.full(
        (num_global_experts,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    expert_map[local_start : local_start + num_local_experts] = torch.arange(
        num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    token_lora_indices = torch.arange(num_tokens, dtype=torch.int64, device=device) % num_loras
    if num_tokens > 2:
        token_lora_indices[2::5] = -1
    adapter_enabled = torch.ones(num_loras, dtype=torch.int32, device=device)
    if num_loras > 2:
        adapter_enabled[-1] = 0
    return {
        "expanded_row_idx": expanded_row_idx,
        "topk_ids": topk_ids,
        "token_lora_indices": token_lora_indices,
        "expert_map": expert_map,
        "adapter_enabled": adapter_enabled,
        "num_local_experts": num_local_experts,
        "num_global_experts": num_global_experts,
        "local_start": local_start,
    }


def _reference(case: dict[str, object]) -> torch.Tensor:
    expanded_row_idx = case["expanded_row_idx"]
    topk_ids = case["topk_ids"]
    expert_map = case["expert_map"]
    token_lora_indices = case["token_lora_indices"]
    expanded = expanded_row_idx.to(torch.float32).abs()
    flat_expert_ids = topk_ids.reshape(-1).to(torch.long)
    local_expert_ids = expert_map[flat_expert_ids].to(torch.long)
    is_local = local_expert_ids >= 0
    destination = expanded.to(torch.long).clamp_(max=max(flat_expert_ids.numel() - 1, 0))
    lora_per_pair = token_lora_indices.unsqueeze(-1).expand_as(topk_ids).reshape(-1)
    encoded_expert_ids = torch.where(
        is_local,
        local_expert_ids + 1,
        torch.zeros_like(local_expert_ids),
    )
    expert_per_row = torch.zeros_like(encoded_expert_ids)
    expert_per_row.scatter_add_(0, destination, encoded_expert_ids)
    expert_per_row.sub_(1).clamp_min_(0)

    encoded_lora_ids = torch.where(
        is_local & (lora_per_pair >= 0),
        lora_per_pair + 1,
        torch.zeros_like(lora_per_pair),
    )
    lora_per_row = torch.zeros_like(encoded_lora_ids)
    lora_per_row.scatter_add_(0, destination, encoded_lora_ids)
    lora_per_row.sub_(1)

    safe_slots = lora_per_row.clamp(min=0)
    adapter_enabled = case["adapter_enabled"]
    enabled = (lora_per_row >= 0) & adapter_enabled[safe_slots].bool()
    return torch.where(
        enabled,
        safe_slots * case["num_local_experts"] + expert_per_row,
        torch.full_like(lora_per_row, -1),
    ).contiguous()


def _fused(case: dict[str, object]) -> torch.Tensor:
    return moe_lora_prepare_allgather_bgmv_indices(
        case["expanded_row_idx"],
        case["topk_ids"],
        case["token_lora_indices"],
        case["expert_map"],
        case["adapter_enabled"],
        case["num_local_experts"],
    )


def _combined_scatter(case: dict[str, object]) -> torch.Tensor:
    topk_ids = case["topk_ids"]
    flat_expert_ids = topk_ids.reshape(-1).to(torch.long)
    local_expert_ids = case["expert_map"][flat_expert_ids].to(torch.long)
    is_local = local_expert_ids >= 0
    destination = case["expanded_row_idx"].to(torch.float32).abs().to(torch.long)
    destination.clamp_(max=max(flat_expert_ids.numel() - 1, 0))
    lora_per_pair = case["token_lora_indices"].unsqueeze(-1).expand_as(topk_ids).reshape(-1)
    safe_lora_slots = lora_per_pair.clamp(min=0)
    enabled = is_local & (lora_per_pair >= 0) & case["adapter_enabled"][safe_lora_slots].bool()
    encoded_indices = torch.where(
        enabled,
        safe_lora_slots * case["num_local_experts"] + local_expert_ids + 1,
        torch.zeros_like(local_expert_ids),
    )
    output = torch.zeros_like(encoded_indices)
    output.scatter_add_(0, destination, encoded_indices)
    return output.sub_(1).contiguous()


def _assert_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    case: dict[str, object],
) -> None:
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    mismatch = actual_cpu != expected_cpu
    if mismatch.any():
        mismatch_rows = mismatch.nonzero().flatten()
        sample_rows = mismatch_rows[:16]
        expanded_row_idx = case["expanded_row_idx"].cpu()
        topk_ids = case["topk_ids"].reshape(-1).cpu()
        local_expert_ids = case["expert_map"].cpu()[topk_ids.to(torch.long)]
        local_pairs = local_expert_ids >= 0
        local_destinations = expanded_row_idx[local_pairs].to(torch.float32).abs().to(torch.long)
        pair_ids = torch.arange(expanded_row_idx.numel())
        local_pair_ids = pair_ids[local_pairs]
        sample_source_pairs = []
        for row in sample_rows:
            sources = local_pair_ids[local_destinations == row]
            sample_source_pairs.append(sources[0].item() if sources.numel() else -1)
        missing_sources = local_pair_ids[torch.isin(local_destinations, mismatch_rows)]
        print(
            "exactness failure:",
            f"mismatches={mismatch_rows.numel()}/{actual_cpu.numel()}",
            f"rows={sample_rows.tolist()}",
            f"actual={actual_cpu[sample_rows].tolist()}",
            f"expected={expected_cpu[sample_rows].tolist()}",
            f"source_pairs={sample_source_pairs}",
            f"source_pair_mod32={[pair % 32 for pair in sample_source_pairs]}",
            f"missing_source_mod32={torch.bincount(missing_sources % 32, minlength=32).tolist()}",
            f"local_destination_unique={local_destinations.unique().numel() == local_destinations.numel()}",
        )
    torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0, atol=0)


def _benchmark(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1_000_000 / iterations


def _benchmark_graph(fn, *, warmup: int, iterations: int) -> tuple[float, torch.Tensor, torch.npu.NPUGraph]:
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
    return (time.perf_counter() - start) * 1_000_000 / iterations, output, graph


def _validate_graph_update(
    case: dict[str, object],
    output: torch.Tensor,
    graph: torch.npu.NPUGraph,
) -> None:
    token_lora_indices = case["token_lora_indices"]
    token_lora_indices.copy_(torch.roll(token_lora_indices, shifts=1))
    case["adapter_enabled"].copy_(torch.roll(case["adapter_enabled"], shifts=1))
    topk_ids = case["topk_ids"]
    updated_topk_ids = torch.roll(topk_ids, shifts=1, dims=0)
    hidden_states = torch.zeros(
        topk_ids.shape[0],
        16,
        dtype=torch.bfloat16,
        device=topk_ids.device,
    )
    _, updated_row_idx, _, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        updated_topk_ids,
        active_num=updated_topk_ids.numel(),
        expert_num=case["num_global_experts"],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[
            case["local_start"],
            case["local_start"] + case["num_local_experts"],
        ],
        quant_mode=-1,
        row_idx_type=0,
    )
    topk_ids.copy_(updated_topk_ids)
    case["expanded_row_idx"].copy_(updated_row_idx)
    expected = _reference(case).cpu()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-global-experts", type=int, default=256)
    parser.add_argument("--num-local-experts", type=int, default=32)
    parser.add_argument("--num-loras", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    print(
        "tokens,pairs,reference_us,combined_us,combined_speedup,"
        "fused_us,fused_speedup,graph_reference_us,graph_combined_us,"
        "graph_combined_speedup,graph_fused_us,graph_fused_speedup"
    )
    for batch_size in args.batch_sizes:
        case = _make_case(
            batch_size,
            top_k=args.top_k,
            num_global_experts=args.num_global_experts,
            num_local_experts=args.num_local_experts,
            num_loras=args.num_loras,
            device=device,
        )
        expected = _reference(case)
        combined = _combined_scatter(case)
        direct_supported = case["topk_ids"].numel() <= 4096
        actual = _fused(case) if direct_supported else None
        torch.npu.synchronize()
        _assert_exact(combined, expected, case=case)
        if actual is not None:
            _assert_exact(actual, expected, case=case)

        reference_case = lambda current_case=case: _reference(current_case)
        combined_case = lambda current_case=case: _combined_scatter(current_case)
        fused_case = lambda current_case=case: _fused(current_case)
        reference_us = _benchmark(
            reference_case,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        combined_us = _benchmark(
            combined_case,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        fused_us = (
            _benchmark(
                fused_case,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            if direct_supported
            else float("nan")
        )
        graph_reference_us, _, _ = _benchmark_graph(
            reference_case,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        graph_combined_us, _, _ = _benchmark_graph(
            combined_case,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        if direct_supported:
            graph_fused_us, graph_output, graph = _benchmark_graph(
                fused_case,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            _validate_graph_update(case, graph_output, graph)
        else:
            graph_fused_us = float("nan")
        print(
            f"{batch_size},{batch_size * args.top_k},{reference_us:.3f},"
            f"{combined_us:.3f},{reference_us / combined_us:.4f},"
            f"{fused_us:.3f},{reference_us / fused_us:.4f},"
            f"{graph_reference_us:.3f},{graph_combined_us:.3f},"
            f"{graph_reference_us / graph_combined_us:.4f},"
            f"{graph_fused_us:.3f},{graph_reference_us / graph_fused_us:.4f}"
        )


if __name__ == "__main__":
    main()
