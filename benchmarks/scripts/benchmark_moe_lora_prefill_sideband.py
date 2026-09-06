#!/usr/bin/env python3
"""Benchmark complete EP prefill routing with and without the LoRA sideband."""

from __future__ import annotations

import argparse
import time

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: F401


def _benchmark(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / iterations


def _make_case(
    num_tokens: int,
    *,
    hidden_size: int,
    top_k: int,
    num_global_experts: int,
    num_local_experts: int,
    num_loras: int,
    device: torch.device,
):
    hidden_states = torch.zeros(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids = (
        torch.arange(num_tokens * top_k, dtype=torch.int32, device=device).view(num_tokens, top_k) * 29 + 7
    ) % num_global_experts
    topk_ids[:, 0] = torch.arange(num_tokens, dtype=torch.int32, device=device) % num_local_experts
    token_lora_indices = torch.arange(num_tokens, dtype=torch.int64, device=device) % num_loras
    if num_tokens > 2:
        token_lora_indices[2::5] = -1
    token_lora_slots = token_lora_indices.to(torch.float32).contiguous()
    adapter_enabled = torch.ones(num_loras, dtype=torch.int32, device=device)
    if num_loras > 2:
        adapter_enabled[-1] = 0
    expert_map = torch.full(
        (num_global_experts,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    expert_map[:num_local_experts] = torch.arange(
        num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    routing_kwargs = {
        "active_num": topk_ids.numel(),
        "expert_num": num_global_experts,
        "expert_tokens_num_type": 1,
        "expert_tokens_num_flag": True,
        "active_expert_range": [0, num_local_experts],
        "quant_mode": -1,
        "row_idx_type": 0,
    }

    def baseline() -> torch.Tensor:
        _, expanded_row_idx, _, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            **routing_kwargs,
        )
        expanded = expanded_row_idx.to(torch.float32).abs()
        flat_expert_ids = topk_ids.reshape(-1).to(torch.long)
        local_expert_ids = expert_map[flat_expert_ids].to(torch.long)
        is_local = local_expert_ids >= 0
        destination = expanded.to(torch.long).clamp_(max=max(flat_expert_ids.numel() - 1, 0))
        encoded_expert_ids = torch.where(
            is_local,
            local_expert_ids + 1,
            torch.zeros_like(local_expert_ids),
        )
        expert_per_row = torch.zeros_like(encoded_expert_ids)
        expert_per_row.scatter_add_(0, destination, encoded_expert_ids)
        expert_per_row.sub_(1).clamp_min_(0)

        lora_per_pair = token_lora_indices.unsqueeze(-1).expand_as(topk_ids).reshape(-1)
        encoded_lora_ids = torch.where(
            is_local & (lora_per_pair >= 0),
            lora_per_pair + 1,
            torch.zeros_like(lora_per_pair),
        )
        lora_per_row = torch.zeros_like(encoded_lora_ids)
        lora_per_row.scatter_add_(0, destination, encoded_lora_ids)
        lora_per_row.sub_(1)
        safe_slots = lora_per_row.clamp(min=0)
        enabled = (lora_per_row >= 0) & adapter_enabled[safe_slots].bool()
        return torch.where(
            enabled,
            safe_slots * num_local_experts + expert_per_row,
            torch.full_like(lora_per_row, -1),
        ).contiguous()

    def sideband() -> torch.Tensor:
        _, _, group_list, routed_lora_slots = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=token_lora_slots,
            **routing_kwargs,
        )
        output = torch.empty_like(routed_lora_slots, dtype=torch.long)
        torch.ops._C_ascend.moe_lora_prepare_bgmv_indices(
            routed_lora_slots,
            group_list,
            adapter_enabled,
            output,
        )
        return output

    return baseline, sideband


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-global-experts", type=int, default=256)
    parser.add_argument("--num-local-experts", type=int, default=32)
    parser.add_argument("--num-loras", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    print("tokens,routed_rows,baseline_us,sideband_us,speedup")
    for batch_size in args.batch_sizes:
        baseline, sideband = _make_case(
            batch_size,
            hidden_size=args.hidden_size,
            top_k=args.top_k,
            num_global_experts=args.num_global_experts,
            num_local_experts=args.num_local_experts,
            num_loras=args.num_loras,
            device=device,
        )
        expected = baseline()
        actual = sideband()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        baseline_us = _benchmark(
            baseline,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        sideband_us = _benchmark(
            sideband,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(
            f"{batch_size},{batch_size * args.top_k},{baseline_us:.3f},"
            f"{sideband_us:.3f},{baseline_us / sideband_us:.4f}"
        )


if __name__ == "__main__":
    main()
