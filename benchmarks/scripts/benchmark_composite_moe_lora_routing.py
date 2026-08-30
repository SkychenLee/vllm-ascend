#!/usr/bin/env python3
"""Benchmark composite MoE LoRA GMM routing on one Ascend NPU.

The benchmark isolates the mixed-prefill metadata path. It compares the
original PyTorch operator sequence with the direct AscendC kernel, validates
dynamic ACLGraph replay, and measures token permutation separately so routing
metadata and hidden-state movement are not conflated.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial

import torch
import torch_npu  # noqa: F401 -- register NPU operators
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: F401

RoutingFn = Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def original_composite_routing(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = group_counts.numel()
    num_loras = adapter_enabled.numel()
    num_groups = num_loras * num_experts
    row_ids = torch.arange(
        routed_slots.numel(),
        dtype=group_counts.dtype,
        device=routed_slots.device,
    )
    valid_rows = row_ids < group_counts.sum()
    expert_ids = torch.searchsorted(
        group_counts.cumsum(0),
        row_ids,
        right=True,
    ).clamp_(max=num_experts - 1)
    safe_slots = routed_slots.clamp(min=0, max=num_loras - 1)
    enabled = valid_rows & (routed_slots >= 0) & adapter_enabled[safe_slots.long()].bool()
    active_group_ids = safe_slots.long() * num_experts + expert_ids
    group_ids = torch.where(enabled, active_group_ids, num_groups).to(torch.int32)
    composite_counts = torch.zeros(
        num_groups,
        dtype=torch.int64,
        device=routed_slots.device,
    )
    composite_counts.scatter_add_(0, active_group_ids, enabled.to(torch.int64))
    return group_ids, composite_counts, enabled


def fused_composite_routing(
    routed_slots: torch.Tensor,
    group_counts: torch.Tensor,
    adapter_enabled: torch.Tensor,
    group_ids: torch.Tensor,
    composite_counts: torch.Tensor,
    enabled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.ops._C_ascend.moe_lora_prepare_composite_gmm_routing(
        routed_slots,
        group_counts,
        adapter_enabled,
        group_ids,
        composite_counts,
        enabled,
    )
    return group_ids, composite_counts, enabled


def _make_case(
    num_rows: int,
    *,
    num_experts: int,
    num_loras: int,
    local_fraction: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    valid_rows = max(1, min(num_rows, round(num_rows * local_fraction)))
    base_count, remainder = divmod(valid_rows, num_experts)
    counts_cpu = torch.full((num_experts,), base_count, dtype=torch.int64)
    counts_cpu[:remainder].add_(1)

    slots_cpu = torch.full((num_rows,), float("nan"), dtype=torch.float32)
    slot_pattern = torch.arange(valid_rows, dtype=torch.int64) % (num_loras + 1)
    slot_pattern = torch.where(
        slot_pattern == num_loras,
        torch.full_like(slot_pattern, -1),
        slot_pattern,
    )
    slots_cpu[:valid_rows] = slot_pattern.float()
    if valid_rows < num_rows:
        slots_cpu[valid_rows] = float("inf")
    adapter_enabled_cpu = torch.ones(num_loras, dtype=torch.int32)
    if num_loras > 1:
        adapter_enabled_cpu[1] = 0

    return {
        "slots": slots_cpu.to(device),
        "counts": counts_cpu.to(device),
        "adapter_enabled": adapter_enabled_cpu.to(device),
        "group_ids": torch.empty(num_rows, dtype=torch.int32, device=device),
        "composite_counts": torch.empty(
            num_experts * num_loras,
            dtype=torch.int64,
            device=device,
        ),
        "enabled": torch.empty(num_rows, dtype=torch.bool, device=device),
    }


def _assert_routing_equal(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    labels = ("group_ids", "composite_counts", "enabled")
    for label, actual_tensor, expected_tensor in zip(labels, actual, expected, strict=True):
        actual_cpu = actual_tensor.cpu()
        expected_cpu = expected_tensor.cpu()
        try:
            torch.testing.assert_close(
                actual_cpu,
                expected_cpu,
                rtol=0,
                atol=0,
            )
        except AssertionError as error:
            mismatch = torch.nonzero(actual_cpu != expected_cpu).flatten()
            preview = mismatch[:16]
            raise AssertionError(
                f"{label} mismatch at {preview.tolist()}: "
                f"actual={actual_cpu[preview].tolist()} "
                f"expected={expected_cpu[preview].tolist()}"
            ) from error


def _benchmark(fn: Callable[[], object], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def _capture_graph(fn: RoutingFn) -> tuple[torch.npu.NPUGraph, tuple[torch.Tensor, ...]]:
    fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = fn()
    return graph, outputs


def _validate_dynamic_graph(
    case: dict[str, torch.Tensor],
    fused_fn: RoutingFn,
) -> tuple[torch.npu.NPUGraph, tuple[torch.Tensor, ...]]:
    graph, outputs = _capture_graph(fused_fn)
    original_counts = case["counts"].clone()
    original_slots = case["slots"].clone()
    original_enabled = case["adapter_enabled"].clone()
    for variant in range(3):
        counts = torch.roll(original_counts, shifts=variant + 1)
        if variant == 2:
            counts.zero_()
        valid_rows = int(counts.cpu().sum())
        slots = torch.full_like(original_slots, float("nan"))
        if valid_rows:
            slots[:valid_rows].copy_(original_slots[:valid_rows])
        case["counts"].copy_(counts)
        case["slots"].copy_(slots)
        case["adapter_enabled"].copy_(original_enabled)
        if variant == 1 and original_enabled.numel() > 2:
            case["adapter_enabled"][2] = 0
        graph.replay()
        torch.npu.synchronize()
        _assert_routing_equal(
            outputs,
            original_composite_routing(
                case["slots"],
                case["counts"],
                case["adapter_enabled"],
            ),
        )
    case["counts"].copy_(original_counts)
    case["slots"].copy_(original_slots)
    case["adapter_enabled"].copy_(original_enabled)
    return graph, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[768, 6144, 49152, 131072])
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--max-loras", type=int, default=3)
    parser.add_argument("--local-fraction", type=float, default=0.125)
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--compare-init-routing", action="store_true")
    parser.add_argument("--compare-compact-permute", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    for num_rows in args.rows:
        case = _make_case(
            num_rows,
            num_experts=args.num_experts,
            num_loras=args.max_loras,
            local_fraction=args.local_fraction,
            device=device,
        )
        original_fn = partial(
            original_composite_routing,
            case["slots"],
            case["counts"],
            case["adapter_enabled"],
        )
        fused_fn = partial(
            fused_composite_routing,
            case["slots"],
            case["counts"],
            case["adapter_enabled"],
            case["group_ids"],
            case["composite_counts"],
            case["enabled"],
        )
        expected = original_fn()
        actual = fused_fn()
        torch.npu.synchronize()
        _assert_routing_equal(actual, expected)

        original_us = _benchmark(original_fn, warmup=args.warmup, iterations=args.iterations)
        fused_us = _benchmark(fused_fn, warmup=args.warmup, iterations=args.iterations)
        graph, graph_outputs = _validate_dynamic_graph(case, fused_fn)
        graph_us = _benchmark(graph.replay, warmup=args.warmup, iterations=args.iterations)
        _assert_routing_equal(graph_outputs, original_fn())
        print(
            f"rows={num_rows:6d} valid={int(case['counts'].cpu().sum()):6d} "
            f"original={original_us:9.3f} us fused={fused_us:9.3f} us "
            f"speedup={original_us / fused_us:6.2f}x graph={graph_us:9.3f} us"
        )

        if args.hidden_size:
            tokens = torch.randn(
                num_rows,
                args.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            permute_fn = partial(
                torch_npu.npu_moe_token_permute,
                tokens=tokens,
                indices=case["group_ids"],
                num_out_tokens=num_rows,
            )
            permute_us = _benchmark(
                permute_fn,
                warmup=max(5, args.warmup // 5),
                iterations=max(20, args.iterations // 10),
            )
            print(f"  token_permute hidden={args.hidden_size}: {permute_us:9.3f} us")

            if args.compare_compact_permute:

                def compact_permute_fn(
                    tokens=tokens,
                    group_ids=case["group_ids"],
                    composite_counts=case["composite_counts"],
                ):
                    active_rows = int(composite_counts.sum().item())
                    return torch_npu.npu_moe_token_permute(
                        tokens=tokens,
                        indices=group_ids,
                        num_out_tokens=active_rows,
                    )

                grouped_inputs, reverse_mapping = permute_fn()
                compact_inputs, compact_mapping = compact_permute_fn()
                torch.npu.synchronize()
                active_rows = compact_inputs.shape[0]
                torch.testing.assert_close(
                    compact_inputs.cpu(),
                    grouped_inputs[:active_rows].cpu(),
                    rtol=0,
                    atol=0,
                )
                restored = torch_npu.npu_moe_token_unpermute(
                    permuted_tokens=compact_inputs,
                    sorted_indices=compact_mapping,
                )
                expected_restored = tokens * case["enabled"].unsqueeze(-1).to(tokens.dtype)
                torch.testing.assert_close(
                    restored.cpu(),
                    expected_restored.cpu(),
                    rtol=0,
                    atol=0,
                )
                del reverse_mapping
                compact_us = _benchmark(
                    compact_permute_fn,
                    warmup=max(5, args.warmup // 5),
                    iterations=max(20, args.iterations // 10),
                )
                print(f"  compact_permute active={active_rows:6d}: {compact_us:9.3f} us vs full={permute_us:9.3f} us")

            if args.compare_init_routing:
                num_composite_groups = args.num_experts * args.max_loras
                composite_ids = case["group_ids"].view(-1, 1)

                def init_routing_fn(
                    tokens=tokens,
                    composite_ids=composite_ids,
                    active_num=num_rows,
                    expert_num=num_composite_groups + 1,
                ):
                    return torch_npu.npu_moe_init_routing_v2(
                        tokens,
                        composite_ids,
                        active_num=active_num,
                        expert_num=expert_num,
                        expert_tokens_num_type=1,
                        expert_tokens_num_flag=True,
                        quant_mode=-1,
                    )

                grouped_inputs, reverse_mapping = permute_fn()
                routed_inputs, routed_mapping, routed_counts, _ = init_routing_fn()
                torch.npu.synchronize()
                torch.testing.assert_close(routed_inputs.cpu(), grouped_inputs.cpu(), rtol=0, atol=0)
                torch.testing.assert_close(
                    routed_counts[:num_composite_groups].to(torch.int64).cpu(),
                    case["composite_counts"].cpu(),
                    rtol=0,
                    atol=0,
                )
                restored = torch_npu.npu_moe_token_unpermute(
                    permuted_tokens=routed_inputs,
                    sorted_indices=routed_mapping,
                )
                torch.testing.assert_close(restored.cpu(), tokens.cpu(), rtol=0, atol=0)
                del reverse_mapping
                init_routing_us = _benchmark(
                    init_routing_fn,
                    warmup=max(5, args.warmup // 5),
                    iterations=max(20, args.iterations // 10),
                )
                print(
                    "  composite init-routing "
                    f"hidden={args.hidden_size}: {init_routing_us:9.3f} us "
                    f"vs fused+permute={fused_us + permute_us:9.3f} us"
                )


if __name__ == "__main__":
    main()
