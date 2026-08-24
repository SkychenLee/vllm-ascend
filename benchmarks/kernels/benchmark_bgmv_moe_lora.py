"""Microbenchmark the rank-16 BGMV shapes used by MoE LoRA decode."""

import argparse
import json
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op


def _benchmark(fn, warmup: int, iterations: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return min(samples), sum(samples) / len(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-loras", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    enable_custom_op()
    torch.npu.set_device(args.device)
    torch.manual_seed(7)
    dtype = getattr(torch, args.dtype)
    rank = 16
    indices = (torch.arange(args.batch_size, dtype=torch.int64) % args.num_loras).npu()
    results: dict[str, dict[str, float]] = {}

    top_k = 6
    num_tokens = max(1, args.batch_size // top_k)
    routing_rows = num_tokens * top_k
    expanded_row_idx = torch.randperm(routing_rows, dtype=torch.int32, device="npu")
    topk_ids = torch.randint(0, 256, (num_tokens, top_k), dtype=torch.int32, device="npu")
    token_lora_indices = (torch.arange(num_tokens, dtype=torch.int64) % args.num_loras).npu()
    adapter_enabled = torch.ones(args.num_loras, dtype=torch.bool, device="npu")
    combined_indices = torch.empty(routing_rows, dtype=torch.int64, device="npu")

    def argsort_routing():
        inverse = torch.argsort(torch.abs(expanded_row_idx))
        experts = topk_ids.reshape(-1)[inverse].to(torch.long)
        loras = token_lora_indices[(inverse // top_k).clamp_(max=token_lora_indices.numel() - 1)]
        return experts, loras

    def scatter_routing():
        destinations = torch.abs(expanded_row_idx).reshape(-1).to(torch.long)
        experts = torch.empty(routing_rows, dtype=torch.long, device="npu")
        experts.scatter_(0, destinations, topk_ids.reshape(-1).to(torch.long))
        original_loras = token_lora_indices[:num_tokens].view(-1, 1).expand(-1, top_k).reshape(-1)
        loras = torch.empty_like(original_loras)
        loras.scatter_(0, destinations, original_loras)
        return experts, loras

    def ascendc_routing():
        return torch.ops._C_ascend.moe_lora_routing(
            expanded_row_idx,
            topk_ids,
            token_lora_indices,
            adapter_enabled,
            combined_indices,
            top_k,
            256,
        )

    for name, fn in (
        ("argsort_routing", argsort_routing),
        ("scatter_routing", scatter_routing),
        ("ascendc_routing", ascendc_routing),
    ):
        best, mean = _benchmark(fn, args.warmup, args.iterations, args.repeats)
        results[name] = {"best_us": best, "mean_us": mean}

    for hidden_dim in (2048, 4096):
        x = torch.randn((args.batch_size, hidden_dim), dtype=dtype, device="npu")
        weight = torch.randn((args.num_loras, rank, hidden_dim), dtype=dtype, device="npu")
        output = torch.zeros((args.batch_size, rank), dtype=torch.float32, device="npu")
        best, mean = _benchmark(
            lambda x=x, weight=weight, output=output: torch.ops._C_ascend.bgmv_shrink(x, weight, indices, output, 1.0),
            args.warmup,
            args.iterations,
            args.repeats,
        )
        results[f"shrink_b{args.batch_size}_h{hidden_dim}"] = {
            "best_us": best,
            "mean_us": mean,
        }

    for output_dim in (2048, 4096):
        x = torch.randn((args.batch_size, rank), dtype=torch.float32, device="npu")
        weight = torch.randn((args.num_loras, output_dim, rank), dtype=dtype, device="npu")
        output = torch.zeros((args.batch_size, output_dim), dtype=dtype, device="npu")
        best, mean = _benchmark(
            lambda x=x, weight=weight, output=output, output_dim=output_dim: torch.ops._C_ascend.bgmv_expand(
                x, weight, indices, output, 0, output_dim
            ),
            args.warmup,
            args.iterations,
            args.repeats,
        )
        results[f"expand_b{args.batch_size}_h{output_dim}"] = {
            "best_us": best,
            "mean_us": mean,
        }

    hidden_dim, output_slice = 4096, 2048
    x = torch.randn((args.batch_size, hidden_dim), dtype=dtype, device="npu")
    a0 = torch.randn((args.num_loras, rank, hidden_dim), dtype=dtype, device="npu") * 0.01
    a1 = torch.randn_like(a0) * 0.01
    b0 = torch.randn((args.num_loras, output_slice, rank), dtype=dtype, device="npu") * 0.01
    b1 = torch.randn_like(b0) * 0.01
    legacy_workspace = torch.empty((2, args.batch_size, rank), dtype=torch.float32, device="npu")
    fused_workspace = torch.empty_like(legacy_workspace)
    legacy_output = torch.zeros((args.batch_size, output_slice * 2), dtype=dtype, device="npu")
    fused_output = torch.zeros_like(legacy_output)

    def legacy_w13():
        torch.ops._C_ascend.bgmv_shrink(x, a0, indices, legacy_workspace[0], 1.0)
        torch.ops._C_ascend.bgmv_shrink(x, a1, indices, legacy_workspace[1], 1.0)
        torch.ops._C_ascend.bgmv_expand(legacy_workspace[0], b0, indices, legacy_output, 0, output_slice)
        torch.ops._C_ascend.bgmv_expand(legacy_workspace[1], b1, indices, legacy_output, output_slice, output_slice)

    best, mean = _benchmark(legacy_w13, args.warmup, args.iterations, args.repeats)
    results["legacy_w13_four_launches"] = {"best_us": best, "mean_us": mean}

    best, mean = _benchmark(
        lambda: torch.ops._C_ascend.bgmv_moe_w13(
            x,
            a0,
            a1,
            b0,
            b1,
            indices,
            fused_workspace,
            fused_output,
            0,
            1.0,
        ),
        args.warmup,
        args.iterations,
        args.repeats,
    )
    results["fused_w13_two_launches"] = {"best_us": best, "mean_us": mean}

    # W13 has two independent LoRA slices and W2 has one.  This is the six-op
    # BGMV portion of add_lora_fused_moe for H=4096 and I=2048.
    results["projected_w13_w2_bgmv_chain"] = {
        metric: (
            2 * results[f"shrink_b{args.batch_size}_h4096"][metric]
            + 2 * results[f"expand_b{args.batch_size}_h2048"][metric]
            + results[f"shrink_b{args.batch_size}_h2048"][metric]
            + results[f"expand_b{args.batch_size}_h4096"][metric]
        )
        for metric in ("best_us", "mean_us")
    }
    results["projected_fused_w13_w2_bgmv_chain"] = {
        metric: (
            results["fused_w13_two_launches"][metric]
            + results[f"shrink_b{args.batch_size}_h2048"][metric]
            + results[f"expand_b{args.batch_size}_h4096"][metric]
        )
        for metric in ("best_us", "mean_us")
    }
    results["projected_chain_speedup_percent"] = {
        metric: 100.0
        * (1.0 - results["projected_fused_w13_w2_bgmv_chain"][metric] / results["projected_w13_w2_bgmv_chain"][metric])
        for metric in ("best_us", "mean_us")
    }
    results["projected_routing_plus_bgmv_speedup_percent"] = {
        metric: 100.0
        * (
            1.0
            - (results["ascendc_routing"][metric] + results["projected_fused_w13_w2_bgmv_chain"][metric])
            / (results["argsort_routing"][metric] + results["projected_w13_w2_bgmv_chain"][metric])
        )
        for metric in ("best_us", "mean_us")
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
