"""TileLang prototype for ``moe_lora_build_combined_idx``.

This file is intentionally shape-specialized.  It validates the direct-scatter
algorithm and provides the AscendC source used as the starting point for the
production direct kernel.
"""

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


PASS_CONFIGS = {}


def build_program(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_loras: int,
    num_cores: int,
):
    num_pairs = num_tokens * top_k
    del num_cores

    @T.prim_func
    def main(
        expanded_row_idx: T.Tensor((num_pairs,), "int32"),
        topk_ids: T.Tensor((num_pairs,), "int32"),
        token_lora_indices: T.Tensor((num_tokens,), "int64"),
        adapter_enabled: T.Tensor((max_loras,), "int32"),
        combined_idx: T.Tensor((num_pairs,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            expanded_ub = T.alloc_ub((num_pairs,), "int32")
            expert_ub = T.alloc_ub((num_pairs,), "int32")
            lora_ub = T.alloc_ub((num_tokens,), "int64")
            enabled_ub = T.alloc_ub((max_loras,), "int32")
            output_ub = T.alloc_ub((num_pairs,), "int32")

            with T.Scope("V"):
                if vid == 0:
                    T.copy(expanded_row_idx, expanded_ub)
                    T.copy(topk_ids, expert_ub)
                    T.copy(token_lora_indices, lora_ub)
                    T.copy(adapter_enabled, enabled_ub)
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)

                    for pair in T.serial(num_pairs):
                        if expanded_ub[pair] < 0:
                            expanded_ub[pair] = -expanded_ub[pair]

                        lora = lora_ub[pair // top_k]
                        if lora >= 0 and enabled_ub[T.Cast("int32", lora)] != 0:
                            output_ub[expanded_ub[pair]] = T.Cast(
                                "int32",
                                lora * num_experts + expert_ub[pair],
                            )
                        else:
                            output_ub[expanded_ub[pair]] = -1

                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.copy(output_ub, combined_idx)

    return main


def compile_kernel(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_loras: int,
    num_cores: int,
    target: str,
):
    ub_bytes = (
        num_tokens * top_k * 12
        + num_tokens * 8
        + max_loras * 4
    )
    if ub_bytes > 160 * 1024:
        raise ValueError(
            f"TileLang decode prototype needs {ub_bytes} UB bytes; "
            "use the AscendC multi-core implementation for this shape"
        )
    program = build_program(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        max_loras=max_loras,
        num_cores=num_cores,
    )
    return tilelang.compile(
        program,
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target=target,
    )


def reference(
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_indices: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    top_k = topk_ids.shape[1]
    inv_perm = torch.argsort(torch.abs(expanded_row_idx))
    expert_per_row = topk_ids.reshape(-1)[inv_perm].to(torch.int64)
    orig_token = (inv_perm // top_k).clamp_(
        max=token_lora_indices.numel() - 1
    )
    lora_per_row = token_lora_indices[orig_token]
    safe_lora = lora_per_row.clamp(min=0)
    enabled = (lora_per_row >= 0) & adapter_enabled[safe_lora].bool()
    return torch.where(
        enabled,
        safe_lora * num_experts + expert_per_row,
        torch.full_like(lora_per_row, -1),
    ).to(torch.int32).contiguous()


def make_inputs(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_loras: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260822 + num_tokens + top_k)
    num_pairs = num_tokens * top_k
    expanded = torch.randperm(
        num_pairs, generator=generator, dtype=torch.int64
    ).to(torch.int32)
    if num_pairs > 1:
        negative_mask = (torch.arange(num_pairs) % 2 == 1) & (expanded != 0)
        expanded[negative_mask] = -expanded[negative_mask]
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        generator=generator,
        dtype=torch.int32,
    )
    token_lora = torch.randint(
        -1,
        max_loras,
        (num_tokens,),
        generator=generator,
        dtype=torch.int64,
    )
    adapter_enabled = torch.randint(
        0,
        2,
        (max_loras,),
        generator=generator,
        dtype=torch.int32,
    )
    return tuple(
        tensor.to(device)
        for tensor in (expanded, topk_ids, token_lora, adapter_enabled)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=160)
    parser.add_argument("--max-loras", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=8)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--target", choices=("ascendc", "pto"), default="ascendc")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()

    if args.tokens <= 0 or args.top_k <= 0:
        raise ValueError("tokens and top-k must be positive")
    num_pairs = args.tokens * args.top_k
    num_cores = min(args.num_cores, (num_pairs + 1) // 2)

    device = torch.device(args.device)
    torch.npu.set_device(device)
    kernel = compile_kernel(
        num_tokens=args.tokens,
        top_k=args.top_k,
        num_experts=args.num_experts,
        max_loras=args.max_loras,
        num_cores=num_cores,
        target=args.target,
    )
    source = kernel.get_kernel_source()
    if args.export is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(source, encoding="utf-8")

    expanded, topk_ids, token_lora, adapter_enabled = make_inputs(
        args.tokens,
        args.top_k,
        args.num_experts,
        args.max_loras,
        device,
    )
    actual = kernel(
        expanded,
        topk_ids.reshape(-1),
        token_lora,
        adapter_enabled,
    )
    expected = reference(
        expanded,
        topk_ids,
        token_lora,
        adapter_enabled,
        args.num_experts,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)

    print(
        f"PASS target={args.target} tokens={args.tokens} top_k={args.top_k} "
        f"pairs={num_pairs} cores={num_cores}"
    )


if __name__ == "__main__":
    main()
