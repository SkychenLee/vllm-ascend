#!/usr/bin/env python3
"""Precision tests for fused MoE-LoRA routing and int32 BGMV indices."""

from __future__ import annotations

import glob
import os
from typing import Any

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


SUPPORTED_DTYPES = (torch.bool, torch.int8)

# (category, description, tokens, top_k, num_experts, max_loras,
#  capacity_extra, boundary_mode)
ROUTING_CASES = (
    ("Decode", "1 token, DeepSeek top-k", 1, 6, 160, 16, 0, None),
    ("Decode", "2 tokens", 2, 6, 160, 16, 0, None),
    ("Decode", "8 tokens", 8, 6, 160, 16, 0, None),
    ("Decode", "32 tokens", 32, 6, 160, 16, 0, None),
    ("Prefill", "128 tokens", 128, 6, 160, 16, 0, None),
    ("Prefill", "512 tokens", 512, 6, 160, 16, 0, None),
    ("TopK", "top-k 1", 32, 1, 256, 4, 0, None),
    ("TopK", "top-k 2", 32, 2, 64, 8, 0, None),
    ("TopK", "top-k 8", 32, 8, 256, 16, 0, None),
    ("Experts", "small expert table", 16, 2, 8, 4, 0, None),
    ("Small", "single pair", 1, 1, 8, 1, 0, None),
    ("Small", "non-aligned 5 pairs", 5, 1, 8, 4, 0, None),
    ("Small", "non-aligned 18 pairs", 3, 6, 160, 4, 0, None),
    ("Large", "2048-token prefill", 2048, 6, 160, 16, 0, None),
    ("Large", "4096-token prefill", 4096, 6, 160, 16, 0, None),
    ("Capacity", "mapping capacity exceeds T", 17, 2, 64, 16, 15, None),
    ("Boundary", "all token LoRA indices are -1", 32, 6, 160, 16, 0, "all_no_lora"),
    ("Boundary", "all adapters disabled", 32, 6, 160, 16, 0, "all_disabled"),
    ("Boundary", "all adapters enabled", 32, 6, 160, 16, 0, "all_enabled"),
    ("Boundary", "maximum valid LoRA/expert IDs", 32, 6, 160, 16, 0, "max_ids"),
    ("Boundary", "all nonzero routing positions negative", 32, 6, 160, 16, 0, "negative_expanded"),
)

BGMV_CASES = tuple(
    (operation, dtype, index_dtype)
    for operation in ("shrink", "expand")
    for dtype in (torch.float16, torch.bfloat16)
    for index_dtype in (torch.int32, torch.int64)
)

GRAPH_CASE_IDS = (1, 15)

_OPS_LOADED = False


def load_ops() -> None:
    global _OPS_LOADED
    if _OPS_LOADED:
        return
    build_dir = os.environ.get("VLLM_ASCEND_BUILD_DIR")
    if not build_dir:
        raise RuntimeError("set VLLM_ASCEND_BUILD_DIR to the CMake build directory")
    torch.ops.load_library(
        os.path.join(build_dir, "lib", "libvllm_ascend_kernels.so")
    )
    modules = glob.glob(os.path.join(build_dir, "vllm_ascend_C*.so"))
    if len(modules) != 1:
        raise RuntimeError(f"expected one vllm_ascend_C module, got {modules}")
    torch.ops.load_library(modules[0])
    _OPS_LOADED = True


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def make_routing_inputs(
    case_id: int,
    case: tuple[Any, ...],
    adapter_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    _, _, tokens, top_k, num_experts, max_loras, capacity_extra, boundary = case
    generator = torch.Generator().manual_seed(20260822 + case_id)
    num_pairs = tokens * top_k
    expanded = torch.randperm(
        num_pairs, generator=generator, dtype=torch.int64
    ).to(torch.int32)
    if num_pairs > 1:
        negative = (torch.arange(num_pairs) % 5 == 1) & (expanded != 0)
        expanded[negative] = -expanded[negative]
    topk_ids = torch.randint(
        0,
        num_experts,
        (tokens, top_k),
        generator=generator,
        dtype=torch.int32,
    )
    token_lora = torch.randint(
        -1,
        max_loras,
        (tokens + capacity_extra,),
        generator=generator,
        dtype=torch.int64,
    )
    adapter_enabled = torch.randint(
        0,
        2,
        (max_loras,),
        generator=generator,
        dtype=torch.int8,
    ).to(adapter_dtype)

    if boundary == "all_no_lora":
        token_lora.fill_(-1)
    elif boundary == "all_disabled":
        adapter_enabled.zero_()
    elif boundary == "all_enabled":
        adapter_enabled.fill_(1)
    elif boundary == "max_ids":
        topk_ids.fill_(num_experts - 1)
        token_lora.fill_(max_loras - 1)
        adapter_enabled.fill_(1)
    elif boundary == "negative_expanded":
        expanded = -expanded.abs()

    return expanded, topk_ids, token_lora, adapter_enabled


def routing_reference(
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_indices: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    top_k = topk_ids.shape[1]
    lora_per_pair = token_lora_indices[: topk_ids.shape[0]].repeat_interleave(
        top_k
    )
    safe_lora = lora_per_pair.clamp(min=0)
    enabled = (lora_per_pair >= 0) & adapter_enabled[safe_lora].bool()
    per_original_pair = torch.where(
        enabled,
        safe_lora * num_experts + topk_ids.reshape(-1).to(torch.int64),
        torch.full_like(lora_per_pair, -1),
    ).to(torch.int32)
    expected = torch.empty_like(per_original_pair)
    expected[expanded_row_idx.abs().to(torch.int64)] = per_original_pair
    return expected


def compute_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.to(torch.float32)
    expected_f = expected.to(torch.float32)
    abs_err = (actual_f - expected_f).abs()
    rel_err = abs_err / (expected_f.abs() + 1e-7)
    return {
        "max_abs_err": abs_err.max().item() if abs_err.numel() else 0.0,
        "mean_abs_err": abs_err.mean().item() if abs_err.numel() else 0.0,
        "MARE": rel_err.max().item() if rel_err.numel() else 0.0,
        "MERE": rel_err.mean().item() if rel_err.numel() else 0.0,
    }


def evaluate_routing_case(
    case_id: int,
    case: tuple[Any, ...],
    adapter_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    category, description, tokens, top_k, num_experts, _, _, boundary = case
    cpu_inputs = make_routing_inputs(case_id, case, adapter_dtype)
    expected = routing_reference(*cpu_inputs, num_experts)
    npu_inputs = tuple(value.to(device) for value in cpu_inputs)
    actual = torch.ops._C_ascend.moe_lora_build_combined_idx(
        *npu_inputs, num_experts
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    metrics = compute_metrics(actual_cpu, expected)
    mismatch_count = int((actual_cpu != expected).sum().item())
    passed = (
        actual_cpu.dtype == torch.int32
        and actual_cpu.is_contiguous()
        and tuple(actual_cpu.shape) == tuple(expected.shape)
        and mismatch_count == 0
    )
    return {
        "suite": "routing",
        "case_id": case_id,
        "category": category,
        "description": description,
        "shape": [tokens, top_k],
        "dtype": dtype_name(adapter_dtype),
        "boundary": boundary,
        "numel": expected.numel(),
        "mismatch_count": mismatch_count,
        **metrics,
        "passed": passed,
    }


def _make_bgmv_inputs(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260822)
    batch, slots, rank, hidden, output = 7, 4, 8, 128, 128
    return {
        "index": torch.tensor([-1, 0, 3, 1, -1, 2, 0]),
        "x": torch.randn(batch, hidden, generator=generator, dtype=dtype),
        "a": torch.randn(slots, rank, hidden, generator=generator, dtype=dtype),
        "delta": torch.randn(batch, rank, generator=generator, dtype=torch.float32),
        "b": torch.randn(slots, output, rank, generator=generator, dtype=dtype),
        "base": torch.randn(batch, output, generator=generator, dtype=dtype),
    }


def _run_bgmv(
    operation: str,
    inputs: dict[str, torch.Tensor],
    index_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = inputs["index"].to(index_dtype).to(device)
    if operation == "shrink":
        actual = torch.zeros(
            inputs["x"].shape[0],
            inputs["a"].shape[1],
            dtype=torch.float32,
            device=device,
        )
        torch.ops._C_ascend.bgmv_shrink(
            inputs["x"].to(device),
            inputs["a"].to(device),
            index,
            actual,
            0.5,
        )
        expected = torch.zeros_like(actual, device="cpu")
        for row, slot in enumerate(inputs["index"].tolist()):
            if slot >= 0:
                expected[row] = torch.mv(
                    inputs["a"][slot].float(), inputs["x"][row].float()
                ) * 0.5
    else:
        actual = inputs["base"].to(device)
        actual = torch.ops._C_ascend.bgmv_expand(
            inputs["delta"].to(device),
            inputs["b"].to(device),
            index,
            actual,
            0,
            inputs["base"].shape[1],
        )
        expected = inputs["base"].clone()
        for row, slot in enumerate(inputs["index"].tolist()):
            if slot >= 0:
                expected[row] = (
                    inputs["base"][row].float()
                    + torch.mv(inputs["b"][slot].float(), inputs["delta"][row])
                ).to(inputs["base"].dtype)
    torch.npu.synchronize()
    return actual.cpu(), expected


def evaluate_bgmv_case(
    case_id: int,
    operation: str,
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    inputs = _make_bgmv_inputs(dtype)
    actual, expected = _run_bgmv(operation, inputs, index_dtype, device)
    int64_actual, _ = _run_bgmv(operation, inputs, torch.int64, device)
    metrics = compute_metrics(actual.float(), expected.float())
    parity_max_abs = (actual.float() - int64_actual.float()).abs().max().item()
    tolerance = 0.02 if operation == "shrink" or dtype == torch.float16 else 0.15
    reference_close = torch.allclose(
        actual.float(), expected.float(), rtol=tolerance, atol=tolerance
    )
    passed = parity_max_abs == 0.0 and reference_close
    return {
        "suite": "bgmv_index",
        "case_id": case_id,
        "category": operation,
        "description": f"{operation} int32/int64 index compatibility",
        "shape": list(actual.shape),
        "dtype": dtype_name(dtype),
        "index_dtype": dtype_name(index_dtype),
        "numel": actual.numel(),
        "index_parity_max_abs": parity_max_abs,
        "reference_rtol_atol": tolerance,
        **metrics,
        "passed": passed,
    }


def evaluate_graph_case(
    case_id: int,
    device: torch.device,
) -> dict[str, Any]:
    case = ROUTING_CASES[case_id - 1]
    category, description, tokens, top_k, num_experts, _, _, _ = case
    cpu_inputs = make_routing_inputs(case_id, case, torch.bool)
    expected = routing_reference(*cpu_inputs, num_experts)
    npu_inputs = tuple(value.to(device) for value in cpu_inputs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        output = torch.ops._C_ascend.moe_lora_build_combined_idx(
            *npu_inputs, num_experts
        )
    graph.replay()
    torch.npu.synchronize()
    actual = output.cpu()
    metrics = compute_metrics(actual, expected)
    mismatch_count = int((actual != expected).sum().item())
    return {
        "suite": "acl_graph",
        "case_id": case_id,
        "category": category,
        "description": description,
        "shape": [tokens, top_k],
        "dtype": "bool",
        "numel": actual.numel(),
        "mismatch_count": mismatch_count,
        **metrics,
        "passed": mismatch_count == 0,
    }


@pytest.fixture(scope="session")
def npu_device() -> torch.device:
    load_ops()
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    return device


@pytest.mark.parametrize("adapter_dtype", SUPPORTED_DTYPES, ids=dtype_name)
@pytest.mark.parametrize(
    "case_id,case",
    tuple(enumerate(ROUTING_CASES, start=1)),
    ids=[f"{case[0]}-{case[1]}" for case in ROUTING_CASES],
)
def test_routing_precision(
    npu_device: torch.device,
    case_id: int,
    case: tuple[Any, ...],
    adapter_dtype: torch.dtype,
) -> None:
    result = evaluate_routing_case(case_id, case, adapter_dtype, npu_device)
    assert result["passed"], result


@pytest.mark.parametrize(
    "case_id,operation,dtype,index_dtype",
    tuple((case_id, *case) for case_id, case in enumerate(BGMV_CASES, start=1)),
    ids=[
        f"{operation}-{dtype_name(dtype)}-{dtype_name(index_dtype)}"
        for operation, dtype, index_dtype in BGMV_CASES
    ],
)
def test_bgmv_index_precision(
    npu_device: torch.device,
    case_id: int,
    operation: str,
    dtype: torch.dtype,
    index_dtype: torch.dtype,
) -> None:
    result = evaluate_bgmv_case(
        case_id, operation, dtype, index_dtype, npu_device
    )
    assert result["passed"], result


@pytest.mark.parametrize("case_id", GRAPH_CASE_IDS, ids=("decode", "prefill"))
def test_acl_graph_precision(
    npu_device: torch.device,
    case_id: int,
) -> None:
    result = evaluate_graph_case(case_id, npu_device)
    assert result["passed"], result
