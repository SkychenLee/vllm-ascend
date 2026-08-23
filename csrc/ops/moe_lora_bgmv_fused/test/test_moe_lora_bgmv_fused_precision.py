#!/usr/bin/env python3
"""Precision tests for the fused MoE LoRA BGMV operator."""

from __future__ import annotations

import glob
import os
from typing import Any

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
THRESHOLD = {
    torch.float16: 2**-10,
    torch.bfloat16: 2**-7,
}

# category, description, M, H, O, Y, slice offset, index dtype,
# index pattern, scale, optional boundary mode
PRECISION_CASES = (
    ("Grouped", "single 4-row group", 4, 64, 64, 64, 0, torch.int32, "group4_same", 1.0, None),
    ("Grouped", "two groups with slice", 8, 128, 96, 160, 32, torch.int64, "group4_same", 0.5, None),
    ("MoE", "small W2-like", 16, 512, 512, 512, 0, torch.int32, "expert_sorted", 1.0, None),
    ("MoE", "Qwen W13 dimensions", 32, 2048, 1024, 1024, 0, torch.int32, "expert_sorted", 1.0, None),
    ("MoE", "Qwen W2 dimensions", 32, 512, 2048, 2048, 0, torch.int32, "expert_sorted", 1.0, None),
    ("Mixed", "mixed adapters inside group", 64, 768, 1000, 1100, 64, torch.int64, "mixed_within4", 0.25, None),
    ("Grouped", "128 W13 rows", 128, 2048, 1024, 1024, 0, torch.int32, "expert_sorted", 1.0, None),
    ("Fallback", "no consecutive equal index", 256, 512, 2048, 2048, 0, torch.int64, "alternating", 1.0, None),
    ("Small", "minimum supported dimensions", 1, 17, 17, 17, 0, torch.int32, "single", 1.0, None),
    ("Small", "unaligned H/O and slice", 3, 17, 19, 23, 2, torch.int64, "mixed_within4", 0.75, None),
    ("Boundary", "all rows disabled", 4, 31, 33, 40, 3, torch.int32, "all_negative", 1.0, None),
    ("Boundary", "tail row after one group", 5, 2048, 2048, 2048, 0, torch.int64, "group4_same", 1.0, None),
    ("Large", "integration threshold W13", 512, 2048, 1024, 1024, 0, torch.int32, "expert_sorted", 1.0, None),
    ("Large", "1024-row W2", 1024, 512, 2048, 2048, 0, torch.int32, "expert_sorted", 1.0, None),
    ("Large", "4096-row W13", 4096, 2048, 1024, 1024, 0, torch.int32, "expert_sorted", 1.0, None),
    ("Large", "8192-row W2", 8192, 512, 2048, 2048, 0, torch.int32, "expert_sorted", 1.0, None),
    ("DeepSeekV4", "TP8 W13 slice", 512, 4096, 256, 512, 256, torch.int32, "expert_sorted", 1.0, None),
    ("DeepSeekV4", "TP8 W2", 512, 256, 4096, 4096, 0, torch.int64, "expert_sorted", 1.0, None),
    ("DeepSeekV4", "TP1 W13 slice", 512, 4096, 2048, 4096, 2048, torch.int64, "expert_sorted", 1.0, None),
    ("DeepSeekV4", "TP1 W2", 512, 2048, 4096, 4096, 0, torch.int32, "expert_sorted", 1.0, None),
    ("Boundary", "all disabled indices", 9, 63, 65, 96, 7, torch.int32, "single", 1.0, "all_negative"),
    ("Boundary", "minimum valid weight", 9, 63, 65, 96, 7, torch.int64, "single", 1.0, "minimum_index"),
    ("Boundary", "maximum valid weight", 9, 63, 65, 96, 7, torch.int32, "single", 1.0, "maximum_index"),
    ("Boundary", "zero scale", 9, 63, 65, 96, 7, torch.int64, "expert_sorted", 0.0, None),
    ("Boundary", "negative scale", 9, 63, 65, 96, 7, torch.int32, "expert_sorted", -0.5, None),
    ("Boundary", "last valid slice", 9, 63, 33, 97, 64, torch.int64, "expert_sorted", 1.0, None),
)

_OPS_LOADED = False


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


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


def build_indices(rows: int, pattern: str, dtype: torch.dtype) -> torch.Tensor:
    if pattern == "all_negative":
        values = torch.full((rows,), -1, dtype=torch.int64)
    elif pattern == "group4_same":
        values = torch.arange((rows + 3) // 4).repeat_interleave(4)[:rows]
    elif pattern == "alternating":
        values = torch.arange(rows, dtype=torch.int64).remainder(4)
    elif pattern == "mixed_within4":
        values = torch.arange(rows, dtype=torch.int64).remainder(3)
        values[::5] = -1
    elif pattern == "expert_sorted":
        values = torch.arange((rows + 7) // 8).repeat_interleave(8)[:rows]
    else:
        values = torch.zeros(rows, dtype=torch.int64)
    return values.to(dtype)


def _positive_random(
    shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    return (torch.rand(shape, generator=generator) * 0.04 + 0.01).to(dtype)


def make_inputs(
    case_id: int,
    case: tuple[Any, ...],
    dtype: torch.dtype,
) -> dict[str, Any]:
    (
        _,
        _,
        rows,
        input_hidden,
        output_hidden,
        output_full,
        slice_offset,
        index_dtype,
        pattern,
        scale,
        boundary,
    ) = case
    indices = build_indices(rows, pattern, index_dtype)
    if boundary == "all_negative":
        indices.fill_(-1)
    num_weights = max(int(indices.max().item()) + 1, 1)
    if boundary == "minimum_index":
        num_weights = 5
        indices.zero_()
    elif boundary == "maximum_index":
        num_weights = 5
        indices.fill_(num_weights - 1)

    generator = torch.Generator().manual_seed(20260823 + case_id)
    rank = 16
    return {
        "x": _positive_random((rows, input_hidden), generator, dtype),
        "a": _positive_random(
            (num_weights, rank, input_hidden), generator, dtype
        ),
        "b": _positive_random(
            (num_weights, output_hidden, rank), generator, dtype
        ),
        "indices": indices,
        "y": (torch.rand((rows, output_full), generator=generator) * 0.1 + 1.0).to(dtype),
        "slice_offset": slice_offset,
        "slice_size": output_hidden,
        "scale": scale,
    }


def reference(inputs: dict[str, Any], chunk_rows: int = 64) -> torch.Tensor:
    x = inputs["x"]
    a = inputs["a"]
    b = inputs["b"]
    indices = inputs["indices"]
    expected = inputs["y"].clone()
    slice_offset = inputs["slice_offset"]
    slice_size = inputs["slice_size"]
    for begin in range(0, x.shape[0], chunk_rows):
        end = min(begin + chunk_rows, x.shape[0])
        index_chunk = indices[begin:end].to(torch.int64)
        valid = index_chunk >= 0
        safe_indices = index_chunk.clamp_min(0)
        selected_a = a[safe_indices].float()
        rank_out = torch.bmm(
            x[begin:end].float().unsqueeze(1), selected_a.transpose(1, 2)
        ).squeeze(1)
        rank_out.mul_(inputs["scale"])
        selected_b = b[safe_indices].float()
        delta = torch.bmm(
            rank_out.unsqueeze(1), selected_b.transpose(1, 2)
        ).squeeze(1)
        delta.masked_fill_(~valid.unsqueeze(1), 0.0)
        output_slice = expected[
            begin:end, slice_offset : slice_offset + slice_size
        ]
        output_slice.copy_((output_slice.float() + delta).to(expected.dtype))
    return expected


def compute_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_err = (actual_f - expected_f).abs()
    rel_err = abs_err / (expected_f.abs() + 1e-7)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten().unsqueeze(0), expected_f.flatten().unsqueeze(0)
    ).item()
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "MARE": rel_err.max().item(),
        "MERE": rel_err.mean().item(),
        "cosine_sim": cosine,
    }


def evaluate_case(
    case_id: int,
    case: tuple[Any, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    category, description = case[:2]
    inputs = make_inputs(case_id, case, dtype)
    expected = reference(inputs)
    y_npu = inputs["y"].to(device)
    returned = torch.ops._C_ascend.moe_lora_bgmv_fused(
        inputs["x"].to(device),
        inputs["a"].to(device),
        inputs["b"].to(device),
        inputs["indices"].to(device),
        y_npu,
        inputs["slice_offset"],
        inputs["slice_size"],
        inputs["scale"],
    )
    torch.npu.synchronize()
    actual = y_npu.cpu()
    metrics = compute_metrics(actual, expected)
    threshold = THRESHOLD[dtype]
    begin = inputs["slice_offset"]
    end = begin + inputs["slice_size"]
    outside_equal = True
    if begin:
        outside_equal &= torch.equal(actual[:, :begin], inputs["y"][:, :begin])
    if end < actual.shape[1]:
        outside_equal &= torch.equal(actual[:, end:], inputs["y"][:, end:])
    alias_preserved = returned.data_ptr() == y_npu.data_ptr()
    passed = (
        alias_preserved
        and outside_equal
        and metrics["MERE"] < threshold
        and metrics["MARE"] < 10 * threshold
    )
    return {
        "case_id": case_id,
        "category": category,
        "description": description,
        "shape": [case[2], case[3], case[4], case[5]],
        "dtype": dtype_name(dtype),
        "index_dtype": dtype_name(case[7]),
        "index_pattern": case[8],
        "slice_offset": case[6],
        "scale": case[9],
        "threshold": threshold,
        "alias_preserved": alias_preserved,
        "outside_slice_equal": outside_equal,
        **metrics,
        "passed": passed,
    }


@pytest.fixture(scope="session")
def npu_device() -> torch.device:
    load_ops()
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    return device


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=dtype_name)
@pytest.mark.parametrize(
    "case_id,case",
    tuple(enumerate(PRECISION_CASES, start=1)),
    ids=[f"{case[0]}-{case[1]}" for case in PRECISION_CASES],
)
def test_moe_lora_bgmv_fused_precision(
    npu_device: torch.device,
    case_id: int,
    case: tuple[Any, ...],
    dtype: torch.dtype,
) -> None:
    result = evaluate_case(case_id, case, dtype, npu_device)
    assert result["passed"], result
