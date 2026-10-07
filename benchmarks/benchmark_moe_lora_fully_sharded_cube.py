# SPDX-License-Identifier: Apache-2.0
"""Compare native Vector A projections with the bounded fully-sharded Cube path.

Assign an idle physical device through ASCEND_RT_VISIBLE_DEVICES and pass its
logical --device explicitly. Timings include weight packing, casts, prefix
counts, scale and masking. This benchmark does not measure TP communication.
"""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.lora.fully_sharded_cube import CubePrefillRouting, shrink_cube_prefill


def make_case(rows, pair, slot, pattern, device):
    experts, hidden, rank = 256, (4096 if pair else 256), (2 if pair else 16)
    counts = torch.zeros(experts, dtype=torch.int64)
    groups = experts if pattern == "balanced" else (17 if pattern == "sparse" else 1)
    counts[:groups] = rows // groups
    counts[: rows % groups] += 1
    expert = torch.repeat_interleave(torch.arange(experts), counts)
    indices = expert + slot * experts
    indices[::11] = -1
    x = torch.randint(-127, 128, (rows, hidden), dtype=torch.int8)
    weights = (torch.randn(2 if pair else 1, 3, experts, rank, hidden) * 0.01).bfloat16()
    scales = torch.rand(rows) * 0.01 + 0.001
    scales[::11] = float("nan")
    return {
        "x": x.to(device),
        "weights": weights.to(device),
        "indices": indices.to(device),
        "scales": scales.to(device),
        "counts": counts.to(device),
        "expert_cpu": expert,
        "pair": pair,
        "slot": slot,
    }


def vector(case, output):
    x, weights = case["x"], case["weights"]
    if case["pair"]:
        torch.ops._C_ascend.bgmv_shrink_int8_pair(
            x, weights.view(2, -1, 2, x.shape[1]), case["indices"], case["scales"], output
        )
    else:
        torch.ops._C_ascend.bgmv_shrink_int8(
            x, weights[0].view(-1, 16, x.shape[1]), case["indices"], case["scales"], output
        )
    return output


def candidate(case, output):
    routing = CubePrefillRouting(case["counts"].cumsum(0), case["slot"], case["x"].shape[0])
    shrink_cube_prefill(case["x"], case["weights"].unbind(0), case["indices"], case["scales"], output, routing)
    return output


def timing(fn, repeats):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(3):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return statistics.median(samples)


def validate(case):
    rows = case["x"].shape[0]
    width = 4 if case["pair"] else 16
    old = torch.empty((rows, width), dtype=torch.float32, device=case["x"].device)
    new = torch.empty_like(old)
    reference = vector(case, old).cpu()
    actual = candidate(case, new).cpu()
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=2e-5)
    assert torch.equal(actual[::11], torch.zeros_like(actual[::11]))
    # Independent CPU equations also check output packing and slot selection.
    chosen = torch.linspace(0, rows - 1, min(rows, 33)).long()
    x = case["x"][chosen.to(case["x"].device)].cpu().float()
    scales = case["scales"][chosen.to(case["x"].device)].cpu()
    weights = case["weights"][:, case["slot"]].cpu().float()
    expected = torch.zeros(len(chosen), width)
    for index, row in enumerate(chosen.tolist()):
        if row % 11:
            expert = case["expert_cpu"][row]
            expected[index] = torch.cat([w[expert] @ x[index] for w in weights]) * scales[index]
    torch.testing.assert_close(actual[chosen], expected, atol=3e-6, rtol=2e-5)
    return (
        old,
        new,
        {
            "max_abs": (actual - reference).abs().max().item(),
            "relative_l2": ((actual - reference).norm() / reference.norm().clamp_min(1e-30)).item(),
            "cpu_reference_rows": len(chosen),
            "masked_rows_exact": True,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[288, 16384, 24576, 98304])
    parser.add_argument(
        "--patterns", nargs="+", choices=["balanced", "sparse", "skewed"], default=["balanced", "sparse"]
    )
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.npu.set_device(args.device)
    torch.manual_seed(20261007)
    device = torch.device(f"npu:{args.device}")
    result = {"library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(), "cases": []}
    with torch.inference_mode():
        for rows in args.rows:
            for pattern in args.patterns:
                for pair in (True, False):
                    case = make_case(rows, pair, 1, pattern, device)
                    old, new, metrics = validate(case)
                    metrics.update(
                        {
                            "rows": rows,
                            "projection": "W13" if pair else "W2",
                            "pattern": pattern,
                            "vector_us": timing(partial(vector, case, old), args.repeats),
                            "cube_us": timing(partial(candidate, case, new), args.repeats),
                        }
                    )
                    metrics["speedup"] = metrics["vector_us"] / metrics["cube_us"]
                    result["cases"].append(metrics)
                    print(json.dumps(metrics), flush=True)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    del case, old, new
        result["status"] = "PASS"
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
