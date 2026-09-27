# SPDX-License-Identifier: Apache-2.0
"""Compare only routing recovery inside the TP MoE layer benchmark.

The grouped Prefill setting stays fixed for both variants. The original
recovery is a diagnostic C++ copy of the established sort chain; the candidate
is the installed production operator with its real shape/resource dispatch.
Outside the large-row interval both variants use the same production operator;
their timing difference is measurement noise, not a second-item speedup.
All layer setup, HCCL, warmup, precision, graph updates and alternating Event
measurements reuse moe_lora_grouped_prefill_tp.py.

Example (the diagnostic library's dependencies must be discoverable):
  ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 torchrun --nnodes=1 \
      --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29678 \
      benchmarks/moe_lora_recover_large_tp.py \
      --baseline-library /path/to/librecover_diagnostic.so \
      --grouped-prefill 1 --tokens 8 8192 --shape deepseek \
      --adapters single mixed --output-dir /tmp/recover_tp8

Includes routing recovery, independent expert/adapter grouping, both base
GMMs and low-rank HCCL. Initial dispatch, final combine and output AllReduce
remain outside the measurement. These are layer results, not TTFT or TPOT.
"""

import argparse
import hashlib
import sys
from functools import partial
from pathlib import Path

import moe_lora_grouped_prefill_tp as layer
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--grouped-prefill", type=int, choices=[0, 1], default=1)
    parser.add_argument("--large-min-rows", type=int, default=8161)
    parser.add_argument("--large-max-rows", type=int, default=262144)
    options, remaining = parser.parse_known_args()
    if not options.baseline_library.is_file():
        parser.error("the diagnostic original-recovery library does not exist")
    if not 0 < options.large_min_rows <= options.large_max_rows:
        parser.error("invalid large recovery row interval")
    torch.ops.load_library(str(options.baseline_library.resolve()))
    assert layer.enable_custom_op()
    native_recover = torch.ops._C_ascend.moe_lora_recover
    original_recover = torch.ops.recover_diagnostic.original
    original_grouped_setting = layer.grouped.ENABLE_GROUPED_PREFILL
    original_selector, original_run_case = layer.set_grouped, layer.run_case
    original_argv = sys.argv
    loaded_paths = {
        Path(line.split()[-1])
        for line in Path("/proc/self/maps").read_text().splitlines()
        if ".so" in line and ("vllm_ascend" in line or "recover_diagnostic" in line)
    }
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(loaded_paths)}

    def dispatch(expanded, topk, slots, top_k, *, candidate):
        changed_shape = options.large_min_rows <= expanded.numel() <= options.large_max_rows
        # Outside the candidate range, prior production already has small and
        # midsize native kernels. Comparing those with generic sorting would
        # incorrectly attribute an existing Decode optimization to this one.
        operator = native_recover if candidate or not changed_shape else original_recover
        return operator(expanded, topk, slots, top_k)

    variants = {enabled: partial(dispatch, candidate=enabled) for enabled in (False, True)}

    def select_recovery(candidate):
        # Benchmark-local operator selection happens before graph capture;
        # eager calls include the same selection overhead for both variants.
        layer.grouped.ENABLE_GROUPED_PREFILL = bool(options.grouped_prefill)
        torch.ops._C_ascend.moe_lora_recover = variants[candidate]

    def run_case(*args, **kwargs):
        result = original_run_case(*args, **kwargs)
        names = {"legacy": "original_dispatch", "grouped": "native_dispatch"}
        result["comparison"] = "large_recovery_only"
        result["grouped_prefill_enabled"] = bool(options.grouped_prefill)
        result["grouped_prefill_path_selected"] = result.pop("grouped_path_selected")
        result["order"] = [[names[name] for name in order] for order in result["order"]]
        result["measurements"] = {names[name]: value for name, value in result["measurements"].items()}
        result["loaded_libraries_sha256"] = hashes
        result["baseline_operator"] = "recover_diagnostic::original"
        result["candidate_operator"] = "_C_ascend::moe_lora_recover"
        result["matched_native_launch_paths"] = True
        result["large_recovery_interval"] = [options.large_min_rows, options.large_max_rows]
        result["same_recovery_for_both_variants"] = not (
            options.large_min_rows <= result["routed_rows"] <= options.large_max_rows
        )
        if result["same_recovery_for_both_variants"]:
            result["baseline_operator"] = result["candidate_operator"]
        return result

    try:
        layer.set_grouped = select_recovery
        layer.run_case = run_case
        sys.argv = [sys.argv[0], *remaining]
        layer.main()
    finally:
        torch.ops._C_ascend.moe_lora_recover = native_recover
        layer.grouped.ENABLE_GROUPED_PREFILL = original_grouped_setting
        layer.set_grouped, layer.run_case = original_selector, original_run_case
        sys.argv = original_argv


if __name__ == "__main__":
    main()
