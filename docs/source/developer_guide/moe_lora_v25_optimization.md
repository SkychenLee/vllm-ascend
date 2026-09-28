# Experimental v0.25 MoE LoRA optimizations

This branch records work on W8A8 MoE inference with fully sharded LoRA.
The target workload uses TP8, rank16, three adapter slots, a maximum of
16384 batched tokens, eight concurrent 8192-input/4096-output requests,
model runner v1, and five D-Spark speculative tokens.

## Changes

- Vectorize large MoE routing recovery, preserving full-width INT64 values.
- Honor `VLLM_ASCEND_DISABLE_PIN_MEMORY` through the platform capability check.
- Add `moe_lora_combined_indices`, fusing adapter masking and expert/slot
  index construction. `VLLM_ASCEND_MOE_LORA_FUSED_ROUTING=1` enables the
  experimental path for up to 98304 routed rows. The default is off;
  unsupported inputs and older libraries use the existing tensor path.
- Add column tiling and a native capability marker for grouped W2 expansion.
  **This experiment was slower than the existing BGMV implementation in the
  measured target shapes. Keep `VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL=0`.**
  The implementation and tests are retained as an experimental checkpoint,
  not as a recommended performance setting.

An alternative whole-weight BF16 cache is a separate local prototype and is
not integrated into this branch. Neither the prototype nor binary build
products are included here.

## Validation and limits

Validation used CANN 9.0.1, PyTorch 2.10, torch_npu 2.10.post2, and an Ascend
910 A3 environment with vLLM 0.25.1. Native kernels and bindings compiled.

- Large routing recovery: 85 NPU system tests passed in the preceding work.
- Fused indices: 35 NPU precision cases and 12 changed-input graph replays
  passed. CPU/helper/host-Meta tests and the 98304/98305 dispatch boundary
  checks passed.
- Grouped W2: 123 NPU precision/graph tests passed, together with 149
  host-Meta and 36 CPU dispatch checks. Passing precision did not imply
  a performance improvement.

With all rows using an enabled adapter, isolated graph replay measurements
for index construction were 28.115 to 11.660 microseconds at 288 rows and
216.885 to 125.256 microseconds at 98304 rows. These are operator timings,
**not end-to-end TTFT or TPOT measurements**.

The new end-to-end baseline could not start after another workload occupied
the devices: available memory was approximately 37.74 GiB per device, below
the 56.99 GiB required by the unchanged memory-utilization setting of 0.93.
No new formal serving requests completed. End-to-end improvements have not
been established, and experimental settings remain disabled by default.

## Reproduction

Build and install matching Python and native code in the target environment.
The new kernel and binding files are included by the existing CMake source
discovery. Python-only updates do not install the native operator.

CPU dispatch tests can be run with:

```bash
python -m pytest --noconftest tests/ut/lora/test_combined_indices.py -q
```

The optional host-Meta cases require `MOE_LORA_COMBINED_TEST_LIBRARY` pointing
to an appropriate library; they skip if none is supplied. A host-only Meta
test library does not execute or validate the device kernel.

The native correctness, changed-input graph, and timing harness is:

```bash
python benchmarks/benchmark_moe_lora_combined_indices.py \
  --device 0 --library /path/to/matching/native-binding.so \
  --output fused-routing-results.json
```

Use an available device and a fresh output file. The harness tests the native
operator directly; Python dispatch coverage is provided separately. Before
accepting a serving configuration, compare warmed baseline and candidate
runs with the same workload, adapter, input seeds, cache state, and output
lengths. Record D-Spark acceptance and preemptions alongside TTFT and TPOT.
