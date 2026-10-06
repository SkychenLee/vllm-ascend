# Compiler cache isolation update

Source-only update, 2026-10-07. Files:
`compilation/compiler_interface.py`, `patch/platform/patch_compile_cache.py`,
and `tests/ut/compilation/test_compiler_hash.py`. Baseline runtime is unchanged.

Four optional graph-changing flags now enter both AscendCompiler's inner
hash and upstream env-based outer AOT factors: FUSED_ROUTING,
FUSED_ACTIVATION, FUSED_SHRINK_PAIR, and OPTIMIZE_MTP_BUFFER_COPY.
The existing GROUPED_PREFILL factor still hashes its effective import-time
constant rather than later environment mutations.

The MTP allocation/copy omission depends on method and explicit V1 as well
as its flag. Official `vllm.envs.compile_factors()` already hashes the known
VLLM_USE_V2_MODEL_RUNNER environment value; it is not ignored. No virtual
environment factor or additional runner patch was needed. Inner
AscendCompiler now also hashes the exact derived skip boolean, matching
the model's constructor decision without importing model kernels.

Upstream `SpeculativeConfig.compute_hash` hashes aux-output eligibility and
layer IDs, so DSpark and eagle3 with identical IDs otherwise collide at this
level. VllmConfig.compute_hash includes that method's result, which in turn
feeds the outer AOT key. The existing Ascend cache patch now wraps only
SpeculativeConfig.compute_hash: when OPTIMIZE_MTP_BUFFER_COPY is1, its
normalized method is hashed together with the original hash. Flag0 returns
the original hash byte-for-byte. Model/method selection semantics are not
modified. The wrapper captures the original callable in its closure, uses
functools.wraps, and marks the callable so reload/repeated application does
not nest wrappers or change hashes.

Tests include all16 combinations of the four flags at both levels;
persisted-cache toggle/miss/restore for each flag; same-aux DSpark/eagle3
original equality and optimized AOT separation; flag0 restoration to the
original hash; V1/V2/None derived-skip behavior; official0/1/None outer
factor isolation; and idempotent repeated patch/module reload.
23 cache unit tests passed. They use fake configs and no model weights or
NPU tensor execution. Model skip tests independently passed119 cases.
Final lint and combined validation are recorded by the implementing agent.

Final combined run of cache23, MTP119, Cube47 and associated quantization8
tests passed197/197. Ruff check/format passed on all eight owned source/test
files; diff whitespace checks passed. See
`source-only-candidate-manifest.json` and
`mtp-cache-cube-source-only-candidate.patch` for the reviewable file list and
diff against the unchanged installed baseline. The env-comment edit and the
separately owned new pair-flag definition are not duplicated in this patch.

Set flags and runner selection before constructing an engine. Allocation
decisions and captured graphs cannot be updated by changing environment
variables during a live engine run. The current benchmark script disables
compile cache; this patch protects later cache-enabled runs as well.
