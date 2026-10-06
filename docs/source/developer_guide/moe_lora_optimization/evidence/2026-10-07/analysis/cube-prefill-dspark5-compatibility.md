# Cube W13 Prefill compatibility with V1 DSpark5

Initial read-only review followed by an authorized source-only candidate,
2026-10-07. No baseline runtime source was installed and no NPU experiment
was started. Paths refer to host `vllm_ascend` and the container's installed
upstream `vllm`. Existing exclusions below describe the pre-candidate baseline.

## Existing candidate and current exclusions

`lora/cube_moe.py:31-60` currently allows no speculative configuration.
Its compiled-mode exception requires FULL_DECODE_ONLY capture sizes<=8;
otherwise it requires enforce_eager. DSpark5 with capture sizes<=48 thus
fails both the no-spec and small-capture guards. Enabling its flag alone
does not create a Cube state on this run.

State is created only by an actual LoRA MoE wrapper with W8A8 base weights
(`lora/fused_moe.py:479-493`). All existing TP8/EP8, DP1/PCP1/DCP1/PP1,
max requests8, max batch tokens16384, AllGather, no shared-expert overlap,
non-fully-sharded, two W13 slices, H4096/E32/K6, BF16/rank16 limits matter.
Persistent A-cache generation is invalidated on adapter reset/load and
loaded slots are marked only after successful weight installation.

The runtime preparation guard (`cube_moe.py:185-195`) rejects before
counts readback unless there is a loaded single slot, BF16 contiguous
`[R,4096]`, 6144<=R<=98304, 32 INT64 expert counts in count mode1, and
the expected device/stream. `counts.cpu().tolist()` occurs only after
these conditions (198). The tile plan also requires total local active
rows>=512, valid32-expert counts and complete bounded tiling. Both A slices
are packed to `[32,4096,32]`; tile BMM has BF16 inputs and FP32 output.
This still accelerates W13 double-A only, not B expansion, W2 or base GMM.

## Target/Draft separation

Target LoRA wrapping happens before draft loading:
container upstream `vllm/v1/worker/gpu_model_runner.py:5228-5235` calls
`self.load_lora_model(self.model, ...)` then `self.drafter.load_model(self.model)`.
The V1 proposer independently calls generic `get_model` to load DSpark
(`spec_decode/llm_base_proposer.py:265-290`), without loading a second
LoRA manager or calling `create_lora_weights` on the draft. Adapter weight
prefixes correspond to target expert layers. DSpark draft MoE prefixes are
`mtp.{idx}` (`models/deepseek_v4_dspark.py:109-116`), with its own model and
forward. Passing the same VllmConfig to a draft loader is not evidence
that draft layers received Cube state.

Consequently the current DSpark draft has no CubeW13State. Even if a
future change wrapped it, its query block has5*C tokens when sampling from
the anchor, or6*C with a bonus anchor (`dspark_proposer.py:37-44`): T40/M240
or T48/M288 at C8. Both are far below the6144-row threshold. DSpark's
large prompt context is projected into K/V by `_project_shared_kv`; it
does not mean that MoE query dispatch has those prompt-length rows.

Recommendation: keep a direct `not ctx.is_draft_model` runtime guard before
Cube prepare in any DSpark expansion. This makes the intended target-only
scope explicit even if future LoRA draft loading or unusual batches change.

## Graph and compilation safety

V1 genuine target prefill is forced to runtime NONE by
`model_runner_v1.py:2695-2704,2727-2728`; request computed token counts
below prompt token counts identify it. The force is not based on Punica's
is_prefill field. With FULL_DECODE_ONLY, target verification graphs retain
their configured full decode mode; DSpark5 targets are T<=48/M<=288 at
concurrency<=8 (before exceptional padding), well below6144 regardless.

The Cube caller additionally requires runtime NONE, profile flagFalse,
and non-None attention metadata (`lora/quant_moe.py:253-265`). Runtime
FULL/PIECEWISE and normal profile runs therefore never call prepare or
read counts. `set_ascend_forward_context` provides `in_profile_run` and
`is_draft_model` (`ascend_forward_context.py:147,198`). Draft contexts
explicitly set `is_draft_model=True` (`llm_base_proposer.py:977-985`).

Runtime NONE alone does not generally mean torch.compile is disabled.
There is nevertheless a relevant opaque boundary in this concrete stack:
upstream `model_executor/layers/fused_moe/runner/moe_runner.py:193-209`
registers `moe_forward`/`moe_forward_shared` with fake implementations,
300-311 selects `torch.ops.vllm` entries on NPU, and685-694 invokes them.
Ascend's override `_forward_impl` ultimately calls its own comm/quant MLP
(`ops/fused_moe/fused_moe.py:876-887`). Thus FakeTensor/AOT tracing sees
the opaque MoE operation and its fake output; it does not execute Cube's
Python count readback. Real execution under NONE invokes the operator
outside NPU graph capture. Real execution under FULL sees the runtime
guard and small rows before any CPU readback.

This safety argument depends on the current opaque MoE boundary. If an
experimental path traces quant_apply_mlp directly or decomposes the custom
op, a separate `not torch.compiler.is_compiling()`/capture guard is needed.
Do not assume a generic direct torch.compile of Cube.prepare is supported.

Profile flagTrue and graph runtime modes are explicitly excluded. Normal
dummy runs with no attention metadata also fall back. However, the current
three caller conditions do not establish that *all* large dummy/warmup
forwards are excluded: `_dummy_run` can request force_attention while
is_profile=False and runtime NONE. Such a run can have real attention
metadata and dummy LoRA weights marked loaded. It may use Cube without
capturing a graph; that can affect warmup/accounting rather than introduce
a CPU-readback graph capture. Do not describe this as universal dummy
exclusion. Trace `prepare_accept` logs during engine startup to check it.

## Minimal static expansion recommendation

Keep the existing no-spec branch as-is. Permit one additional branch only
when normalized spec exists, `use_dspark()` isTrue, num_speculative_tokens=5,
and official `VLLM_USE_V2_MODEL_RUNNER is False` (explicit0). Unset envNone
can auto-select V2 and must remain unsupported.

For that branch, the existing FULL_DECODE_ONLY exception may permit positive
capture sizes<=48 instead of<=8. Keep all other static shape/backend/parallel
limits. Preserve all existing dynamic conditions, occupancy thresholds,
cache generation and stream checks. Add an explicit target-only runtime
guard if adopting the recommendation above; no draft cache allocation is
intended.

At R=6*T, the runtime R threshold6144 corresponds to at least1024 target
tokens, not6144 target tokens. Long8K prefill chunks are candidates, while
DSpark target decode and draft query blocks fall back before readback.

This is a source-level compatibility recommendation, not a performance
result or authorization to enable the path by default. Measure genuine
long target Prefill with counts readback, metadata transfer, packing and
scratch overhead included. Compare same adapter, shapes, KV/cache state,
accepted-token counts and output lengths; preserve default flag0 until
the full-model profile shows a benefit. Multi-adapter/mixed base-token
batches continue to fall back under the single-loaded-slot contract.

## Implemented source-only candidate and CPU/fake validation

Following review, implemented the narrow static extension in
`lora/cube_moe.py`. Explicit V1 DSpark5 requires FULL_DECODE_ONLY and
capture sizes1..48, including when enforce_eager isTrue; no-spec retains
cap8 and its existing unrestricted enforce_eager exception. Other speculative methods,
different DSpark block counts, explicit V2 and automatic runner selection
are rejected. All other static and dynamic limits are retained.

Added `supports_cube_target_prefill_context` and called it from the actual
`lora/quant_moe.py` entry. It requires runtime NONE, profile flag exactlyFalse,
non-None metadata, and draft flag exactlyFalse. None contexts, missing
fields or None flags fail conservatively. There is no device metadata read
to decide execution mode. No-spec and DSpark target paths share this check;
FULL/PIECEWISE/profile/draft cannot invoke prepare through this entry.

New `tests/ut/lora/test_cube_dspark.py` passed47 cases, including explicit
V1 DSpark5 cap48, unchanged no-spec eager behavior, all important fallback
modes, other static limits, actual quant-MLP entry invocation/noninvocation,
R240/R288/R6143 fallback before counts.cpu, R6144 CPU plan/metadata-copy
positive path, and flag0 wrapper allocation absence. Positive fake-config
and CPU metadata tests prove control/data flow, not NPU BMM performance.

The registered upstream `torch.ops.vllm.moe_forward` was also invoked under
FakeTensorMode with Cube.prepare and the real layer lookup replaced by
failure sentinels. The fake returned the correct shape/dtype without
executing either sentinel. This validates the existing opaque boundary;
it is not a replacement operator created for testing.

Isolated pytest without conftest requires importing `vllm_ascend.ops` before
loading this file's shared quant-test helpers; the repository conftest
normally arranges initialization. The successful CPU command used that
bootstrap and disabled plugin autoload. It did not install the candidate.
Ruff check/format and diff whitespace checks passed for the candidate files.
The original host Cube file had formatting differences from the earlier
publication worktree; formatting touched existing text without changing
its unaffected arithmetic/cache/tiling behavior.
