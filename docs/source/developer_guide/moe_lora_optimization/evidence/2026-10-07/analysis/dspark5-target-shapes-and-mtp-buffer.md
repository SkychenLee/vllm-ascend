# DSpark5 target shapes and ordinary MTP buffer candidate

Reviewed 2026-10-07. Host source: `{repo_root}`.
Runtime baseline: container `ds-ltc-v25-ep`, repositories
`{installed_repo}` and `{upstream_repo}`.
No NPU execution was performed for this review or candidate's unit tests.
The candidate has not been copied into the baseline runtime repository.

## Confirmed restored configuration

The launch script `{site_launch_script}`
sets TP8, EP enabled, DP1, V1 explicitly (`VLLM_USE_V2_MODEL_RUNNER=0`),
AllGather EP LoRA, three LoRA slots, rank16, max sequences8,
max batched tokens16384, and DSpark with five speculative tokens.
Its model is `{model_path}`.

Read-only inspection of that model's `config.json` confirmed:

- hidden size4096, HC multiplicity4, BF16;
- routed top-k6, global experts256, expert intermediate size2048;
- target layers43;
- DSpark target layer IDs `[40, 41, 42]`.

The resulting local experts are32 at EP8. The launch script does not request
fully sharded LoRA; these findings concern ordinary BF16/FP16 EP LoRA over
an INT8 W8A8 base, rather than the distinct fully sharded INT8 LoRA branch.

## Target tokens and fusion eligibility

V1 uses query length `1 + num_speculative_tokens` for uniform verification:
`model_runner_v1.py:580-585`, upstream
`vllm/v1/worker/gpu_model_runner.py:842`, and
`vllm/v1/cudagraph_dispatcher.py:37`. Therefore DSpark5 verifies six target
positions per request once all five draft positions are scheduled. This is
not six separate target forwards.

Let T be the target token count presented to the MoE dispatch after
preparation, M=T*6 be original routing pairs, and R be the producer's
dispatched capacity. In this InitRouting path R normally retains the M-sized
static capacity, while only the prefix represented by expert counts is
active. The operator call explicitly sets `active_num=T*6`
(`token_dispatcher.py:451-460`, `device/device_op.py:108-132`). Verify the
actual R in the full-model profile; eligibility always checks physical
capacity, never `sum(expert_tokens)`.

| Active requests | Unpadded T | M / usual R | EP routing M<=384 | Activation R<=48 |
| --- | ---: | ---: | --- | --- |
| 1 | 6 | 36 | Eligible | Eligible before padding |
| 2 | 12 | 72 | Eligible | Ineligible |
| 4 | 24 | 144 | Eligible | Ineligible |
| 8 | 48 | 288 | Eligible | Ineligible |

Uniform full graphs may pad T to a captured batch size. Sequence parallelism
may first pad to TP8 divisibility; its AllGather/unpad logic and LoRA slots
must stay aligned. The table is not a promise of an actual capture size.
The current run's capture sizes are at most48 according to the coordinating
agent. Graph padding at those sizes remains within routing's384-pair guard.
For a general run T>64 causes that fusion to fall back. A low-concurrency
T6 batch padded above T8 can also lose activation eligibility.

`prepare_finalize.py:367-415` mirrors hidden-state gather/unpad with adapter
slots. A gathered T48 is already the global token set; multiplying it again
by TP8 would overestimate M. The usual local active routed count is smaller
than R (approximately36 rows/rank for balanced T48), but its holes/capacity
tails still make the activation tensor R288.

The primary target MoE shapes at T48 are expected to be:

- prepared hidden `[48,4096]`, top-k IDs `[48,6]`, adapter slots `[48]`;
- routed float hidden `[288,4096]`, INT8 base input same shape plus row scales;
- W13 base and LoRA output `[288,4096]` (gate/up combined);
- activated BF16/FP16 and INT8 W2 base input `[288,2048]`, scales `[288]`;
- routed local expert IDs and adapter slots `[288]`;
- ordinary W13 A slices `[3,32,16,4096]`, W2 A `[3,32,16,2048]`.

EP routing fusion (`lora/routing.py:11-84`) additionally requires its flag,
map<=512, INT32/INT64 metadata, INT64 slots, contiguous same-device NPU
tensors, an installed schema, and physical base formats0/2. Unsupported
eager inputs use the old tensor implementation; during compilation the
native runtime rejects unsupported physical formats instead of silently
reading blocked storage. This shape is eligible with all guards satisfied.

Activation fusion (`lora/activation.py:12-43`) additionally requires its
flag, AllGather, EP, non-fully-sharded LoRA, SiLU, no top-k activation scale,
finite limit0..65504, BF16/FP16 NPU `[R,4096]`, contiguous base format0/2,
and an installed schema. The production row cap is48, even though the
native implementation supports a wider test range. R288 therefore uses
the existing activation and dynamic quantization path. The limit is
65504 in current source, not an older proposed guard of10.

Combined expert/adapter indices still use one computation shared by W13
and W2 (`quant_moe.py:307-321,356,407`) in this ordinary EP path when the two
A stacks agree in their first two dimensions. Its optional native index
construction supports up to98304 rows and64 slots
(`lora/combined_indices.py:10-32`), so DSpark5 does not disqualify it.

Fully sharded INT8/grouped LoRA operators are not this EP path.
The runtime baseline Cube Prefill static guard rejects all speculative
configs, so merely setting its flag does not enable it there. A subsequent
source-only V1 DSpark5 expansion is described in
`cube-prefill-dspark5-compatibility.md`; baseline was not updated during this
review.

Parallel candidate update after the initial shape audit: activation source
now permits positive clipping limits up to R384 while unclipped limit0
retains R48. Therefore a DSpark T48/R288 target can use the positive-clipping
candidate when every other guard holds. The table's activation column and
R48 discussion above describe the unchanged container baseline and the
unclipped case, not the newly expanded source-only candidate. Use the actual
checkpoint/layer clipping limit in profile interpretation.

## Why the ordinary MTP buffer is redundant in V1 DSpark

Before this candidate, `DeepseekV4Model` unconditionally allocated the
pre-HC residual buffer at original `deepseek_v4.py:1085-1090` and wrote it
at original1189/1191. FlashComm1 gathered and unpadded a temporary dedicated
to that write. It did not replace the target hidden state or auxiliary
outputs.

Target auxiliary outputs are collected before this stash by averaging HC
dimension (`forward`, current1184-1188). V1 selects DSpark auxiliary layers
from draft config and enables auxiliary outputs
(`model_runner_v1.py:591-593,618-628`). Original IDs40/41/42 are translated
to layer boundaries41/42/43.

V1 obtains the ordinary MTP getter at1574-1578, but rebinds hidden states
only for normalized `method == "mtp"` and a non-None result. DSpark passes
concatenated auxiliary outputs at1587/1607. These have width3*4096=12288;
the draft's `main_proj` and `main_norm` produce4096-wide draft features
(`deepseek_v4_dspark.py:119-129,167-168`,
`spec_decode/llm_base_proposer.py:756-769`). DSpark's draft model owns its
forward and has no ordinary `_mtp_hidden_buffer` writer.

The v1 proposer comment that DSpark uses `mtp` in user configuration must
not determine the guard. Fully constructed `SpeculativeConfig` normalizes
to `method="dspark"` (container upstream `config/speculative.py:834-852`),
and `use_dspark()` means exactly that normalized method (1247-1248).
The target model constructor already receives this constructed
`vllm_config.speculative_config`; no deferred device/state detection is
needed.

V2 has a compatibility hazard: upstream
`vllm/v1/worker/gpu/model_runner.py:599-601,1463-1465` calls and slices the
getter whenever it exists, without checking None or limiting to ordinary
MTP. Its DSpark speculator subsequently prefers aux features
(`gpu/spec_decode/dflash/speculator.py:285-290`), but that preference is
after the unsafe slice. Ascend V2 inherits these consumers. Consequently
the candidate retains the ordinary buffer for all V2 cases.

An unset `VLLM_USE_V2_MODEL_RUNNER` is None in the current upstream env
module, and `VllmConfig.use_v2_model_runner` selects V2 automatically for
DSpark (`config/vllm.py:519-532`). A truthiness check would be unsafe.

## Implemented minimal candidate

Changed files are `vllm_ascend/models/deepseek_v4.py`,
`tests/ut/models/test_deepseek_v4_mtp_copy.py`, and the existing flag comment
in `vllm_ascend/envs.py` (no new environment variable).

Allocation is skipped only when all four constructor conditions hold:

1. `VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY=1`;
2. `vllm.envs.VLLM_USE_V2_MODEL_RUNNER is False` (explicit `=0`);
3. speculative config exists;
4. its normalized `use_dspark()` returns true.

In that case buffer and persistent update indices are None. The whole
dedicated pre-HC flatten/gather/unpad/copy block is guarded by buffer
presence. The copy helper also returns early when disabled. Auxiliary
outputs, target final HC head and normalization, draft execution,
`mtp.*` checkpoint weights, and shared top-k index buffers are retained.

Flag0 retains previous behavior. Ordinary MTP, no speculation, other
speculative methods, V2=1, and automatic runner selection retain allocation
and writes; flag1 continues the existing row-update copy optimization there.
This intentionally does not optimize the no-speculation case.

For this checkpoint and max tokens16384, skipping the ordinary buffer
avoids16384*4*4096*2=536870912 bytes =512 MiB per TP rank (4 GiB across
eight TP ranks), plus131072 bytes/rank for the row-index buffer when flag1
would otherwise create it. Actual allocator/KV cache utilization and speed
require full-model measurements. With FlashComm1 disabled there is no stash
AllGather to save, but allocation/copy are still removed. Other necessary
model/drafter gathers remain.

## Validation and review requirements

CPU tests execute the actual constructor buffer assignments and model
forward/copy methods extracted from source, without importing model kernels
or constructing weights. They are not a distributed/NPU simulation.

-119 tests passed with pytest, including flag0/1; DSpark, ordinary MTP,
  no-spec and other-method allocation checks; explicit V1, V2 and None
  selection; TP1/TP8; FlashComm1 on/off; changed-row writes; partial padding;
  unchanged target and auxiliary outputs; disabled-helper no-write.

- Ruff check and format check passed on the model, env module and test.
- `git diff --check` passed for the candidate files.
- No baseline container runtime source was installed or mutated.

For review/deployment, restart the engine after changing the flag: allocation
and graph branch are decided at construction. All ranks must agree on the
flag and runner selection so removing a collective is consistent. Ensure
V1's DSpark aux-layer setup remains active, and compare warmed acceptance,
output lengths and preemptions alongside TTFT/TPOT/throughput. A test of
the copy optimization alone is not proof of whole-model speedup.

The existing flag0/flag1 isolation now also covers the derived skip decision
in AscendCompiler. The official runner-selection env is already included
by upstream outer AOT factors. A minimal, idempotent speculative hash wrapper
adds normalized method only when this optimization is enabled; flag0 returns
the original speculative hash exactly. See `compile-cache-isolation-update.md`
for the source changes and fake-config validation. This avoids future
same-aux DSpark/eagle3 ambiguity without modifying model selection.
