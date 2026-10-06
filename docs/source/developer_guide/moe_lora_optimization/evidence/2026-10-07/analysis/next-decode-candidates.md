# Next Decode candidates: W2 single A and shared gate/up A

Read-only analysis, 2026-10-07. No production source was changed, no native
build was started, and no NPU work was run for this analysis. The current
paired-W13/activation/Cube candidates must finish their real model evaluation
before starting another implementation round.

## Measured baseline projection split

The CPU-only analyzer `analyze_shrink_projections.py` reads the existing rank0
baseline profiler database in SQLite read-only mode. Its result is
`baseline-shrink-projections.json`. This extends
`../bench/baseline-rank0-generic-analysis.json` without re-recording a profile.

Scope: physical8/rank0, main captured model49, 43 layers, T48, K6, routed
capacity R288, DSpark5. Each global task occurs11 times; the table summarizes
steady replays6–10. The profiling workload was 8192 input + 128 output/C8,
which is separate from the formal 8192+4096/C8 throughput benchmark.

**Custom BGMV/SGMV input shapes are N/A.** H and projection roles below are
inferred from known same-stream base-matmul input-shape anchors plus the
production call order; they are not directly observed custom-kernel input
shapes. Every routed layer has two shrink calls following the
`GroupedMatmulWeightNz` input `[288,4096]` and before routed SwiGLU, then one
shrink following its `[288,2048]` down GMM. Every shared layer has two shrink
calls following its `[48,4096]` QuantMatmul, then one following `[48,256]`.
The analyzer checks exactly43 calls for each of the six roles and the same
three-projection routed pattern in all43 layers. Chronological gate/up naming
also follows the source A tuple order.

| Inferred role | H | Calls/replay | Mean summed device ms/replay | Mean us/call |
| --- | ---: | ---: | ---: | ---: |
| Routed W13 gate A | 4096 | 43 | 3.605637 | 83.8520 |
| Routed W13 up A | 4096 | 43 | 3.496043 | 81.3033 |
| Routed W2 A | 2048 | 43 | 2.754090 | 64.0486 |
| Shared gate A | 4096 | 43 | 1.147602 | 26.6884 |
| Shared up A | 4096 | 43 | 1.044629 | 24.2937 |
| Shared down A | 256 | 43 | 0.634438 | 14.7544 |

Routed shrink totals9.855770ms: W13 is7.101680ms and W2 is2.754090ms.
Shared shrink totals2.826669ms: gate/up is2.192231ms and down is0.634438ms.
These values reproduce the aggregate in the generic baseline result. They
are summed device work on this rank, not a cross-rank critical-path model
or a predicted wall-time reduction. Do not add these numbers to independently
overlapping groups or present them as candidate end-to-end gains.

An illustrative first-layer replay6 sequence is: W13 gate shrink task14087,
up task14177, SwiGLU task14262 `[288,4096]`, quant task14284 `[288,2048]`,
down GMM task14353 `[288,2048;32,2048,4096;...]`, W2 shrink task14402.

## Candidate1: dedicated strided W2 floating shrink

This is the next lower-risk native candidate if the current real model
results justify more work. `lora/fused_moe.py:moe_lora_apply_w2` passes the
floating activation `[R,2048]` and one A plane to
`PunicaWrapperNPU.add_lora_fused_moe`; its existing paired guard requires two
A planes, so W2 uses the legacy single `bgmv_shrink`. In this configuration
the base down GMM is W8A8, while the ordinary EP LoRA A path is floating.
Changing that floating path to the fully-sharded INT8 family would change
semantics and is not the scheduling optimization proposed here.

The legacy binding sets `numTokensPerCore=ceil(R/aiv)` and the kernel owns
contiguous row chunks. At R288/aiv64 this is five rows/block and58 launched
blocks. If balanced EP initialized rows occupy a36-row front prefix, only
eight blocks execute dot products. A fixed strided grid with64 blocks spreads
that illustrative prefix over36 blocks. The actual local count is dynamic:
36 is the R/8 balance estimate, not a measured occupancy from this profile.
All negative combined indices, including padding and inactive adapters,
remain skipped. No host readback of expert counts is needed.

A dedicated H2048/r16 small-row implementation can retain the exact old
per-rank FP32 math: cast X/weight, multiply, `ReduceSum` over2048, initialize
`acc=0`, add the reduced value, then apply FP32 scale. Replacing the legacy
11776-element UB buffers with2048-element buffers would reduce logical UB
allocation from141440 bytes to24704 bytes at rank16. This is a resource
calculation, not evidence that UB allocation itself dominates latency. Rank
reductions, BF16/FP16 rounding boundaries and scale order should remain
unchanged; rows alone change ownership. Output remains FP32 `[R,16]`.

The current pair-prefix synthetic test proves the scheduling technique is
useful for W13, but it is not a W2 measurement. W2 has half the input H, one
weight plane and one original launch. It cannot claim the W13 pair speedup.
Uniform/random valid rows already engage more old blocks; full-valid rows
have similar maximum rows/block under either schedule. No extra launch is
removed by a W2-only scheduling path. Amdahl bounds from this baseline are
small: even eliminating all W2 A work would remove at most2.754090ms of
arithmetic work/replay on rank0. Hypothetical 2x/3x improvements correspond to
1.377045/1.836060ms of such work, not guaranteed model-time savings.

Suggested future isolation: a dedicated opt-in Decode schema/entry, or an
explicitly selected internal specialized entry, leaving all legacy single
callers unchanged. Use named constants, same-device metadata validation,
Meta shape validation, runtime ND/base format0/2 checks, contiguous floating
X and A, INT64 IDs, FP32 output, non-overlapping storage, H2048/r16 and a
validated small-row cap. Python should require ordinary non-fully-sharded
MoE, input_scaleNone, one A plane, and supported hardware; do not extend to
310P without a build and test. During graph tracing, avoid FakeTensor format
probes but retain native runtime storage checks. Unmatched Python guards use
the existing single shrink. Direct native invalid metadata/storage raises
an error; it must not reinterpret blocked storage or silently fall back.
Positive slot IDs retain the caller's valid-slot contract unless explicit
device-side bounds validation is added.

A future verification window should cover M48/144/288/384 with prefix,
random-negative and full-valid IDs, mixed nonzero adapter/expert weights,
BF16/FP16, signed-zero skipped outputs, guarded padding, scale variations,
changed inputs/indices and all-inactive/reactivation replay. Compare all FP32
raw bits with the unchanged old single kernel using production taskqueue1,
compiler options and outer NPUGraph capture. Measure after warmup with enough
weight slots to avoid overstating a single-slot cache effect. Then use the
new real model profile, not the isolated timing, to decide deployment.

## Candidate2: shared-expert paired gate/up shrink

Shared gate/up A contributes2.192231ms/replay in the same baseline. Source
`models/deepseek_v4.py:DeepseekV2MLP` uses a merged gate/up projection and a
TP-sharded intermediate; the observed shared activation is `[48,256]`, so
down A has H256 rather than routed H2048. Its clipping limit10 does not change
the routed limit0 activation contract.

The dense `PunicaWrapperNPU.add_shrink` loop currently invokes the two A
projections separately; the captured target uses SGMV rather than BGMV for
this path. The old SGMV kernel scans sequence lengths to obtain each row's
adapter, then loads/casts X separately for each A call. A real one-launch
paired SGMV could share both that metadata lookup and X load/cast while
preserving each A's rank reduction and downstream separate B expansion.

However T48<aiv64 means the current contiguous scheduler already assigns one
row/block, with up to48 useful blocks. It has no R288-prefix imbalance to
repair. Pairing removes one launch and duplicated X handling, but none of
its32 rank dot products or weight reads. Its predicted value is therefore
lower and less certain than the routed W2 scheduling candidate. Shared down
H256 already averages14.7544us/call; fusing A with its B would introduce a new
compound kernel and base-output addition/rounding contract for a smaller
0.634438ms/replay budget.

A shared paired op must accept the actual prefill sequence metadata, including
mixed adapters, zero-length segments and negative IDs. It cannot reuse MoE's
combined expert/adapter IDs without a separate adapter mapping. Full Decode
or speculative verification classification is represented by `is_prefill`
for the observed SGMV path; a guard must follow the actual metadata backend,
not infer the call from the word Decode. No `.item()` or CPU count readback
may be introduced inside capture.

Dense `add_lora_linear` also accepts caller-provided buffers. A rewrite must
preserve those buffers and skipped-row contents rather than always replacing
them with a newly zeroed `[2,M,16]` tensor. A minimal first guard could select
only internal buffers and measured shared H4096/r16/M<=48, with format/device/
layout checks. Keep generic multi-slice, fully-sharded, larger Prefill and
caller-buffer behavior unchanged. Runtime schema safety and compiler-cache
flags would need the same review as the W13 paired candidate.

## Deferred larger fusion/Cube alternatives

Activation+W2 A fusion could reuse floating activated X inside the activation
kernel and avoid a separate X load/launch. It adds A weights and IDs to an op
that must still emit rounded floating activation for LoRA, INT8 activation
and FP32 scale for base GMM. It needs the exact2048-element legacy rank
reduction and potentially more UB/staging; it is more intrusive than strided
W2 and should not be the first step without a remaining-hotspot profile.

A W2 fixed-shape Cube design needs device-only grouping, stable capture
arenas and correct slot/expert packing. At balanced36 valid routed rows over
32 local experts, per-expert M is often very small, and padding can dominate.
The existing eager Prefill W13 Cube path is not such a Decode implementation.
No current evidence supports predicting a Cube speedup for this W2 shape.

Decision for this round: **analysis complete; W2/shared implementation is
deferred until current installed candidates have a valid model result.**
