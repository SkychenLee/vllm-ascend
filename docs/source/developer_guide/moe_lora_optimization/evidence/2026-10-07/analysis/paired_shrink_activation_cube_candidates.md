# TP8/EP8 LoRA Decode candidates, 2026-10-07

This records the initial source audit and the subsequent authorized independent
operator tests on physical12 after root stopped the full baseline. The new
source candidate is prepared but is not installed by this subtask. No candidate
end-to-end model performance is claimed here.

## Target and important semantic finding

The restored launch script uses TP8, EP8, ordinary (not fully-sharded) LoRA-r16,
three adapter slots, AllGather, DSpark with five speculative tokens, eight
requests, maximum batched tokens 16384, and FULL_DECODE_ONLY/npugraph_ex.
The maximum candidate verification shape is T48, K6, routed capacity R288.
Requests, verification tokens, dispatched capacity, and initialized local rows
are different quantities. Padding and DSpark acceptance can create other
captured T/R buckets. New real traces are needed to establish their frequency.

Root read model config `swiglu_limit=10.0`. However, current
`vllm_ascend/models/deepseek_v4.py:370` reads that limit, line 412 passes it to
shared experts, while the routed `FusedMoE` constructor at lines 436–466 does
not pass `swiglu_limit`. Therefore config 10 alone does not establish routed
MoE clipping. The common builder preserves an actual layer limit:
`ops/fused_moe/fused_moe.py:298` / `moe_runtime_args.py:258`, and
`lora/quant_moe.py:363,369` consumes that value. Confirm the constructed routed
instance or profiler before claiming clip10 target coverage. Do not add
clipping to routed experts as a performance change.

Update: the restored real profile confirms routed T48/R288 SwiGLU dim-1 with
no routed Clamp; shared [48,256] uses clip10. Routed limit0 is the actual target.

## 1. True floating gate/up paired A shrink

The initial `lora/punica_npu.py:497–532` loops over A0/A1 and calls the legacy
floating shrink twice. The public `bgmv_shrink_pair` schema already exists
(`torch_binding.cpp:318`), but `kernels/bgmv_shrink.cpp:274` implements it as two
original launches. Initial production Python did not call that pair entry. The prepared default-off
FUSED_SHRINK_PAIR candidate now uses a true one-launch fast shape, preserving
legacy two-launch behavior outside its specialized domain.
Current ordinary EP W8A8 LoRA uses floating routed activations even though the
base experts use INT8; the fully-sharded non-EP INT8 pair family is a separate
path and cannot be applied to this configuration by changing a flag.

At H4096/r16 each old projection loads and casts one X row, then serially
loads/casts/multiplies/reduces 16 BF16/FP16 weight rows into FP32. Two A calls
read 262144 weight bytes plus 16384 X bytes per valid row (logical accesses,
not measured DRAM traffic; weights may cache). Sharing X saves 8192 bytes per
valid row and one X cast, but saves none of the 32 dot products. It also saves
one launch and potentially one separate zero allocation/fill. Do not assume
2× throughput from pairing alone.

A more significant possible issue is row scheduling. The old host uses
`numTokensPerCore=ceil(R/aiv)` and each core owns a contiguous chunk. With
R288 and aiv64 this is five rows/core and 58 launched blocks. If approximately
36 local initialized rows form the front prefix, only eight cores execute dot
products. Strided rows spread that illustrative prefix across 36 cores.
This depends on actual routing occupancy and prefix order; 36 is a balanced
EP estimate R/8, not a measurement. `token_dispatcher.py:435–469` supplies local
expert counts, capacity tails, and explicitly masks unwritten rows.

Candidate: one new specialized Vector launch for H4096/r16/R1..384. Each core
loads one X row and casts once, performs gate rank 0..15 then up rank 0..15
using the exact old FP32 Mul/ReduceSum/acc+= ordering, and writes two separate
64-byte rank rows into `[2,R,16]` FP32. Python uses a single zero tensor; each
negative combined index still skips both planes. Strided row scheduling changes
only independent row ownership. Keep old single shrink untouched and preserve
the public pair's old two-launch behavior for other shapes.

Guard: independent default-off flag; ordinary non-fully-sharded double A,
input_scale None, x BF16/FP16/H4096/contiguous, both A weights identical layout
`[slots,16,4096]`, rows at most384, shared INT64 contiguous combined indices,
same device, physical format0/2. Native runtime checks physical format for all
fast-path storage, including graph compilation where FakeTensor cannot query
format. No CPU reads of indices/counts; graph replay re-reads current combined
indices. New native entry behavior must remain valid for changed adapters,
negative indices, mixed slots, zero rows, and nonzero output references.

Minimum implementation ownership: `kernels/bgmv_shrink.cpp`, the floating A
dispatch in `lora/punica_npu.py`, pair-related checks only in
`csrc/torch_binding.cpp`, one `envs.py` flag, and focused LoRA UT. No new schema
or weight packing is required. Native positive indices retain the established
caller contract: valid combined adapter/expert slots; do not claim host value
validation without a device-side check.

## 2. Activation + dynamic INT8 quantization at R288

Native `moe_lora_activation_quant` supports BF16/FP16 ND0/2 `[0..384,4096]`,
returning rounded floating `[R,2048]` for W2 LoRA, INT8 `[R,2048]`, and FP32
scales `[R]` for base W2. It is one Vector launch, not Cube. Initial Python
permitted R<=48, silu, no topk scales, ordinary AllGather EP; all other activations
remain on their existing paths. Native default rows>48 uses up to the device's
available vector cores, whereas rows<=48 is capped at24. Current UB43136 bytes
processes one complete row at a time with strided rows and no cross-core reduce.

For silu clipping, preserve gate clamp(max=limit), up clamp[-limit,limit],
BF16/FP16 clamp rounding, SwiGLU output rounding, scale maxAbs*float(1/127),
vector 127/maxAbs multiplier, FP32 integer rounding, and exact zero-row scale0.
Do not remove the floating output: ordinary EP W2 LoRA consumes it.

Prior 2026-10-06 independent graph core sweep (100 operators/graph, 20
replays/sample, three sample blocks) has R8/R48/R384; it has no R96/R192/R288.
BF16 R384/noClip, core64: 20.2526→19.9144 microseconds, 1.017×; core48:0.997×.
BF16 R384/clip10, core64:79.9586→21.1774 microseconds,3.776×; core48:3.643×.
FP16 R384/clip10, core64:66.3283→20.9913 microseconds,3.160×.
Each of those recorded outputs matched all floating/INT8/scale raw bits.
These are synthetic operator timings, not restored DSpark/model results.

Final prepared candidate allows positive clip R<=384 and noClip R<=288 after
the new physical12 measurements below. NoClip384 still uses the old path. This
source is not deployed by this subtask. `activation_clip_microbench.py` prepares fresh installed-library tests
for R48/96/192/288/384, BF16/FP16, clip10, default cores, nonzero random/edge
inputs plus a zero row, exact raw bits, changed-input graph replay, and device
event timings. Optional `--limits 0 10 --core-limits 0 32 48 64` tests the noClip
region and tuning. It must run on an allocated physical device after baseline
stops. New R288 noClip has a modest measured operator gain; do not extrapolate
clip10 gains to the actual routed limit0 workload.

Evidence: original host repo
`work-dir/decode-optimization-20261006/activation-native/core-tuning-result.json`,
`final-default-result.json`, `result.json`, `stress-core24-result.json`; strict
installed reference `work-dir/decode-optimization-20261006/installed-activation-reference.json`.

## 3. Graph-safe fixed-shape Decode Cube alternatives

Current Cube is an eager Prefill PyTorch gather/mask + bmm(FP32 output) + scatter
composition. `lora/cube_moe.py:static_config_supported` excludes any speculative
config; `prepare` permits R6144..98304 and counts mode1, reads counts.cpu().tolist,
and requires >=512 initialized rows. `quant_moe.py:259` also excludes graph,
profile/dummy, and absent attention metadata. It cannot serve DSpark Decode.
Its paired packed A weights are `[32,4096,32]` BF16 for one adapter; 64-row tiles,
layer cache generations, and same-stream scratch are deliberate restrictions.

Device-only alternatives are technically possible, but are new designs:

1. Fixed expert/adapter padded BMM arena. A device planner reads counts and
   combined IDs; gather/mask and scatter use fixed-capacity metadata and static
   shapes. Packing must happen at adapter load/reset or be safely cached before
   capture, never selected by a CPU read of current batch counts. No overflow may
   be dropped: allocate maximumR or use a device fallback. Capture/cache lifetimes
   must handle up to three adapters and changed mapping without stale pointers.
2. Native mixed Vector/Cube launch with a static grid of expert/tile candidates.
   Each device block reads counts to skip empty tiles, and handles device-generated
   start/count metadata. Static host launch avoids CPU synchronization while actual
   work varies by routing. It still needs a validated CANN Matmul implementation,
   padded M/N/K layouts, UB/L1/L0 budgets and producer/consumer synchronization.
   This is not an existing `grouped_w13_cube` op.
3. Per-row micro-GEMM with paired N32. It avoids host grouping but repeatedly
   stages the same weight and pads small M to Cube tiles. Compare all staging and
   launches against the simpler paired Vector kernel; raw peak Cube FLOPs is not
   a useful performance prediction for this sparse small-rank workload.

Illustrative cost: E32, paddedM64, H4096, pairedN32 processes536.9MFLOP and uses
16MiB BF16 gathered X plus8MiB BF16 packed weights. Against36 local valid rows,
useful pairedA work is9.44MFLOP, so M64 per-expert padding is ~56.9×. PaddedM16
still computes134.2MFLOP (~14.2×). Against fully local288 valid rows these factors
fall, but actual Decode experts often have only a few rows. These are shape
estimates, not observed utilization or required hardware tile guarantees.

Therefore prioritize paired Vector/scheduling first if real traces show A shrink
hot. For Cube, require per-layer local count distribution and adapter distribution,
new quantization-aware FP32 accumulate comparisons (Cube summation order can
differ from the Vector ReduceSum reference), and a graph-native lifecycle design.
Precision is user-lower-priority, but do not call a differing reduction bit-exact.

## Suggested next measurements after baseline/profile

- Stage shape histograms: T, capacity R, local initialized count, expert/adapter
  occupancy, actual activation name/limit, fused guard hits.
- Paired A end-to-end operator test must compare original two single projections
  and include both output initialization/fills, mixed adapters/negative slots,
  prefix/sparse patterns and changed-input/mapping graph replays.
- BF16 A, gate/up B, W2 A/B separately and the complete local MoE LoRA layer,
  then the restored8192+4096/C8/maxbatch16384 DSpark model. Micro timings alone
  cannot establish full throughput or accepted-token Decode gains.

## Final independent evidence and prepared sources

Physical12 activation tests:20cases (BF16/FP16, R48/96/192/288/384, limits0/10),
eight changed-input graph replays/case. Floating, INT8 and FP32-scale outputs
match reference raw bits throughout; zero-row scales are exactly0. BF16 R288
noClip:19.0214→16.3302us (1.1648×); clip10:71.4884→17.6040us (4.0609×).
BF16 noClip R96/R192/R384:1.701/1.350/1.026×. Evidence:
activation-rows-result.json and activation-rows-test.log.

Physical12 paired-A tests:24cases (two dtypes, M48/144/288/384, prefix/random/full
patterns), 96 weight slots, nonzero independent weights. Compare with two
original shrink projections: all eager and four changed-input/weight/index
graph replay outputs raw bit exact; all-inactive then reactivated routing
clears stale output through the recorded pair zero fill; signed-zero inactive
rows and protected offset-output padding are retained. BF16 prefix M288,
36 valid rows:3.5807×. M144prefix:2.5012×; M384prefix:2.6205×; randomM288:1.0104×;
fullM384:0.9897× (about1% regression). Benefits depend on initialized prefix
balance. Evidence:shrink-pair-result-slots96.json / shrink-pair-test-slots96.log.

Pair tests use a Torch-free PODwrapper with TASK_QUEUE_ENABLE=0 and fresh
Ascend910_9391 CANN build, not the full production binding. This avoids
duplicate schema registrations and queue reorder; output fills are included
in timing. Native ND runtime guard/Meta/compiled production validation is a
subsequent step after the full binding is built. Do not call this end-to-end
LoRA or model throughput, nor claim310P support (the new Python gate excludes it).

Final CPU UT:98passed, one import-rewrite warning. Scoped Ruff0.14 format/check
PASS. Source list: csrc/kernels/bgmv_shrink.cpp, new
csrc/kernels/bgmv_shrink_pair_config.h, csrc/torch_binding.cpp pair guard only,
vllm_ascend/lora/punica_npu.py, vllm_ascend/lora/activation.py, envs.py,
tests/ut/lora/test_float_moe_shrink_pair.py and test_decode_moe_lora.py.
Core math/ABI is unchanged by the final named-constant extraction; first probe
source snapshot and finalsource hashes are recorded separately.

Fresh full native source snapshot: native-runtime-20261007-source, original
source manifest native-runtime-source-sha256.json. Dedicated runtime_builder
now owns configure/build and installation remains with root. Physical12
operator processes exited; there is no outstanding NPU job from this subtask.
