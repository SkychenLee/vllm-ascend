# MoE LoRA optimization design and portable evidence

The [design document](design.md) describes the AllGather W8A8 MoE+LoRA paths,
operator fusion contracts, fallback behavior, graph capture, compiler cache,
Cube prefill, and the explicit V1 DSpark5 configuration validated on 2026-10-07.
Earlier evidence remains in `evidence/2026-10-06/`; new measurements and reviews
are in `evidence/2026-10-07/`. Implementation does not imply support for every
model, shape, hardware generation, or speculative configuration.

## Current single-adapter status

The [single-adapter guide](../moe_lora_single_adapter.md) documents the latest
portable implementation contract and evidence. The final 2026-10-07
fully-sharded comparison used `VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER=0/1`:

| Metric | Flag OFF (0) | Flag ON (1) |
| --- | --- | --- |
| Output throughput | 316.02 tokens/s | 378.05 tokens/s |
| Mean TTFT | 7.497 s | 7.512 s |
| Mean TPOT | 19.120 ms | 16.313 ms |
| DSpark acceptance | 75.30% | 89.33% |

The observed throughput increase was 19.63% for this single pair of runs.
The acceptance rate also changed, so this is a conditional end-to-end result
and does not establish a pure kernel speedup or model quality equivalence.
The existing Cube prefill flag remains limited to ordinary TP+EP LoRA.
The independent [fully-sharded Cube path](../moe_lora_fully_sharded_cube.md)
now accelerates large eager prefill A projections behind a separate,
default-off flag; see that guide for its limits and validation.

## Earlier EP comparison

The same TP8+EP8 setup completed eight requests, each with 8192 input and 4096
output tokens, with `max_num_batched_tokens=16384` and DSpark5. Baseline throughput
was 259.432696 output tokens/s; the combined candidate reached 278.559021, a 7.37%
increase. Mean TTFT changed from 17.550260 to 13.484378 s and mean TPOT from
23.084438 to 20.106955 ms. These are single unprofiled runs with synthetic random
token prompts. All eight generated token sequences differ between the two runs;
draft acceptance changed from 85.87% to 82.72%. Model quality and output accuracy
equivalence have not been validated, and this comparison does not isolate each
optimization's contribution.

See the [performance report](evidence/2026-10-07/performance_report.md),
[comparison](evidence/2026-10-07/bench/optimized-comparison.json), and
[profile comparison](evidence/2026-10-07/bench/profile-comparison.json). Operator
and changed-input graph checks used nonzero synthetic data; their numerical
results are separate from the model output diagnostic.

## Reproduction and scope

Build and install the native libraries in a matching Python 3.12/aarch64,
Torch 2.10, torch_npu 2.10.0.post4, Ascend910_9391 environment. The design lists
the startup configuration and all flag conditions. The model/adapter paths in
this publication are placeholders; no weights are included.

The full workload client is [benchmark_dspark_lora.py](../../../../benchmarks/benchmark_dspark_lora.py).
Run smoke warmup with seed `2026100701`, then full mode with seed `2026100702`
against a fresh server for each condition. For example, from the repository root:

```bash
python benchmarks/benchmark_dspark_lora.py --mode full --model lora1 \
  --seed 2026100702 --output-dir ./benchmark-results/full
```

Use the same workload and server arguments for both conditions. A repeated
prompt seed on an existing prefix-caching server is not a valid comparison.
The client verifies returned token IDs, token counts, finish reason and usage;
it includes prefill in the throughput interval. DSpark can stream several tokens
per chunk, so TPOT is an aggregate token-arrival measure.

The existing single-layer benchmark in `benchmarks/benchmark_moe_lora_decode.py`
uses the two old source snapshots in `evidence/2026-10-06/baseline/`. Those files,
and new validation scripts stored with the evidence, are historical fixtures,
not production implementations. Evidence script snapshots retain site-specific
paths as placeholders and require adaptation before standalone use.

No model weights, adapter weights, prompt/output token dumps, native binaries,
compiled caches, profiler databases/traces, or large embedded vendor build
manifests are published. Compact JSON files identify omitted raw fields and
retain the original artifact SHA256 and measured results. They are summaries
of actual experiments, not substitutes for running code.

## Paths and provenance

| Placeholder | Meaning |
| --- | --- |
| `{repo_root}` | Original host repository |
| `{installed_repo}` | Editable Ascend installation in the container |
| `{upstream_repo}` | Upstream vLLM checkout |
| `{experiment_root}` | Local 2026-10-07 experiment artifacts |
| `{previous_experiment_root}` | Local 2026-10-06 experiment artifacts |
| `{site_launch_script}` | User's launch script |
| `{model_path}` / `{adapter_path}` | User's local weight locations |
| `{profiles_root}` / `{original_profiles}` | Local profiler directories |
| `{original_workspace}` / `{compile_cache}` / `{capture_host}` | Historical local paths or host labels |

[evidence-manifest.json](evidence-manifest.json) records original and published
text hashes. Redaction and formatting can change the published text hash;
recorded runtime source/binary hashes and numerical measurements remain intact.
Backup paths and raw profile locations identify local experiment sources and
are not downloadable artifacts. Physical devices 8-15 and the container name
are retained as measurement provenance.

## Publication validation

The earlier publication task ran no NPU work. Eleven related CPU/Meta/Mock test files
passed 388 tests with no NPU initialization. Functional Ascend Python imports
came from this publication worktree; installed generated `_build_info` was the
only metadata exception. The complete repository `bash format.sh ci` hooks
passed before the signed-off local commit. These checks do not represent the
full repository test suite or a new model performance run of the formatted
publication worktree.

The 388-test record belongs to that earlier publication and does not describe
new single-adapter test results. Current evidence and its scope are recorded
in the [single-adapter guide](../moe_lora_single_adapter.md).

Publication preparation preserves prior native implementations and applies
formatting only in this worktree. The new native kernels and installed model
measurements correspond to the runtime manifests linked from the design.
