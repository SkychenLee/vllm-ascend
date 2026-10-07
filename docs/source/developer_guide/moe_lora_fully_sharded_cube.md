# Fully sharded MoE LoRA Cube prefill

`VLLM_ASCEND_MOE_LORA_FULLY_SHARDED_CUBE=1` enables an experimental Cube path
for W13 and W2 LoRA A projections. It defaults to `0`, accepts only `0` or `1`,
and must be set before starting the engine. Both compilation cache layers
include the flag. This flag is independent of the existing ordinary EP
`VLLM_ASCEND_MOE_LORA_W13_CUBE_PREFILL` path.

## Entry conditions

The entry is `prepare_cube_prefill` in
`vllm_ascend/lora/fully_sharded_cube.py`, inside the W8A8 AllGather fused
LoRA/SwiGLU path. Every condition below is required; otherwise the existing
Vector path handles the request.

| Condition | Supported scope |
| --- | --- |
| Hardware | Ascend A3; measured with CANN 9.1.0 and torch_npu 2.10.0.post4 |
| Parallelism | TP8, fully sharded LoRA, EP disabled |
| MoE | 256 experts, top-k 6, hidden width 4096, local intermediate width 256 |
| Adapter weights | BF16, full rank 16; local W13 A rank 2, W2 A rank 16 |
| Batch | CPU Punica mapping contains one positive adapter ID throughout |
| Routed rows | 16384 through 98304; shape alone does not establish a prefill |
| Forward | Eager `NONE`, real attention metadata, non-draft, non-profile |
| Graph | No Dynamo tracing, PIECEWISE execution or FULL graph capture/replay |
| Routing | Existing expert-major AllGather rows; INT64 count-mode group list |
| Storage | Same NPU and contiguous; input/counts format 0 or 2; weights base format 0, 2 or 30 (NCDHW) |

Mixed adapters and mixed base/LoRA batches fall back. Padded or disabled rows
inside a supported single-adapter batch are explicitly zeroed using current
combined indices. Slot zero is not required. Punica recomputes the CPU slot
on every metadata update, including when the ordinary EP Cube flag is off.
Decode and small prefills retain Vector projections: measured Cube overhead
outweighs its benefit at 48 and 288 routed rows.

## Matrix fusion and communication

For each expert, the two W13 A matrices are concatenated along their rank
axis. One grouped matrix multiplication produces adjacent gate/up rank-2
outputs in FP32. The established paired TP AllGather then produces its
rank-major layout; the existing B/add/SwiGLU/quantization kernel consumes it.
W2 performs a rank-16 grouped A projection before the established TP
AllReduce and B expansion, including the existing output-column offset.

INT8 activations and BF16 weights convert exactly to FP32. The non-quantized
GMM consumes FP32 inputs and returns FP32 outputs. The row dequantization scale
is multiplied after the dot product, followed by the current adapter mask.
This changes floating-point reduction order from Vector BGMV. It does not
re-quantize the adapter or round the projection result to BF16, but it is not
bitwise identical to the original path. Model-quality equivalence requires
separate validation.

The installed CANN GMM rejects BF16 inputs with FP32 output. Its supported
[dtype combinations](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)
therefore inform the explicit FP32 conversion used here.

## Routing, memory and weight lifetime

The existing AllGather producer must provide a complete expert-major
partition: nonnegative counts with sum equal to the routed row count. Prefix
counts are computed once on device and shared by W13 and W2. For a chunk
`[start, stop)`, boundaries are `clamp(prefix-start, 0, stop-start)`. Empty
experts and boundaries that split an expert are supported. There is no
device-to-host count transfer, dynamic host sorting or cached routing.

The FP32 input conversion is bounded by 128 MiB per chunk: 8192 W13 rows, or
up to 131072 W2 rows (the entry gate caps the latter at 98304). Temporary
weight packing is approximately 16 MiB for W13 and 4 MiB for W2, plus the
short-lived BF16 concatenation, outputs, masks and CANN workspace. These
figures describe tensors owned by this implementation, not total model or
allocator peak memory. Runtime allocations use the caller's current stream.

Weights are selected and converted on every call. Reloading an adapter or
switching its slot takes effect immediately; there is no additional per-layer
weight cache, persistent scratch arena or new mutable global state. Dummy
memory profiling retains the original path, so this experimental option
requires enough runtime headroom for the bounded temporary tensors.

## Validation and measured scope

CPU tests cover the metadata-only gate, graph/profile/draft fallback,
mixed/base mappings, strict flag parsing, chunk boundaries, zero-count
experts, masked rows, live weights, TP layouts and cache separation.
NPU regression tests compare the actual implementation with native INT8
BGMV, including adapter slot 1, empty experts, a chunk boundary inside an
expert, disabled rows and a subsequent weight update.
The implementation passed 302 CPU checks across the affected LoRA, cache
and environment suites, plus three NPU regression cases exercising the
actual entry gate. These are targeted suites, not the full repository suite.

The standalone benchmark is
`benchmarks/benchmark_moe_lora_fully_sharded_cube.py`. Assign an idle device
with `ASCEND_RT_VISIBLE_DEVICES`, then supply explicit `--device`, `--library`
and `--output` arguments. It includes packing, conversion, prefix counts,
scale and masking; it excludes TP communication and B projections.

The initial 2026-10-07 run passed 24 cases at 288, 16384, 24576 and 98304 rows,
with balanced, sparse and single-expert distributions. Each case also checks
33 independent CPU reference rows and exact zeros for inactive rows.

| Balanced 98304-row projection | Native Vector | Cube implementation | Speedup |
| --- | ---: | ---: | ---: |
| W13 gate/up A | 5.891 ms | 3.877 ms | 1.52x |
| W2 A | 2.137 ms | 0.368 ms | 5.81x |

Relative L2 differences in these cases were approximately `3.1e-7` for W13
and `1.0e-7` for W2. These synthetic operator results do not establish model
quality. The benchmark includes disabled rows with NaN scales and checks
that masking produces exact zeros before communication.

## Whole-model comparison

One OFF/ON pair completed on 2026-10-08 (Asia/Shanghai) using physical devices
8 through 15, DeepSeek-V4-Flash W8A8, TP8, rank-16 fully sharded LoRA, D-Spark5
and prefix caching disabled. Each fresh engine ran the same eight concurrent
requests of exactly 8192 input and 4096 output tokens, with max batched tokens
16384, max sequences 8, max model length 128000 and memory utilization 0.93.
Both runs used the same source and native binaries, adapter configuration,
token-ID prompts and prior optimization flags; only the new Cube flag changed.

| Metric | Cube OFF | Cube ON |
| --- | ---: | ---: |
| Verified requests | 8/8 | 8/8 |
| Output tokens | 32768 | 32768 |
| Measurement wall time | 87.014 s | 78.532 s |
| Output throughput | 376.585 tokens/s | 417.257 tokens/s |
| Mean TTFT | 7.403 s | 6.906 s |
| P95 TTFT | 12.613 s | 11.777 s |
| Mean client TPOT | 16.785 ms | 16.029 ms |
| D-Spark draft acceptance | 86.504% | 92.499% |
| KV capacity | 157554 tokens | 157554 tokens |
| Prefix cache hits / preemptions | 0 / 0 | 0 / 0 |

The observed throughput change was +10.80% and mean TTFT change was -6.71%.
Acceptance and generated tokens also changed: none of the eight full output
sequences was identical between runs. Therefore the throughput change cannot
be attributed entirely to Cube, and this pair is not a quality-equivalence
test. Random-token prompts and a single pair also limit generalization.
Decode remains on the original Vector path; the measured TPOT change is not
evidence of a new decode Cube kernel.

An eight-request, 8192-input/128-output warmup used a profiler capped at two
iterations. The current rank-0 device trace contained 559 FP32 W13 Cube GMMs
and 86 FP32 W2 Cube GMMs with `MIX_AIC` execution; OFF contained neither.
All eight rank collections finalized and the profiler stopped before the
formal performance run. Profiled warmup latency is excluded from the table.

Independent CPU audits passed 90 checks per run, recomputing request token
counts, timing distributions, throughput, Prometheus deltas and Cube counts
from raw artifacts. They also verified source hashes, identical prompts,
fresh service ownership and release of the test devices. A compact record
with artifact digests is available in
[the measured evidence](moe_lora_fully_sharded_cube_evidence.json).
