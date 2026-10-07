# Fully sharded MoE LoRA single-adapter routing

`VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER=1` enables an experimental routing
metadata fusion for quantized MoE with AllGather, fully sharded LoRA, and
expert parallelism disabled. The default is `0`. Set it before engine
initialization and graph capture; both Ascend compilation cache layers include
the flag. Values other than `0` and `1` are rejected.

The operation combines routing recovery, adapter masking, and combined-index
construction into one `int64[R]` result. W13 and W2 share that result. It caches
the first token's adapter arithmetic inside each device invocation; it still
handles other adapters, inactive rows, and mixed base/LoRA requests correctly.
It neither requires adapter slot zero nor fixes the adapter identity at graph
capture time.

## Entry conditions and fallback

The Python entry is `vllm_ascend/lora/single_adapter.py`, called by the
AllGather branch in `vllm_ascend/lora/quant_moe.py`. Existing supplied routing,
EP LoRA, and non-fully-sharded LoRA retain their established paths.

| Input or condition | Supported contract |
| --- | --- |
| Routing | `expanded[R]` and `topk[T,K]`; `R=T*K`, `K>0` |
| Routing dtypes | Each of expanded and topk can be INT32 or INT64 |
| Adapter metadata | INT64 token slots, INT32 enabled mask; 1–64 allocated adapters |
| Weight tables | W13/W2 share their leading adapter and expert dimensions |
| Enabled mask | Length A or A+1; the reserved final entry is excluded from lookup |
| Storage | Same NPU; native inputs contiguous and physical format 0 or 2 |
| Small path | `R+T<=1024`, subject to actual device UB capacity |
| Large path | `8161<=R<=98304`, sufficient UB and available vector cores |
| Empty routing | Returns empty INT64 output without launching a kernel |

Unsupported Python metadata, missing native registration, unsupported shapes,
or eager blocked storage return to the existing routing path. Native direct
calls reject invalid metadata, blocked storage, or insufficient resources.
During compilation the helper avoids querying FakeTensor storage; the native
host checks real storage during eager invocation or capture. Replay runs the
captured kernels.

For K6, ordinary eight-token decode has R48 and D-Spark5 target verification
with 48 tokens has R288, both within the small path. The middle interval uses
the existing recovery implementation.

## Fusion and producer contract

For source row `s`, the kernel computes:

```text
destination = abs(expanded[s])
slot = slots[min(s // K, len(slots) - 1)]
combined[destination] = slot * num_experts + topk.flatten()[s]
```

Negative, out-of-table, or disabled slots produce `-1`. Integer payloads remain
full width, with the same wrapped INT64 arithmetic as the existing combined
index operation. Nonzero mask values enable an adapter.

The non-EP AllGather producer must supply a complete unique permutation in
`abs(expanded)`: every destination in `[0,R)` occurs exactly once. Signed
indices are accepted under that contract. Duplicate, missing, or out-of-range
destinations are unsupported; this is not the EP negative-exclusion contract.

The small kernel performs inverse scatter in UB and one contiguous output
write with one AIV launch. The large path scatters each result into a private
64-byte destination record, then packs the first eight bytes of each record
with a second launch. Record padding prevents neighboring writers from sharing
DMA cache lines. Scratch is `R*64` bytes, at most 6 MiB, scoped to the invocation.

## Queue and graph lifetime

The host custom handler explicitly captures only pointers, the current stream,
and scalar configuration. It does not capture Tensor owners. TorchNPU can
destroy an old COMPILE_AND_EXECUTE handler while holding its enqueue mutex when
an execution-ring slot is replaced; a Tensor wrapper release requiring the
Python GIL there can cause a lock cycle with another event producer.

Inputs and local allocations remain alive through enqueue. Scratch and output
allocate on the launch stream; same-stream queue ordering and allocator reuse
order both launches and downstream consumers before later writes. Captured
allocations use the graph's private memory pool. Direct callers using transient
inputs allocated on another stream must satisfy the normal synchronization and
`record_stream` lifetime contract.

Every execution reads current token slots and enabled entries. No CPU adapter
identity cache, host value readback, persistent scratch, or per-layer
synchronization is added. Existing W8A8 matmuls, LoRA projections, activation
rounding, W13 AllGather, W2 AllReduce, and D-Spark behavior are unchanged.

## Validation

The focused CPU tests are `tests/ut/lora/test_single_adapter.py`; native tests
are in `tests/e2e/nightly/single_node/ops/singlecard_ops/`.
`benchmarks/benchmark_moe_lora_single_adapter.py` checks native integer results,
changed-input graph replay, temporary-input lifetime, and routing microtimings.
It loads the explicitly supplied matching native library and requires an
explicit logical device.

The long-queue regression is
`benchmarks/validate_moe_lora_single_adapter_queue.py`. It discards intermediate
outputs across 4105 native calls, checks existing shrink/expand consumers,
adds another event-recording thread, and replays graphs with adapter mask
changes. Run it with `TASK_QUEUE_ENABLE=1`, explicitly assigned visible devices,
`--device`, `--library`, `--output`, and an external process timeout. The event
thread is a bounded standalone test; the actual HCCL watchdog interaction is
covered by the whole-model TP test.

End-to-end validation must use fresh OFF/ON engines with identical source and
native libraries. The target comparison is TP8, fully sharded rank16 LoRA,
W8A8, D-Spark5, prefix cache off, eight concurrent 8192-input/4096-output
requests, and max batched tokens 16384. Record actual native kernel hits,
throughput, TTFT, TPOT, speculative acceptance, and KV capacity. Isolated
routing timings do not establish a whole-model improvement.

### Validated target workload, 2026-10-07

The installed candidate was validated on Ascend A3, CANN 9.1.0, PyTorch 2.10,
torch_npu 2.10.0.post4, and vLLM 0.25.1. The matching native library passed
402 correctness checks, 17 Meta checks, and 96 changed-input graph replays.
The additional long-queue test passed 4105 submissions, 35 exact consumer
snapshots, concurrent event recording, and 24 drop-output graph replays.
The Python dispatch and cache tests passed 169 checks; the final binding
revision did not change those Python sources.

Both fresh TP8 engines used the same installed sources and libraries, the
configuration above, and identical eight prompts. Only the new flag differed.
Both completed all 32768 output tokens, with no cache hits or preemptions.

| Metric | Flag 0 | Flag 1 |
| --- | ---: | ---: |
| Output throughput, tokens/s | 316.02 | 378.05 |
| Mean TTFT, seconds | 7.497 | 7.512 |
| Mean TPOT, milliseconds | 19.120 | 16.313 |
| D-Spark draft acceptance | 75.30% | 89.33% |
| KV capacity, tokens | 157522 | 157554 |

The observed throughput increase was 19.63%, and mean TPOT decreased 14.68%.
Acceptance also increased 14.03 percentage points; only one of eight output
token sequences was identical across engines. Consequently, this single pair
does not isolate the fusion's end-to-end benefit or establish model quality.
The flag remains experimental and off by default.

The existing warmup profile confirmed 817 new kernel launches and 86 old
recovery launches for unsupported shapes in the enabled run. Profiling stopped
before formal measurement. Isolated R288 routing graph time decreased from
23.494 to 15.798 microseconds, or 32.76%; R48 decreased 16.24% and R98304
decreased 32.54%. These compare recovery plus one shared combined-index
operation with the new metadata operation, not the complete LoRA projection.

All 14 decode graphs captured successfully after the POD-handler revision.
Both serving processes stopped, and the assigned physical devices 8–15 were
released after validation.
