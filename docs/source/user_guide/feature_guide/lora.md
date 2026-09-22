# LoRA Adapters Guide

## Overview

Like vLLM, vllm-ascend supports LoRA as well. The usage and more details can be found in [vLLM official document](https://docs.vllm.ai/en/latest/features/lora/).

You can refer to [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/) to find which models support LoRA in vLLM.

You can run LoRA with ACLGraph mode now. Please refer to [Graph Mode Guide](./graph_mode.md) for better LoRA performance.

Address for downloading models:

- base model: <https://www.modelscope.cn/models/vllm-ascend/Llama-2-7b-hf/files>
- loRA model: <https://www.modelscope.cn/models/vllm-ascend/llama-2-7b-sql-lora-test/files>

## Example

We provide a simple LoRA example here, which enables the ACLGraph mode by default.

```shell
vllm serve meta-llama/Llama-2-7b \
    --enable-lora \
    --lora-modules '{"name": "sql-lora", "path": "/path/to/lora", "base_model_name": "meta-llama/Llama-2-7b"}'
```

## Note

- We have implemented LoRA-related AscendC operators, such as bgmv_shrink, bgmv_expand, sgmv_shrink and sgmv_expand. You can find them under the `csrc/kernels` directory of [vllm-ascend repo](https://github.com/vllm-project/vllm-ascend/tree/main/csrc/kernels).

LoRA is supported for both dense and mixture-of-experts (MoE) models. The current MoE support status is as follows:

| MoE mode | Tensor parallel (AllGather) | Expert parallel (All-to-All) |
| --- | --- | --- |
| Non-quantized | Supported | Supported |
| W8A8 dynamic quantization | Supported | Supported |

Other MoE quantization methods, Fused MC2, and dynamic EPLB are not supported with LoRA.

### W8A8 AllGather activations

The W8A8 AllGather path shares INT8 activations and FP32 per-token scales between
the base experts and LoRA. LoRA A projections dequantize inside the Ascend C
kernel and accumulate in FP32. Fully sharded LoRA retains TP communication
between its A and B projections. This introduces activation quantization error
relative to computing LoRA from the original floating-point activations.

For SiLU/SwiGLU with fusion enabled, the w13 B projections, base addition,
configured `swiglu_limit`, activation, optional routing weights and dynamic
quantization use one kernel. Gate precedes up; a positive limit clips only the
gate upper bound and clips up in both directions. The fused local intermediate
width is at most 256 and divisible by 32, the full rank is a power of two
of at least 8, and width times rank is at most 8192. This conservative
dispatch region avoids measured Decode regressions from wide projections.
The native fused operator supports widths up to 8192 and ranks up to 512;
other activations, shapes outside the dispatch region and LoRA with a single
packed w13 projection use the separate stages. The W8A8 All-to-All path retains
its existing floating-point LoRA inputs.

For two-slice AllGather MoE with per-expert A weights, gate/up share one
preallocated A-weight buffer. Adapter loading and reset update contiguous views
of that buffer in place. The fused path computes both A projections with one
INT8 shrink kernel and, for rank-sharded LoRA, uses one AllGather along the row
dimension. The B/activation/quantization kernel consumes the resulting
`[TP, rows, 2 * local_rank]` layout directly, reusing a vector-gathered gate/up
cache in UB. This avoids a separate transpose and repeated A-input
dequantization. Communication volume and FP32 accumulation semantics are
unchanged. Expert-parallel, single-slice and shared-factor allocation retain
their existing layouts.

Compare the paired local kernels with:

```shell
ASCEND_RT_VISIBLE_DEVICES=8 python benchmarks/moe_lora_w13_pair.py --output pair.json
```

`--local-rank`, `--tp-size`, `--dtype` and `--inactive-fraction` select shapes and
adapter masks; `--profile-dir traces` collects operator traces. This benchmark
simulates the post-AllGather layout and excludes communication. Measure the full
TP MLP and service separately before interpreting local kernel savings as
end-to-end speedups.

Direct `MoEMlpComputeInput` callers passing INT8 must provide `dynamic_scale`
(FP32, one value per row, on the same device) and a BF16/FP16 `output_dtype`.
Normal model execution obtains the output dtype from `moe_config.in_dtype`.
Floating-point input remains supported without a supplied dynamic scale.

Run the local kernel benchmark with:

```shell
ASCEND_RT_VISIBLE_DEVICES=8 python benchmarks/moe_lora_int8.py --graph --output perf.json
```

This benchmark excludes routing, base GMMs, TP communication and service latency.
Add `--profile-dir traces` to collect separate fused and unfused NPU traces.

If the driver cannot allocate pinned host memory while loading large adapters,
set `VLLM_ASCEND_DISABLE_PIN_MEMORY=1` before starting the engine. The default is
`0`, which preserves pinning. This opt-out applies to host allocations across
the engine and can reduce CPU-to-NPU transfer performance.
