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

- You can enable LoRA with dense or mixture-of-experts (MoE) models ([PR #10977](https://github.com/vllm-project/vllm-ascend/pull/10977)). This branch also includes W8A8 dynamic MoE LoRA and the existing AlltoAll EP path. The optimization below is limited to TP fully-sharded AllGather.

## TP fully-sharded W8A8 MoE optimization

This branch backports the W8A8 MoE LoRA optimizations for AllGather with
`--fully-sharded-loras` and expert parallelism disabled. The existing
non-sharded and AlltoAll/EP execution paths retain their floating-point LoRA
activation contract.

The optimized path routes INT8 activations together with one FP32 scale per
row. Base expert GMMs and LoRA A projections share these quantized inputs;
LoRA A dequantizes inside the native kernel. For supported SiLU shapes, gate
and up A weights share a packed allocation, both projections use one kernel
and one TP gather, and B projection, base addition, clamped SwiGLU and dynamic
quantization run in one kernel. TP communication between A and B, adapter
masking and the w2 output partition offsets are preserved.

Fusion is enabled for intermediate widths divisible by 32 and at most 256,
power-of-two LoRA ranks of at least 8, and width times rank at most 8192.
Other shapes and activation functions use the separate projection,
activation and quantization path. GMM2 and w2 LoRA share the resulting INT8
activation and FP32 scale. Quantized LoRA inputs may introduce quantization
error relative to the old floating-point LoRA path.

Direct INT8 MLP callers must supply `dynamic_scale` on the same device and
an explicit BF16/FP16 `output_dtype`. Normal execution propagates
`moe_config.in_dtype` through the 0.25 runtime payload. Floating-point input
remains supported.

The experimental grouped Prefill optimization is opt-in:

```bash
export VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL=1
```

The default is `0`; valid values are `0` and `1`. It sorts the adapter/expert
mapping independently for every batch, reuses weights across nearby routed
rows, and restores original row order before TP communication and GMM2.
It requires at least 8192 routed rows and eight rows per configured
adapter/expert group, hidden width at most 4096 and divisible by 64, full
rank 8/16/32, and a supported packed-A cache size. Decode and unsupported
layouts retain the ordinary INT8 path. The mapping uses fixed shapes, so
adapter changes can be replayed within captured graphs. Batches with many
inactive adapter rows can regress because sorting and restoration still
process those rows. Keep this option disabled unless representative Prefill
benchmarks show a benefit; operator measurements do not establish a general
end-to-end service speedup.

The effective grouped Prefill flag is included in both compiler and outer
vLLM AOT cache keys, so toggling it requires a distinct compiled artifact.
This single flag registration is a dependency of the optional grouped path.

Rebuild the native extension after applying this backport. No 0.28 EP,
DSpark, runner, KV-cache or general compilation-cache changes are required.
