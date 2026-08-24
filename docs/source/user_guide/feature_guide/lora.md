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

- You can enable LoRA with dense or mixture-of-experts (MoE) models now ([PR #10977](https://github.com/vllm-project/vllm-ascend/pull/10977)). Dynamic W8A8 MoE supports LoRA on the AllGather TP and AlltoAll EP paths. MC2, FusedMC2, and dynamic EPLB are not supported for quantized MoE LoRA.

### Fully sharded LoRA with W8A8 MoE

Dynamic W8A8 MoE can use `--fully-sharded-loras` together with tensor parallelism. Fully sharded MoE LoRA and expert parallelism partition the same TP group in incompatible ways, so do not combine `--fully-sharded-loras` with `--enable-expert-parallel`.

```shell
VLLM_ASCEND_ENABLE_FUSED_MC2=0 vllm serve vllm-ascend/Qwen3-30B-A3B-W8A8 \
    --quantization ascend \
    --tensor-parallel-size 2 \
    --enable-lora \
    --fully-sharded-loras \
    --lora-modules '{"name": "sql-lora", "path": "/path/to/lora"}'
```

For a gated 2D MoE layer, let `H` be the hidden size, `I` the expert intermediate size, and `TP` the tensor-parallel size. Fully sharding reduces each rank's routed-expert LoRA weight bank from a value proportional to `H + I / TP` to `(H + I) / TP`. The percentage saved is:

```text
H * (1 - 1 / TP) / (H + I / TP)
```

For Qwen3-30B-A3B (`H=2048`, `I=768`, 48 layers, 128 experts), BF16 LoRA weights with `max_loras=4` and `max_lora_rank=16` use about 5.34 GiB per rank for the routed-expert bank at TP2. Fully sharding reduces this to about 3.09 GiB, saving 2.25 GiB (42.1%). At TP4 the saving is about 3.38 GiB (68.6%), and at TP8 it is about 3.94 GiB (83.6%). These values cover the preallocated routed-expert LoRA weights only; dense targets, allocator overhead, graph pools, and temporary communication buffers are separate.

Fully sharding introduces TP collectives between the LoRA A and B projections: W13 uses AllGather and W2 uses AllReduce. Measure request latency and throughput on the target NPU topology before enabling it by default; the memory saving does not guarantee a latency improvement for small batches.
