# Qwen3.5 MoE 真实权重性能评估

- Checkpoint：`/home/models/Qwen3.5-35B-A3B`
- Model type：`qwen3_5_moe_text`
- 真实配置：hidden=2048、expert intermediate=512、experts=256、top-k=8、BF16。
- 低秩测试 rank：16。A/B 来自 layer-0 真实 expert 权重的 rank 子块，不是训练得到的 LoRA adapter。
- Router key：`model.language_model.layers.0.mlp.gate.weight`，shape=[256, 2048]，fingerprint=`d55bd73f2cfdb0bd`。
- W13 key：`model.language_model.layers.0.mlp.experts.gate_up_proj`，A/B shape=[256, 16, 2048]/[256, 1024, 16]，fingerprint=`7c98ae45ea311629`/`55db3ee58409e318`。
- W2 key：`model.language_model.layers.0.mlp.experts.down_proj`，A/B shape=[256, 16, 512]/[256, 2048, 16]，fingerprint=`181af914084c3f26`/`71c0edc469d32fc8`。
- 标杆：AiCPU argsort 路由恢复 + W13/W2 各自生成 int64 combined_idx + 原 int64 BGMV。
- 自定义路径：一次 AscendC fused routing + W13/W2 复用 int32 combined_idx + int32 BGMV。
- 独立采集：每条路径 3 轮；每轮 repeat=1，奇偶轮交换 custom/baseline 顺序，表中为中位数。
- 可见物理 NPU：`ASCEND_RT_VISIBLE_DEVICES=7`。
- 用例：`/opt/wqy2/temp/vllm-ascend-ds-lora-moe-v25/csrc/ops/moe_lora_build_combined_idx/test/moe_lora_build_combined_idx_real_weight_perf_cases.jsonl`
- Trace：`/opt/wqy2/temp/moe_lora_real_weight_profiler_trace_final`
- 指标：op_statistic.csv 全部算子的 Total Time(us) 求和 / active(5)；warmup=5、active=5。

## 性能对比

| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |
| ---- | ----- | ----- | ------------- | -------- | ------ |
| 0 | [1, 8] | bfloat16 | 41.969 | 146.071 | 3.480 |
| 1 | [2, 8] | bfloat16 | 44.777 | 155.079 | 3.463 |
| 2 | [8, 8] | bfloat16 | 90.866 | 218.197 | 2.401 |
| 3 | [32, 8] | bfloat16 | 181.952 | 336.623 | 1.850 |
| 4 | [128, 8] | bfloat16 | 489.654 | 716.387 | 1.463 |
| 5 | [256, 8] | bfloat16 | 915.430 | 1261.053 | 1.378 |
| 6 | [512, 8] | bfloat16 | 1730.094 | 2303.870 | 1.332 |
| 7 | [1024, 8] | bfloat16 | 3406.804 | 4535.691 | 1.331 |

## 算子级拆分（每步中位数）

| Case | Shape | Path | Routing(us) | Sort(us) | BGMV shrink(us) | BGMV expand(us) | Other(us) |
| ---- | ----- | ---- | ----------- | -------- | --------------- | --------------- | --------- |
| 0 | [1, 8] | custom | 1.600 | 0.000 | 14.080 | 9.528 | 16.792 |
| 0 | [1, 8] | baseline | 0.000 | 34.069 | 16.672 | 11.112 | 84.242 |
| 1 | [2, 8] | custom | 1.752 | 0.000 | 15.368 | 10.640 | 16.948 |
| 1 | [2, 8] | baseline | 0.000 | 35.017 | 18.312 | 12.864 | 88.845 |
| 2 | [8, 8] | custom | 2.956 | 0.000 | 28.357 | 22.492 | 36.849 |
| 2 | [8, 8] | baseline | 0.000 | 37.117 | 38.481 | 29.649 | 113.263 |
| 3 | [32, 8] | custom | 7.788 | 0.000 | 79.618 | 49.381 | 45.237 |
| 3 | [32, 8] | baseline | 0.000 | 56.005 | 90.066 | 55.477 | 134.631 |
| 4 | [128, 8] | custom | 25.233 | 0.000 | 276.222 | 134.135 | 54.173 |
| 4 | [128, 8] | baseline | 0.000 | 138.015 | 284.782 | 142.247 | 151.935 |
| 5 | [256, 8] | custom | 48.381 | 0.000 | 547.591 | 247.773 | 71.749 |
| 5 | [256, 8] | baseline | 0.000 | 266.045 | 554.919 | 256.757 | 184.180 |
| 6 | [512, 8] | custom | 91.986 | 0.000 | 1075.438 | 476.830 | 85.854 |
| 6 | [512, 8] | baseline | 0.000 | 519.078 | 1084.106 | 486.818 | 213.748 |
| 7 | [1024, 8] | custom | 175.688 | 0.000 | 2182.624 | 939.367 | 108.934 |
| 7 | [1024, 8] | baseline | 0.000 | 1108.834 | 2194.744 | 949.879 | 282.598 |

## 全量汇总

| 指标 | 值 |
| ---- | -- |
| 用例数 | 8 |
| 平均加速比 | 2.087 |
| 自定义算子更优 | 8 |
| 标杆更优 | 0 |

### 按数据类型汇总

| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |
| ----- | ------ | ---------- | -------------- | -------- |
| bfloat16 | 8 | 2.087 | 8 | 0 |

## 简短分析

- 数据直接读取自 Qwen3.5-35B-A3B checkpoint；tensor key、shape、dtype 和内容指纹均在报告头记录。
- Router top-k 由真实 router 权重与 checkpoint 权重行构造的确定性 hidden 输入计算，不是随机 top-k，也不是线上抓取的真实激活。
- 每个 profiler step 均在 `prof.step()` 前同步 NPU，避免异步 warmup 工作泄漏到 active 统计。
- 由于本机没有匹配的 LoRA adapter，rank-16 A/B 是 checkpoint expert 权重子块；结果用于评估 kernel 路径和内存形态，不代表某个训练 adapter 的模型精度。
- Qwen3.5-27B 是 dense 模型，没有 expert/routing，因此不能用于验证 moe_lora_build_combined_idx 的真实模型端到端收益。
