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
- Trace：`/opt/wqy2/temp/moe_lora_real_weight_profiler_trace_synced`
- 指标：op_statistic.csv 全部算子的 Total Time(us) 求和 / active(5)；warmup=5、active=5。

## 性能对比

| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |
| ---- | ----- | ----- | ------------- | -------- | ------ |
| 0 | [1, 8] | bfloat16 | 41.357 | 144.831 | 3.502 |
| 1 | [2, 8] | bfloat16 | 44.389 | 148.135 | 3.337 |
| 2 | [8, 8] | bfloat16 | 85.074 | 204.825 | 2.408 |
| 3 | [32, 8] | bfloat16 | 173.292 | 317.099 | 1.830 |
| 4 | [128, 8] | bfloat16 | 482.110 | 706.806 | 1.466 |
| 5 | [256, 8] | bfloat16 | 906.422 | 1244.433 | 1.373 |
| 6 | [512, 8] | bfloat16 | 1726.215 | 2294.598 | 1.329 |
| 7 | [1024, 8] | bfloat16 | 3349.227 | 4468.261 | 1.334 |

## 算子级拆分（每步中位数）

| Case | Shape | Path | Routing(us) | Sort(us) | BGMV shrink(us) | BGMV expand(us) | Other(us) |
| ---- | ----- | ---- | ----------- | -------- | --------------- | --------------- | --------- |
| 0 | [1, 8] | custom | 1.632 | 0.000 | 14.148 | 9.392 | 16.176 |
| 0 | [1, 8] | baseline | 0.000 | 35.437 | 15.744 | 10.228 | 82.942 |
| 1 | [2, 8] | custom | 1.768 | 0.000 | 15.280 | 10.780 | 16.580 |
| 1 | [2, 8] | baseline | 0.000 | 33.817 | 17.224 | 11.752 | 85.705 |
| 2 | [8, 8] | custom | 2.944 | 0.000 | 28.397 | 20.960 | 32.681 |
| 2 | [8, 8] | baseline | 0.000 | 37.789 | 35.133 | 23.452 | 108.003 |
| 3 | [32, 8] | custom | 8.028 | 0.000 | 79.614 | 46.621 | 38.949 |
| 3 | [32, 8] | baseline | 0.000 | 56.593 | 86.146 | 48.825 | 125.535 |
| 4 | [128, 8] | custom | 25.629 | 0.000 | 275.814 | 131.975 | 48.497 |
| 4 | [128, 8] | baseline | 0.000 | 139.419 | 281.946 | 134.947 | 150.267 |
| 5 | [256, 8] | custom | 48.529 | 0.000 | 546.107 | 246.597 | 65.305 |
| 5 | [256, 8] | baseline | 0.000 | 263.153 | 553.639 | 251.201 | 177.439 |
| 6 | [512, 8] | custom | 94.126 | 0.000 | 1076.706 | 476.150 | 79.438 |
| 6 | [512, 8] | baseline | 0.000 | 521.502 | 1084.226 | 481.538 | 207.704 |
| 7 | [1024, 8] | custom | 176.284 | 0.000 | 2129.859 | 940.027 | 103.454 |
| 7 | [1024, 8] | baseline | 0.000 | 1103.274 | 2139.811 | 947.127 | 276.814 |

## 全量汇总

| 指标 | 值 |
| ---- | -- |
| 用例数 | 8 |
| 平均加速比 | 2.072 |
| 自定义算子更优 | 8 |
| 标杆更优 | 0 |

### 按数据类型汇总

| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |
| ----- | ------ | ---------- | -------------- | -------- |
| bfloat16 | 8 | 2.072 | 8 | 0 |

## 简短分析

- 数据直接读取自 Qwen3.5-35B-A3B checkpoint；tensor key、shape、dtype 和内容指纹均在报告头记录。
- Router top-k 由真实 router 权重与 checkpoint 权重行构造的确定性 hidden 输入计算，不是随机 top-k，也不是线上抓取的真实激活。
- 每个 profiler step 均在 `prof.step()` 前同步 NPU，避免异步 warmup 工作泄漏到 active 统计。
- 由于本机没有匹配的 LoRA adapter，rank-16 A/B 是 checkpoint expert 权重子块；结果用于评估 kernel 路径和内存形态，不代表某个训练 adapter 的模型精度。
- Qwen3.5-27B 是 dense 模型，没有 expert/routing，因此不能用于验证 moe_lora_build_combined_idx 的真实模型端到端收益。
