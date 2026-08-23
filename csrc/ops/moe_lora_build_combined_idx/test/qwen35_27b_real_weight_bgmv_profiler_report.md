# Qwen3.5-27B Dense 真实权重 BGMV 性能评估

- Checkpoint：`/opt/wqy2/model-weights/Qwen3.5-27B`
- Model type：`qwen3_5_text`（Dense，无 MoE router/expert）。
- 真实配置：hidden=5120、intermediate=17408、BF16。
- 低秩测试 rank：16；A/B 为真实 Dense MLP 权重子块，不是训练得到的 LoRA adapter。
- Gate/Up/Down keys：`model.language_model.layers.0.mlp.gate_proj.weight` / `model.language_model.layers.0.mlp.up_proj.weight` / `model.language_model.layers.0.mlp.down_proj.weight`。
- 输入 shape=[1024, 5120]，fingerprint=`7731dccbb0ca2289`。
- W13 A/B shape=[16, 5120]/[34816, 16]，fingerprint=`f8ea216410747fb7`/`0a4d05316c3ca316`。
- W2 A/B shape=[16, 17408]/[5120, 16]，fingerprint=`15a5b4ac7b531c7b`/`7d67a37f16d3d60f`。
- 自定义路径：int32 BGMV indices；标杆路径：原 int64 BGMV indices。两条路径不含 routing。
- 独立采集：每条路径 3 轮；每轮 repeat=1，奇偶轮交换 int32/int64 顺序，表中为中位数。
- 可见物理 NPU：`ASCEND_RT_VISIBLE_DEVICES=7`。
- 用例：`/opt/wqy2/temp/vllm-ascend-ds-lora-moe-v25/csrc/ops/moe_lora_build_combined_idx/test/qwen35_27b_real_weight_bgmv_perf_cases.jsonl`
- Trace：`/opt/wqy2/temp/qwen35_27b_real_weight_bgmv_trace`
- 指标：op_statistic.csv 全部算子的 Total Time(us) 求和 / active(5)；warmup=5、active=5。

## 性能对比

| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |
| ---- | ----- | ----- | ------------- | -------- | ------ |
| 0 | [1, 5120] | bfloat16 | 88.346 | 88.322 | 1.000 |
| 1 | [2, 5120] | bfloat16 | 94.434 | 94.430 | 1.000 |
| 2 | [8, 5120] | bfloat16 | 107.098 | 113.790 | 1.062 |
| 3 | [32, 5120] | bfloat16 | 145.455 | 139.107 | 0.956 |
| 4 | [128, 5120] | bfloat16 | 363.671 | 363.655 | 1.000 |
| 5 | [256, 5120] | bfloat16 | 591.052 | 598.824 | 1.013 |
| 6 | [512, 5120] | bfloat16 | 1041.833 | 1051.005 | 1.009 |
| 7 | [1024, 5120] | bfloat16 | 2087.242 | 2097.542 | 1.005 |

## 算子级拆分（每步中位数）

| Case | Shape | Indices | BGMV shrink(us) | BGMV expand(us) | Other(us) |
| ---- | ----- | ------- | --------------- | --------------- | --------- |
| 0 | [1, 5120] | int32 | 42.805 | 32.013 | 13.528 |
| 0 | [1, 5120] | int64 | 42.965 | 31.917 | 13.472 |
| 1 | [2, 5120] | int32 | 42.377 | 32.025 | 19.968 |
| 1 | [2, 5120] | int64 | 42.781 | 31.909 | 19.796 |
| 2 | [8, 5120] | int32 | 43.997 | 33.609 | 29.501 |
| 2 | [8, 5120] | int64 | 44.889 | 33.717 | 35.253 |
| 3 | [32, 5120] | int32 | 52.337 | 40.125 | 53.085 |
| 3 | [32, 5120] | int64 | 51.881 | 40.193 | 47.061 |
| 4 | [128, 5120] | int32 | 170.175 | 123.730 | 69.653 |
| 4 | [128, 5120] | int64 | 169.827 | 124.090 | 69.577 |
| 5 | [256, 5120] | int32 | 291.466 | 210.216 | 89.542 |
| 5 | [256, 5120] | int64 | 290.722 | 210.356 | 97.746 |
| 6 | [512, 5120] | int32 | 531.387 | 380.356 | 130.051 |
| 6 | [512, 5120] | int64 | 529.831 | 381.136 | 139.919 |
| 7 | [1024, 5120] | int32 | 1055.957 | 748.251 | 283.102 |
| 7 | [1024, 5120] | int64 | 1056.305 | 749.443 | 291.458 |

## 全量汇总

| 指标 | 值 |
| ---- | -- |
| 用例数 | 8 |
| 平均加速比 | 1.006 |
| 自定义算子更优 | 4 |
| 标杆更优 | 4 |

### 按数据类型汇总

| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |
| ----- | ------ | ---------- | -------------- | -------- |
| bfloat16 | 8 | 1.006 | 4 | 4 |

## 简短分析

- 数据值和维度直接来自 Qwen3.5-27B layer-0 Dense MLP checkpoint，输入也取自真实 gate_proj 权重行。
- 该模型没有 expert/router，所以本报告仅回答 int32/int64 BGMV 分支差异，不能验证 MoE routing 融合收益。
- 每个 profiler step 均在 `prof.step()` 前同步 NPU，避免异步 warmup 工作泄漏到 active 统计。
- A/B 是 checkpoint 权重子块而非训练 LoRA adapter，结果是 kernel 性能测试，不是 adapter 模型精度测试。
