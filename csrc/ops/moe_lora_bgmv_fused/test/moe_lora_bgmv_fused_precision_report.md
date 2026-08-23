# `moe_lora_bgmv_fused` 精度验证报告

- 测试平台：Ascend 910B3（物理 NPU 7）
- 测试时间：2026-08-23T09:04:14+00:00
- 参考实现：PyTorch CPU FP32 两阶段 BMM，最终 Cast 回输出 dtype。

## 总览

| 指标 | 值 |
| --- | ---: |
| 总用例数 | 44 |
| 通过数 | 44 |
| 失败数 | 0 |
| 通过率 | 100.00% |

## 精度标准

相对误差按 `abs(actual - golden) / (abs(golden) + 1e-7)` 计算；
FP16 要求 MERE < 2^-10 且 MARE < 10 * 2^-10，BF16 要求
MERE < 2^-7 且 MARE < 10 * 2^-7。返回 Tensor 必须与 y alias，
slice 外数据必须逐元素不变。

## 用例结果

| # | 类别 | 描述 | Shape(M,H,O,Y) | dtype | index dtype | MERE | MARE | MaxAbs | alias/slice | 结果 |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | Grouped | single 4-row group | [4, 64, 64, 64] | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 1 | Grouped | single 4-row group | [4, 64, 64, 64] | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 2 | Grouped | two groups with slice | [8, 128, 96, 160] | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 2 | Grouped | two groups with slice | [8, 128, 96, 160] | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 3 | MoE | small W2-like | [16, 512, 512, 512] | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 3 | MoE | small W2-like | [16, 512, 512, 512] | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 4 | MoE | Qwen W13 dimensions | [32, 2048, 1024, 1024] | float16 | int32 | 6.145e-08 | 9.488e-04 | 1.953e-03 | True/True | PASS |
| 4 | MoE | Qwen W13 dimensions | [32, 2048, 1024, 1024] | bfloat16 | int32 | 1.344e-07 | 4.405e-03 | 7.812e-03 | True/True | PASS |
| 5 | MoE | Qwen W2 dimensions | [32, 512, 2048, 2048] | float16 | int32 | 1.183e-08 | 7.752e-04 | 9.766e-04 | True/True | PASS |
| 5 | MoE | Qwen W2 dimensions | [32, 512, 2048, 2048] | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 6 | Mixed | mixed adapters inside group | [64, 768, 1000, 1100] | float16 | int64 | 1.269e-08 | 8.937e-04 | 9.766e-04 | True/True | PASS |
| 6 | Mixed | mixed adapters inside group | [64, 768, 1000, 1100] | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 7 | Grouped | 128 W13 rows | [128, 2048, 1024, 1024] | float16 | int32 | 7.373e-08 | 9.737e-04 | 1.953e-03 | True/True | PASS |
| 7 | Grouped | 128 W13 rows | [128, 2048, 1024, 1024] | bfloat16 | int32 | 6.267e-08 | 4.149e-03 | 7.812e-03 | True/True | PASS |
| 8 | Fallback | no consecutive equal index | [256, 512, 2048, 2048] | float16 | int64 | 1.769e-08 | 8.084e-04 | 9.766e-04 | True/True | PASS |
| 8 | Fallback | no consecutive equal index | [256, 512, 2048, 2048] | bfloat16 | int64 | 1.200e-08 | 6.289e-03 | 7.812e-03 | True/True | PASS |
| 9 | Small | minimum supported dimensions | [1, 17, 17, 17] | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 9 | Small | minimum supported dimensions | [1, 17, 17, 17] | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 10 | Small | unaligned H/O and slice | [3, 17, 19, 23] | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 10 | Small | unaligned H/O and slice | [3, 17, 19, 23] | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 11 | Boundary | all rows disabled | [4, 31, 33, 40] | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 11 | Boundary | all rows disabled | [4, 31, 33, 40] | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 12 | Boundary | tail row after one group | [5, 2048, 2048, 2048] | float16 | int64 | 5.456e-08 | 5.587e-04 | 9.766e-04 | True/True | PASS |
| 12 | Boundary | tail row after one group | [5, 2048, 2048, 2048] | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 13 | Large | integration threshold W13 | [512, 2048, 1024, 1024] | float16 | int32 | 6.207e-08 | 9.681e-04 | 1.953e-03 | True/True | PASS |
| 13 | Large | integration threshold W13 | [512, 2048, 1024, 1024] | bfloat16 | int32 | 4.652e-08 | 7.519e-03 | 1.562e-02 | True/True | PASS |
| 14 | Large | 1024-row W2 | [1024, 512, 2048, 2048] | float16 | int32 | 1.480e-08 | 8.123e-04 | 9.766e-04 | True/True | PASS |
| 14 | Large | 1024-row W2 | [1024, 512, 2048, 2048] | bfloat16 | int32 | 1.174e-08 | 6.369e-03 | 7.812e-03 | True/True | PASS |
| 15 | Large | 4096-row W13 | [4096, 2048, 1024, 1024] | float16 | int32 | 6.698e-08 | 9.747e-04 | 1.953e-03 | True/True | PASS |
| 15 | Large | 4096-row W13 | [4096, 2048, 1024, 1024] | bfloat16 | int32 | 7.622e-08 | 7.812e-03 | 1.562e-02 | True/True | PASS |
| 16 | Large | 8192-row W2 | [8192, 512, 2048, 2048] | float16 | int32 | 1.466e-08 | 8.224e-04 | 9.766e-04 | True/True | PASS |
| 16 | Large | 8192-row W2 | [8192, 512, 2048, 2048] | bfloat16 | int32 | 1.034e-08 | 6.579e-03 | 7.812e-03 | True/True | PASS |
| 17 | Boundary | all disabled indices | [9, 63, 65, 96] | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 17 | Boundary | all disabled indices | [9, 63, 65, 96] | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 18 | Boundary | minimum valid weight | [9, 63, 65, 96] | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 18 | Boundary | minimum valid weight | [9, 63, 65, 96] | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 19 | Boundary | maximum valid weight | [9, 63, 65, 96] | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 19 | Boundary | maximum valid weight | [9, 63, 65, 96] | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 20 | Boundary | zero scale | [9, 63, 65, 96] | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 20 | Boundary | zero scale | [9, 63, 65, 96] | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 21 | Boundary | negative scale | [9, 63, 65, 96] | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 21 | Boundary | negative scale | [9, 63, 65, 96] | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 22 | Boundary | last valid slice | [9, 63, 33, 97] | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 22 | Boundary | last valid slice | [9, 63, 33, 97] | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |

## 按 dtype 汇总

| dtype | 用例数 | 通过数 | 失败数 |
| --- | ---: | ---: | ---: |
| bfloat16 | 22 | 22 | 0 |
| float16 | 22 | 22 | 0 |

## 关键发现

1. 共 44 个 FP16/BF16 与 int32/int64 组合，覆盖 1-row fallback、4-row 复用和尾块。
2. expert-sorted 快速路径覆盖 22 例，mixed/alternating fallback 覆盖 6 例。
3. 非 32B 对齐 H/O、slice offset、负/零 scale、全 -1 index 和最大合法 index 均纳入验证。
4. 每个用例同时检查返回 alias 和 slice 外逐元素不变，避免只验证数值而漏掉 in-place 语义。
