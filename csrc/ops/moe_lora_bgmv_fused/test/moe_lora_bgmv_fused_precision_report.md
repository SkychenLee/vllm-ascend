# `moe_lora_bgmv_fused` 精度验证报告

- 测试平台：Ascend 910B3。
- 测试时间：2026-08-23T15:09:58+00:00
- 参考实现：PyTorch CPU FP32 两阶段 BMM，最终 Cast 回输出 dtype。

## 总览

| 指标 | 值 |
| --- | ---: |
| 总用例数 | 86 |
| 通过数 | 86 |
| 失败数 | 0 |
| 通过率 | 100.00% |

## 精度标准

相对误差按 `abs(actual - golden) / (abs(golden) + 1e-7)` 计算；
FP16 要求 MERE < 2^-10 且 MARE < 10 * 2^-10，BF16 要求
MERE < 2^-7 且 MARE < 10 * 2^-7。返回 Tensor 必须与 y alias，
slice 外数据必须逐元素不变。

## 用例结果

| # | 类别 | 描述 | Shape(M,H,O,Y) | rank | dtype | index dtype | MERE | MARE | MaxAbs | alias/slice | 结果 |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | Grouped | single 4-row group | [4, 64, 64, 64] | 16 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 1 | Grouped | single 4-row group | [4, 64, 64, 64] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 2 | Grouped | two groups with slice | [8, 128, 96, 160] | 16 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 2 | Grouped | two groups with slice | [8, 128, 96, 160] | 16 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 3 | MoE | small W2-like | [16, 512, 512, 512] | 16 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 3 | MoE | small W2-like | [16, 512, 512, 512] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 4 | MoE | Qwen W13 dimensions | [32, 2048, 1024, 1024] | 16 | float16 | int32 | 6.145e-08 | 9.488e-04 | 1.953e-03 | True/True | PASS |
| 4 | MoE | Qwen W13 dimensions | [32, 2048, 1024, 1024] | 16 | bfloat16 | int32 | 1.344e-07 | 4.405e-03 | 7.812e-03 | True/True | PASS |
| 5 | MoE | Qwen W2 dimensions | [32, 512, 2048, 2048] | 16 | float16 | int32 | 1.183e-08 | 7.752e-04 | 9.766e-04 | True/True | PASS |
| 5 | MoE | Qwen W2 dimensions | [32, 512, 2048, 2048] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 6 | Mixed | mixed adapters inside group | [64, 768, 1000, 1100] | 16 | float16 | int64 | 1.269e-08 | 8.937e-04 | 9.766e-04 | True/True | PASS |
| 6 | Mixed | mixed adapters inside group | [64, 768, 1000, 1100] | 16 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 7 | Grouped | 128 W13 rows | [128, 2048, 1024, 1024] | 16 | float16 | int32 | 7.373e-08 | 9.737e-04 | 1.953e-03 | True/True | PASS |
| 7 | Grouped | 128 W13 rows | [128, 2048, 1024, 1024] | 16 | bfloat16 | int32 | 6.267e-08 | 4.149e-03 | 7.812e-03 | True/True | PASS |
| 8 | Fallback | no consecutive equal index | [256, 512, 2048, 2048] | 16 | float16 | int64 | 1.769e-08 | 8.084e-04 | 9.766e-04 | True/True | PASS |
| 8 | Fallback | no consecutive equal index | [256, 512, 2048, 2048] | 16 | bfloat16 | int64 | 1.200e-08 | 6.289e-03 | 7.812e-03 | True/True | PASS |
| 9 | Small | minimum supported dimensions | [1, 17, 17, 17] | 16 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 9 | Small | minimum supported dimensions | [1, 17, 17, 17] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 10 | Small | unaligned H/O and slice | [3, 17, 19, 23] | 16 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 10 | Small | unaligned H/O and slice | [3, 17, 19, 23] | 16 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 11 | Boundary | all rows disabled | [4, 31, 33, 40] | 16 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 11 | Boundary | all rows disabled | [4, 31, 33, 40] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 12 | Boundary | tail row after one group | [5, 2048, 2048, 2048] | 16 | float16 | int64 | 5.456e-08 | 5.587e-04 | 9.766e-04 | True/True | PASS |
| 12 | Boundary | tail row after one group | [5, 2048, 2048, 2048] | 16 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 13 | Large | integration threshold W13 | [512, 2048, 1024, 1024] | 16 | float16 | int32 | 6.207e-08 | 9.681e-04 | 1.953e-03 | True/True | PASS |
| 13 | Large | integration threshold W13 | [512, 2048, 1024, 1024] | 16 | bfloat16 | int32 | 4.652e-08 | 7.519e-03 | 1.562e-02 | True/True | PASS |
| 14 | Large | 1024-row W2 | [1024, 512, 2048, 2048] | 16 | float16 | int32 | 1.480e-08 | 8.123e-04 | 9.766e-04 | True/True | PASS |
| 14 | Large | 1024-row W2 | [1024, 512, 2048, 2048] | 16 | bfloat16 | int32 | 1.174e-08 | 6.369e-03 | 7.812e-03 | True/True | PASS |
| 15 | Large | 4096-row W13 | [4096, 2048, 1024, 1024] | 16 | float16 | int32 | 6.698e-08 | 9.747e-04 | 1.953e-03 | True/True | PASS |
| 15 | Large | 4096-row W13 | [4096, 2048, 1024, 1024] | 16 | bfloat16 | int32 | 7.622e-08 | 7.812e-03 | 1.562e-02 | True/True | PASS |
| 16 | Large | 8192-row W2 | [8192, 512, 2048, 2048] | 16 | float16 | int32 | 1.466e-08 | 8.224e-04 | 9.766e-04 | True/True | PASS |
| 16 | Large | 8192-row W2 | [8192, 512, 2048, 2048] | 16 | bfloat16 | int32 | 1.034e-08 | 6.579e-03 | 7.812e-03 | True/True | PASS |
| 17 | DeepSeekV4 | TP8 W13 slice | [512, 4096, 256, 512] | 16 | float16 | int32 | 6.055e-08 | 8.210e-04 | 1.953e-03 | True/True | PASS |
| 17 | DeepSeekV4 | TP8 W13 slice | [512, 4096, 256, 512] | 16 | bfloat16 | int32 | 3.987e-08 | 5.376e-03 | 1.562e-02 | True/True | PASS |
| 18 | DeepSeekV4 | TP8 W2 | [512, 256, 4096, 4096] | 16 | float16 | int64 | 5.960e-09 | 8.857e-04 | 9.766e-04 | True/True | PASS |
| 18 | DeepSeekV4 | TP8 W2 | [512, 256, 4096, 4096] | 16 | bfloat16 | int64 | 3.179e-09 | 6.667e-03 | 7.812e-03 | True/True | PASS |
| 19 | DeepSeekV4 | TP1 W13 slice | [512, 4096, 2048, 4096] | 16 | float16 | int64 | 5.794e-08 | 8.403e-04 | 1.953e-03 | True/True | PASS |
| 19 | DeepSeekV4 | TP1 W13 slice | [512, 4096, 2048, 4096] | 16 | bfloat16 | int64 | 4.690e-08 | 6.410e-03 | 1.562e-02 | True/True | PASS |
| 20 | DeepSeekV4 | TP1 W2 | [512, 2048, 4096, 4096] | 16 | float16 | int32 | 5.577e-08 | 9.756e-04 | 1.953e-03 | True/True | PASS |
| 20 | DeepSeekV4 | TP1 W2 | [512, 2048, 4096, 4096] | 16 | bfloat16 | int32 | 6.697e-08 | 7.752e-03 | 1.562e-02 | True/True | PASS |
| 21 | Boundary | all disabled indices | [9, 63, 65, 96] | 16 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 21 | Boundary | all disabled indices | [9, 63, 65, 96] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 22 | Boundary | minimum valid weight | [9, 63, 65, 96] | 16 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 22 | Boundary | minimum valid weight | [9, 63, 65, 96] | 16 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 23 | Boundary | maximum valid weight | [9, 63, 65, 96] | 16 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 23 | Boundary | maximum valid weight | [9, 63, 65, 96] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 24 | Boundary | zero scale | [9, 63, 65, 96] | 16 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 24 | Boundary | zero scale | [9, 63, 65, 96] | 16 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 25 | Boundary | negative scale | [9, 63, 65, 96] | 16 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 25 | Boundary | negative scale | [9, 63, 65, 96] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 26 | Boundary | last valid slice | [9, 63, 33, 97] | 16 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 26 | Boundary | last valid slice | [9, 63, 33, 97] | 16 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 27 | Boundary | zero rows | [0, 64, 64, 64] | 8 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 27 | Boundary | zero rows | [0, 64, 64, 64] | 8 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 28 | Generic | rank8 H8191 O=P-1 | [2, 8191, 1023, 1030] | 8 | float16 | int32 | 3.334e-07 | 6.868e-04 | 1.953e-03 | True/True | PASS |
| 28 | Generic | rank8 H8191 O=P-1 | [2, 8191, 1023, 1030] | 8 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 29 | Generic | rank8 H8192 O=P | [3, 8192, 1024, 1031] | 8 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 29 | Generic | rank8 H8192 O=P | [3, 8192, 1024, 1031] | 8 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 30 | Generic | rank8 H8193 O=P+1 | [5, 8193, 1025, 1032] | 8 | float16 | int32 | 5.244e-07 | 9.183e-04 | 4.883e-04 | True/True | PASS |
| 30 | Generic | rank8 H8193 O=P+1 | [5, 8193, 1025, 1032] | 8 | bfloat16 | int32 | 2.577e-06 | 6.803e-03 | 3.906e-03 | True/True | PASS |
| 31 | Generic | rank8 maximum H/O | [1, 16384, 16384, 16384] | 8 | float16 | int64 | 7.250e-07 | 9.470e-04 | 3.906e-03 | True/True | PASS |
| 31 | Generic | rank8 maximum H/O | [1, 16384, 16384, 16384] | 8 | bfloat16 | int64 | 2.441e-07 | 4.000e-03 | 1.562e-02 | True/True | PASS |
| 32 | Generic | rank16 H8191 O=P-1 | [2, 8191, 511, 518] | 16 | float16 | int32 | 8.243e-07 | 8.540e-04 | 3.906e-03 | True/True | PASS |
| 32 | Generic | rank16 H8191 O=P-1 | [2, 8191, 511, 518] | 16 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 33 | Generic | rank16 H8192 O=P | [3, 8192, 512, 519] | 16 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 33 | Generic | rank16 H8192 O=P | [3, 8192, 512, 519] | 16 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 34 | Generic | rank16 H8193 O=P+1 | [5, 8193, 513, 520] | 16 | float16 | int32 | 5.173e-06 | 2.820e-03 | 1.221e-04 | True/True | PASS |
| 34 | Generic | rank16 H8193 O=P+1 | [5, 8193, 513, 520] | 16 | bfloat16 | int32 | 7.852e-06 | 6.452e-03 | 4.883e-04 | True/True | PASS |
| 35 | Generic | rank16 maximum H/O | [1, 16384, 16384, 16384] | 16 | float16 | int64 | 1.743e-07 | 9.452e-04 | 7.812e-03 | True/True | PASS |
| 35 | Generic | rank16 maximum H/O | [1, 16384, 16384, 16384] | 16 | bfloat16 | int64 | 4.423e-07 | 7.246e-03 | 6.250e-02 | True/True | PASS |
| 36 | Generic | rank32 H8191 O=P-1 | [2, 8191, 255, 262] | 32 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 36 | Generic | rank32 H8191 O=P-1 | [2, 8191, 255, 262] | 32 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 37 | Generic | rank32 H8192 O=P | [3, 8192, 256, 263] | 32 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 37 | Generic | rank32 H8192 O=P | [3, 8192, 256, 263] | 32 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 38 | Generic | rank32 H8193 O=P+1 | [5, 8193, 257, 264] | 32 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 38 | Generic | rank32 H8193 O=P+1 | [5, 8193, 257, 264] | 32 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 39 | Generic | rank32 maximum H/O | [1, 16384, 16384, 16384] | 32 | float16 | int64 | 1.926e-07 | 5.568e-04 | 7.812e-03 | True/True | PASS |
| 39 | Generic | rank32 maximum H/O | [1, 16384, 16384, 16384] | 32 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 40 | Generic | rank64 H8191 O=P-1 | [2, 8191, 127, 134] | 64 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 40 | Generic | rank64 H8191 O=P-1 | [2, 8191, 127, 134] | 64 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 41 | Generic | rank64 H8192 O=P | [3, 8192, 128, 135] | 64 | float16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 41 | Generic | rank64 H8192 O=P | [3, 8192, 128, 135] | 64 | bfloat16 | int64 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 42 | Generic | rank64 H8193 O=P+1 | [5, 8193, 129, 136] | 64 | float16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 42 | Generic | rank64 H8193 O=P+1 | [5, 8193, 129, 136] | 64 | bfloat16 | int32 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True/True | PASS |
| 43 | Generic | rank64 maximum H/O | [1, 16384, 16384, 16384] | 64 | float16 | int64 | 1.632e-07 | 5.621e-04 | 1.562e-02 | True/True | PASS |
| 43 | Generic | rank64 maximum H/O | [1, 16384, 16384, 16384] | 64 | bfloat16 | int64 | 7.354e-07 | 4.049e-03 | 1.250e-01 | True/True | PASS |

## 按 dtype 汇总

| dtype | 用例数 | 通过数 | 失败数 |
| --- | ---: | ---: | ---: |
| bfloat16 | 43 | 43 | 0 |
| float16 | 43 | 43 | 0 |

## 关键发现

1. 共 86 个 FP16/BF16 与 int32/int64 组合，覆盖 rank 8/16/32/64、M=0、1-row fallback、4/8-row 复用和尾块。
2. expert-sorted 快速路径覆盖 38 例，mixed/alternating fallback 覆盖 14 例。
3. 非 32B 对齐 H/O、8191/8192/8193/16384 边界、P-1/P/P+1、slice offset、负/零 scale、全 -1 index 和最大合法 index均纳入验证。
4. 每个用例同时检查返回 alias 和 slice 外逐元素不变，避免只验证数值而漏掉 in-place 语义。
5. DeepSeek-V4-Flash 的 TP8 4096->256、256->4096 和完整 4096->2048、2048->4096 形状均已覆盖。
