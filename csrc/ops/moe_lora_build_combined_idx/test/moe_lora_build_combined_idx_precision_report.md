# `moe_lora_build_combined_idx` 精度验证报告

- 测试平台：Ascend 910B3
- 测试时间：2026-08-22T11:49:59+00:00
- Routing 参考：PyTorch CPU direct scatter，输出要求逐元素完全相等。
- BGMV 参考：原 int64 index kernel 严格一致，并与 PyTorch CPU 参考做 allclose。

## 总览

| 指标 | 值 |
| --- | ---: |
| 总用例数 | 52 |
| 通过数 | 52 |
| 失败数 | 0 |
| 通过率 | 100.00% |
| Routing 严格相等 | 42/42 |
| BGMV index 兼容 | 8/8 |
| ACL Graph capture/replay | 2/2 |

## 精度标准

Routing 是整数索引构造，采用比浮点阈值更严格的逐元素完全相等；
因此通过用例的 MERE、MARE、MaxAbsErr 均必须为 0。相对误差仍按
`abs(actual - golden) / (abs(golden) + 1e-7)` 记录。
BGMV 的 int32 与 int64 index 输出要求逐元素一致，同时 FP16 使用
`rtol=atol=2e-2`，BF16 expand 使用 `rtol=atol=1.5e-1` 对比 CPU。

## Routing 测试结果

| # | 类别 | 描述 | Shape | adapter dtype | Mismatch | MERE | MARE | 结果 |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Decode | 1 token, DeepSeek top-k | [1, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 1 | Decode | 1 token, DeepSeek top-k | [1, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 2 | Decode | 2 tokens | [2, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 2 | Decode | 2 tokens | [2, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 3 | Decode | 8 tokens | [8, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 3 | Decode | 8 tokens | [8, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 4 | Decode | 32 tokens | [32, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 4 | Decode | 32 tokens | [32, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 5 | Prefill | 128 tokens | [128, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 5 | Prefill | 128 tokens | [128, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 6 | Prefill | 512 tokens | [512, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 6 | Prefill | 512 tokens | [512, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 7 | TopK | top-k 1 | [32, 1] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 7 | TopK | top-k 1 | [32, 1] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 8 | TopK | top-k 2 | [32, 2] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 8 | TopK | top-k 2 | [32, 2] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 9 | TopK | top-k 8 | [32, 8] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 9 | TopK | top-k 8 | [32, 8] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 10 | Experts | small expert table | [16, 2] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 10 | Experts | small expert table | [16, 2] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 11 | Small | single pair | [1, 1] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 11 | Small | single pair | [1, 1] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 12 | Small | non-aligned 5 pairs | [5, 1] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 12 | Small | non-aligned 5 pairs | [5, 1] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 13 | Small | non-aligned 18 pairs | [3, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 13 | Small | non-aligned 18 pairs | [3, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 14 | Large | 2048-token prefill | [2048, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 14 | Large | 2048-token prefill | [2048, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 15 | Large | 4096-token prefill | [4096, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 15 | Large | 4096-token prefill | [4096, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 16 | Capacity | mapping capacity exceeds T | [17, 2] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 16 | Capacity | mapping capacity exceeds T | [17, 2] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 17 | Boundary | all token LoRA indices are -1 | [32, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 17 | Boundary | all token LoRA indices are -1 | [32, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 18 | Boundary | all adapters disabled | [32, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 18 | Boundary | all adapters disabled | [32, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 19 | Boundary | all adapters enabled | [32, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 19 | Boundary | all adapters enabled | [32, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 20 | Boundary | maximum valid LoRA/expert IDs | [32, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 20 | Boundary | maximum valid LoRA/expert IDs | [32, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 21 | Boundary | all nonzero routing positions negative | [32, 6] | bool | 0 | 0.000e+00 | 0.000e+00 | PASS |
| 21 | Boundary | all nonzero routing positions negative | [32, 6] | int8 | 0 | 0.000e+00 | 0.000e+00 | PASS |

## BGMV int32 index 回归结果

| # | 算子 | 数据 dtype | index dtype | Shape | int64 parity MaxAbs | CPU MaxAbsErr | 容差 | 结果 |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | shrink | float16 | int32 | [7, 8] | 0.000e+00 | 1.907e-06 | 2.00e-02 | PASS |
| 2 | shrink | float16 | int64 | [7, 8] | 0.000e+00 | 1.907e-06 | 2.00e-02 | PASS |
| 3 | shrink | bfloat16 | int32 | [7, 8] | 0.000e+00 | 9.537e-07 | 2.00e-02 | PASS |
| 4 | shrink | bfloat16 | int64 | [7, 8] | 0.000e+00 | 9.537e-07 | 2.00e-02 | PASS |
| 5 | expand | float16 | int32 | [7, 128] | 0.000e+00 | 0.000e+00 | 2.00e-02 | PASS |
| 6 | expand | float16 | int64 | [7, 128] | 0.000e+00 | 0.000e+00 | 2.00e-02 | PASS |
| 7 | expand | bfloat16 | int32 | [7, 128] | 0.000e+00 | 0.000e+00 | 1.50e-01 | PASS |
| 8 | expand | bfloat16 | int64 | [7, 128] | 0.000e+00 | 0.000e+00 | 1.50e-01 | PASS |

## ACL Graph capture/replay

| Case | Shape | Mismatch | MaxAbsErr | 结果 |
| ---: | --- | ---: | ---: | --- |
| 1 | [1, 6] | 0 | 0.000e+00 | PASS |
| 15 | [4096, 6] | 0 | 0.000e+00 | PASS |

## 按 dtype 汇总

| dtype | 用例数 | 通过数 | 失败数 |
| --- | ---: | ---: | ---: |
| bfloat16 | 4 | 4 | 0 |
| bool | 23 | 23 | 0 |
| float16 | 4 | 4 | 0 |
| int8 | 21 | 21 | 0 |

## 关键发现

1. Routing 的 42 个用例全部逐元素相等，最大绝对误差和 mismatch 均为 0。
2. bool/int8 两种 adapter_enabled 存储类型结果一致；非 32B 对齐、小 decode、4096-token prefill 均通过。
3. `lora=-1`、adapter 全禁用/全启用、最大合法 ID、负 routing 位置和 capacity 大于 token 数均已覆盖。
4. BGMV 的 8 个 FP16/BF16、shrink/expand、int32/int64 index 回归用例全部通过；int32 与原 int64 kernel 输出严格一致。
5. ACL Graph capture/replay 的 2 个用例全部通过，`1x6` decode 与 `4096x6` prefill 均逐元素严格相等。
