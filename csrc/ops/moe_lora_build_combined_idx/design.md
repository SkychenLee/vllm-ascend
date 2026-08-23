# `moe_lora_build_combined_idx` 设计文档

## 1. 需求与接口

### 1.1 目标

在 AllGather、非 EP 的 MoE LoRA 路径中，一次生成 BGMV 使用的
`combined_idx`，替换当前的：

```text
abs -> argsort(AiCPU) -> gather expert -> div -> gather lora
    -> clamp -> gather adapter_enabled -> where -> contiguous
```

PyTorch 和 NumPy 不存在同名接口。本算子是 vLLM-Ascend 内部自定义算子，
语义以本设计为准。

### 1.2 Python schema

```text
moe_lora_build_combined_idx(
    Tensor expanded_row_idx,
    Tensor topk_ids,
    Tensor token_lora_indices,
    Tensor adapter_enabled,
    int num_experts
) -> Tensor
```

### 1.3 C++ 接口

```cpp
at::Tensor moe_lora_build_combined_idx(
    const at::Tensor& expanded_row_idx,
    const at::Tensor& topk_ids,
    const at::Tensor& token_lora_indices,
    const at::Tensor& adapter_enabled,
    int64_t num_experts);
```

### 1.4 参数和约束

| 参数 | dtype | shape | 约束 |
|---|---|---|---|
| `expanded_row_idx` | `int32` | `[N]` | `abs(expanded_row_idx)` 是 `[0, N)` 的排列 |
| `topk_ids` | `int32` | `[T, K]` | 连续，`N == T * K`，值域 `[0, num_experts)` |
| `token_lora_indices` | `int64` | `[C]` | `C >= T`，值为 `-1` 或有效 LoRA slot |
| `adapter_enabled` | `bool`/`int8` | `[L]` | 有效 LoRA slot 小于 `L`，非零表示启用 |
| `num_experts` | `int64` | scalar | `num_experts > 0` |
| 返回值 | `int32` | `[N]` | 连续 NPU Tensor；禁用行的值为 `-1` |

`expanded_row_idx` 的排列约束来自 `npu_moe_init_routing_v2`。算子不在
Device 热路径中检查数据值，因为检查会引入 NPU 到 Host 同步；Host 只检查
dtype、shape、device、contiguous 和标量范围。

## 2. 数学定义与等价性

令原始展开位置为 `p in [0, N)`，排序后位置为：

```text
s = abs(expanded_row_idx[p])
t = floor(p / K)
e = topk_ids.flatten()[p]
l = token_lora_indices[t]
```

输出为：

```text
combined_idx[s] = l * num_experts + e
```

当 `l < 0` 或 `adapter_enabled[l] == 0` 时：

```text
combined_idx[s] = -1
```

现实现先计算 `inv_perm = argsort(abs(expanded_row_idx))`，因此
`inv_perm[s] == p`。在排列前提下，按 `p` 直接 scatter 到 `s` 与当前实现
逐元素完全等价，并且不存在多核写冲突。

## 3. 实现路径

- 原型：TileLang Ascend，验证 scatter 语义、数据类型和分核策略。
- 正式实现：AscendC Vector Core direct kernel。
- 不使用 CATLASS：算子没有矩阵乘法。
- 不使用 ACLNN：没有可直接替代整个数据流的 CANN 内置算子。

## 4. AscendC 计算流程

### 4.1 Kernel 伪代码

```cpp
if (num_pairs < 512 || ub_capacity_insufficient) {
    // 单核快速路径；避免小 shape 的多核启动和重复扫描开销。
    for (uint32_t pair = 0; pair < num_pairs; ++pair) {
        uint32_t sorted_row = abs(expanded_row_idx_gm[pair]);
        combined_idx_gm[sorted_row] = build_index(pair);
    }
} else {
    // 每个 Core 独占一个 32B 对齐的连续输出区间。
    output_begin = core_id * output_pairs_per_core;
    output_end = min(output_begin + output_pairs_per_core, num_pairs);
    for (uint32_t begin = 0; begin < num_pairs; begin += tile_length) {
        // 所有 Core 连续搬入相同的 routing tile。
        copy_to_ub(expanded_ub, expanded_row_idx_gm + begin);
        copy_to_ub(expert_ub, topk_ids_gm + begin);
        for (uint32_t local_pair = 0; local_pair < tile_count; ++local_pair) {
            uint32_t sorted_row = abs(expanded_ub[local_pair]);
            if (output_begin <= sorted_row && sorted_row < output_end) {
                output_ub[sorted_row - output_begin] =
                    build_index(begin + local_pair, expert_ub[local_pair]);
            }
        }
    }
    copy_contiguous_to_gm(combined_idx_gm + output_begin, output_ub);
}
```

不能让多个 Core 对随机 GM 地址直接调用 `GlobalTensor::SetValue(int32)`。
910B3 上该接口可能以 32B 块为单位读改写，同一 cache line 内不同 Core 的写入会
互相覆盖。当前方案按 32B 对齐的输出区间分核，scatter 只发生在各 Core 私有 UB，
最后连续写回各自独占的 GM 区间，因此不存在跨核读改写冲突。

代价是每个 Core 都要扫描全部 routing 输入。它换取了正确且可并行的连续写回，
并在大 prefill shape 上取得 21%--26% 的 kernel 级收益。进一步优化需要改变
routing 上游布局或引入安全的分桶/前缀和，而不是恢复随机 GM scatter。

### 4.2 浮点精度

本算子只处理 `int32/int64/bool/int8`，没有 FP16/BF16 输入，也没有升精度或
舍入过程。输出要求与参考实现逐元素严格相等。

## 5. Host Tiling 与 UB 规划

### 5.1 运行时资源查询

Host 首次调用时通过 ACL 查询并缓存当前 device 的 Vector Core 数和每核 UB 容量。
910B3 实测 `ACL_DEV_ATTR_VECTOR_CORE_NUM=40`。当前 CANN 9.0 对
`ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE` 返回 0，因此使用保守的 64KB fallback；
kernel 的实际 UB 占用低于这个下限。后续调用不再执行 ACL 属性查询。

### 5.2 Block 级切分

`N < 512` 使用单核标量快速路径。`N >= 512` 时：

```text
desiredCoreNum = min(vectorCoreNum, ceil(N / 64))
outputPairsPerCore = align_up(ceil(N / desiredCoreNum), 8)
blockDim = ceil(N / outputPairsPerCore)
```

`8` 个 int32 正好是 32B，使相邻 Core 的输出区间保持块对齐；每核至少分配
约 64 个 pair，避免中等 shape 过度分核。最后一个 Core 只写有效尾部。

### 5.3 UB 级切分

多核路径的 `tileLength=min(N, 4096)`。每个 tile 连续搬入两个 int32 输入；
LoRA 和 adapter 因为是间接索引，仍使用标量 GM load。输出区间在 UB 内完成
scatter，一次连续 `DataCopyPad` 写回。

| Buffer | 大小 | 用途 |
|---|---:|---|
| `inputBuf` | `2 * align8(tileLength) * 4` | expanded/expert 连续 tile |
| `outputBuf` | `align8(outputLength) * 4` | 当前 Core 的私有输出区间 |
| Host safety reserve | 1024B | 编译器临时空间和余量 |

Host 在 launch 前验证：

```text
2 * align8(tileLength) * 4
    + align8(outputPairsPerCore) * 4
    + 1024 <= ubBytes
```

如果不满足则回退到安全单核路径。以 `4096 tokens * top_k 6` 为例，
`tileLength=4096`，输入 buffer 为 32KB，每核输出约 2.5KB，低于 64KB fallback。

## 6. Workspace

算子不需要 workspace。输出由 Host 直接分配为 `[N]` 的 `int32` NPU Tensor。
`combined_idx < max_loras * num_experts`，Host 检查该乘积不超过
`INT32_MAX`。BGMV 正式实现增加 int32 indices 分支，同时保留原有 int64 接口。

## 7. Host 与图捕获

Host wrapper 执行以下检查：

1. 四个 Tensor 均在同一 NPU device 且 contiguous。
2. `expanded_row_idx/topk_ids` 为 `int32`。
3. `token_lora_indices` 为 `int64`。
4. `adapter_enabled` 为 `bool` 或 `int8`。
5. `expanded_row_idx.numel() == topk_ids.numel()`。
6. `topk_ids.dim() == 2` 且 `token_lora_indices.numel() >= topk_ids.size(0)`。
7. `num_experts > 0` 且可放入 `uint32_t`。

输出 shape 只依赖输入 shape；kernel 不读取 `.item()`，不产生动态 shape，必须
支持 ACL Graph capture/replay。

## 8. 性能设计

### 8.1 基线

比较对象必须覆盖完整元数据路径，而不仅是单独 `argsort`：

```text
recover(abs/argsort/gather/div/gather)
+ W13 combined_idx(clamp/index/where/contiguous)
+ W2 combined_idx(clamp/index/where/contiguous)
```

融合路径只执行一次本算子，输出由 W13/W2 复用。

### 8.2 预期收益来源

- 消除 AiCPU `argsort` 和 AiCPU/AICore 调度切换。
- 多个小 Tensor op 合并为一个 Vector Core kernel。
- 不再物化 `inv_perm/expert_per_row/lora_per_row`。
- `combined_idx` 只生成一次，W13/W2 复用。

算子本身计算量约为每 pair 4 次 load、1 次 scatter、一次整数乘加和少量分支，
属于固定启动延迟和 GM 访存主导，不是计算受限算子。

### 8.3 性能用例

- `tokens`: 1, 2, 8, 32, 128, 512, 2048, 4096
- `top_k`: 1, 2, 6, 8
- `num_experts`: 8, 64, 160, 256
- `max_loras`: 1, 4, 16
- eager 使用 profiler 采集；ACL Graph 对 decode/prefill 做 capture/replay 精度验证
- 每 case 20 次，前 10 次预热；最终结论取 msprof `Task Duration(us)`

### 8.4 910B3 routing-only 实测结论

固定 profiler 使用 warmup=5、active=5，并将 `op_statistic.csv` 全部算子的
`Total Time(us)` 求和后除以 active 次数。每个 step 在 `prof.step()` 前同步 NPU，
避免异步任务跨 warmup/active 边界。完整 AiCPU 基线包含
`abs/argsort/gather/div/clamp/index/where/contiguous`，不是只测单个 argsort。

| Shape `[tokens, top_k]` | 最终 AscendC (us) | AiCPU 链 (us) | 加速比 |
|---|---:|---:|---:|
| `[1, 6]` | 2.156 | 80.414 | 37.298x |
| `[8, 6]` | 3.024 | 80.697 | 26.684x |
| `[32, 1]` | 2.792 | 74.805 | 26.791x |
| `[128, 6]` | 19.576 | 177.855 | 9.085x |
| `[512, 6]` | 70.673 | 459.385 | 6.500x |
| `[2048, 6]` | 264.669 | 1832.281 | 6.923x |
| `[4096, 6]` | 515.182 | 3735.499 | 7.251x |
| `[512, 8]` | 87.938 | 601.052 | 6.835x |

该表只衡量 routing 元数据链，不能直接当成 `add_lora_fused_moe` 或模型吞吐
收益。当前 kernel 瓶颈不是整数乘加算力，而是各 Core 重复扫描 routing、标量间接
LoRA/adapter load 和调度延迟，属于访存/指令延迟受限。

### 8.5 Qwen3.5-35B-A3B 真实 MoE 权重全路径实测

使用 layer-0 真实 router 与 expert BF16 权重，配置 hidden=2048、expert
intermediate=512、experts=256、top-k=8、rank=16。Router top-k 由真实 router
权重与 checkpoint 权重行构造的确定性 hidden 输入计算。由于 checkpoint 不含
LoRA adapter，A/B 取自真实 expert 权重子块；该口径用于 kernel 性能评估，不是
adapter 模型精度评估。

每条路径独立采集三轮并交换先后顺序，表中为中位数。自定义路径覆盖一次 fused
routing、W13/W2 复用 int32 combined_idx 和 int32 BGMV；标杆覆盖 AiCPU argsort、
W13/W2 各自生成 int64 combined_idx 和 int64 BGMV。

| Shape `[tokens, top_k]` | 融合全路径 (us) | 标杆全路径 (us) | 加速比 |
|---|---:|---:|---:|
| `[1, 8]` | 41.357 | 144.831 | 3.502x |
| `[2, 8]` | 44.389 | 148.135 | 3.337x |
| `[8, 8]` | 85.074 | 204.825 | 2.408x |
| `[32, 8]` | 173.292 | 317.099 | 1.830x |
| `[128, 8]` | 482.110 | 706.806 | 1.466x |
| `[256, 8]` | 906.422 | 1244.433 | 1.373x |
| `[512, 8]` | 1726.215 | 2294.598 | 1.329x |
| `[1024, 8]` | 3349.227 | 4468.261 | 1.334x |

在 `1024x8` 中，融合路径的 BGMV shrink+expand 为 3069.886us，占总时间
91.7%；fused routing 为 176.284us。标杆 Sort 为 1103.274us，占标杆总时间
24.7%。因此小 shape 收益主要来自消除 AiCPU argsort 和小算子调度，大 shape
则由 BGMV 计算与权重访问主导，收益收敛到约 1.33x。

### 8.6 Qwen3.5-27B Dense 真实权重 BGMV 对照

Qwen3.5-27B 是 Dense 模型，配置 hidden=5120、intermediate=17408，不含
expert/router，不能验证本 routing 算子。使用 layer-0 gate/up/down 的真实 BF16
权重子块和真实权重行输入，只比较 rank-16 W13+激活+W2 的 int32/int64 BGMV。
8 个 shape 的平均比值为 1.006x，单点范围 0.956x--1.062x；说明单纯切换
combined_idx dtype 几乎没有稳定收益，融合价值来自消除完整 routing 元数据链。

## 9. 实现检查清单

- [x] TileLang 原型与 PyTorch 参考实现逐元素相等。
- [x] TileLang 可通过 `target="ascendc"` 编译并导出 kernel source。
- [x] AscendC 使用运行时 Vector Core 数量，不硬编码核数。
- [x] 小 shape 和中大 shape 的分核策略均有实测。
- [x] 禁用 adapter、`lora=-1`、负 `expanded_row_idx` 均有覆盖。
- [x] 重复/越界的 `abs(expanded_row_idx)` 作为非法输入记录，不在热路径同步检查。
- [x] Meta 实现返回正确的静态 shape/dtype/device。
- [x] ACL Graph capture/replay 在 `1x6` 和 `4096x6` 上精度正确。
- [x] 42 个 routing 精度 case 和 8 个 BGMV index 回归 case 全部通过。
- [x] 8 个 shape/dtype case 使用 profiler 对比完整基线。
- [x] W13/W2 均复用同一个 `combined_idx`。
