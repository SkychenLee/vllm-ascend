# `moe_lora_bgmv_fused` 设计文档

## 1. 算子接口

### 1.1 目标

在 AllGather、非 EP、非 `fully_sharded` 的 MoE LoRA 路径中，把一对
`bgmv_shrink` 和 `bgmv_expand` 合并为一个 AscendC kernel：

```text
rank_out[row] = x[row] @ A[indices[row]].T * scale
y[row, slice] += rank_out[row] @ B[indices[row]].T
```

当前两个 kernel 会把 FP32 rank 中间结果写入 GM，再由 expand 读回；并且每个
token 都重新从 GM 搬入相同 expert/adapter 的 A/B 权重。新算子把 rank=16 中间
结果留在 UB，并在连续 4 行具有相同 index 时复用 A/B 权重搬运。

PyTorch 和 NumPy 没有同名接口。本算子是 vLLM-Ascend 内部 in-place 自定义算子，
语义以本设计为准。

### 1.2 Python schema

```text
moe_lora_bgmv_fused(
    Tensor x,
    Tensor lora_a,
    Tensor lora_b,
    Tensor indices,
    Tensor(a!) y,
    int slice_offset,
    int slice_size,
    float scale
) -> Tensor(a!)
```

### 1.3 C++ 签名

```cpp
at::Tensor moe_lora_bgmv_fused(
    const at::Tensor& x,
    const at::Tensor& loraA,
    const at::Tensor& loraB,
    const at::Tensor& indices,
    at::Tensor& y,
    int64_t sliceOffset,
    int64_t sliceSize,
    double scale);
```

### 1.4 参数和约束

| 参数 | dtype | shape | 约束 |
|---|---|---|---|
| `x` | FP16/BF16 | `[M, H]` | contiguous，`0 < H <= 2048` |
| `lora_a` | 与 `x` 相同 | `[L, 16, H]` | contiguous |
| `lora_b` | 与 `x` 相同 | `[L, O, 16]` | contiguous，`0 < O <= 2048` |
| `indices` | int32/int64 | `[M]` | contiguous，值为 `-1` 或 `[0, L)` |
| `y` | 与 `x` 相同 | `[M, Y]` | contiguous，in-place 更新 |
| `slice_offset` | int64 | scalar | `>= 0` |
| `slice_size` | int64 | scalar | `== O`，`slice_offset + O <= Y` |
| `scale` | double/float | scalar | Host 转为 FP32 左值后传入 kernel |

第一版固定 LoRA rank=16。`rank != 16`、`fully_sharded=True`、需要在 shrink 与
expand 之间乘 routed weight，或 shape 超出上述范围时，由 Python 调用层回退到
现有 `bgmv_shrink` + `bgmv_expand` 路径。

支持组合：

- FP16 + int32 indices
- FP16 + int64 indices
- BF16 + int32 indices
- BF16 + int64 indices

## 2. 数学定义与计算逻辑

对每一行 `m in [0, M)`，令 `i = indices[m]`。若 `i < 0`，`y[m]` 保持不变；
否则：

```text
r[m, k] = scale * sum_h(float(x[m, h]) * float(lora_a[i, k, h]))
d[m, o] = sum_k(r[m, k] * float(lora_b[i, o, k]))
y[m, slice_offset + o] = cast_dtype(float(y[m, slice_offset + o]) + d[m, o])
```

其中 `k in [0, 16)`、`o in [0, O)`。FP16/BF16 输入全部 Cast 到 FP32 计算，
最终按 `CAST_RINT` 转回输出 dtype。

### 2.1 4-row 分组快速路径

AllGather MoE 的 routed rows 已按 expert 排序；相同 adapter 下，同一 expert 的
`combined_idx` 连续。Kernel 每次观察连续 4 行：

```cpp
index0 = indices[row + 0];
index1 = indices[row + 1];
index2 = indices[row + 2];
index3 = indices[row + 3];
if (index0 == index1 && index0 == index2 && index0 == index3) {
    if (index0 >= 0) {
        ProcessGroup<4>(row, index0);  // A/B 每个 tile 只从 GM 搬一次
    }
    row += 4;
} else {
    ProcessGroup<1>(row, index0);      // 任意 index 排列的正确性回退
    row += 1;
}
```

该判断只影响性能，不改变结果；不要求 `indices` 全局有序。`-1` 的 4-row group
直接跳过，mixed index 逐行处理。

### 2.2 AscendC API 序列

#### Shrink 阶段

```cpp
DataCopyPad(xLocal, xGm[row:row+R, :]);
Cast(xFp32, xLocal, CAST_NONE, R * H);
for (rank = 0; rank < 16; ++rank) {
    DataCopyPad(weightLocal, aGm[index, rank, :]);
    for (r = 0; r < R; ++r) {
        Cast(weightFp32, weightLocal, CAST_NONE, H);
        Mul(weightFp32, xFp32[r * H], weightFp32, H);
        ReduceSum(weightFp32, weightFp32, weightFp32, H);
        rankLocal[r, rank] = weightFp32.GetValue(0) * scale;
    }
}
```

`R` 为 4 或 1。X 每行只从 GM 搬一次；4-row 快速路径中每个 A rank row 只从
GM 搬一次。`ReduceSum` 后不复用其源数据，下一行先重新 Cast A 到 `weightFp32`。

#### Expand 阶段

输出按 512 个元素切 tile。每个 tile 对应 `512 * 16 = 8192` 个连续 B 元素：

```cpp
for (output_begin = 0; output_begin < O; output_begin += 512) {
    DataCopyPad(weightLocal, bGm[index, output_begin, 0]);
    for (r = 0; r < R; ++r) {
        Copy(rankDup, rankLocal[r], rank=16, repeat=4);
        Cast(weightFp32, weightLocal, CAST_NONE, tile_outputs * 16);
        Mul(weightFp32, rankDup, weightFp32,
            mask=64, repeat=ceil(tile_outputs * 16 / 64),
            src0RepStride=0);
        BlockReduceSum(...);
        PairReduceSum(...);  // 每 16 个乘积归约为一个输出
        DataCopyPad(yInputLocal, yGm[row, slice]);
        Cast(yInputFp32, yInputLocal, CAST_NONE, tile_outputs);
        Add(yAccumFp32, yAccumFp32, yInputFp32, tile_outputs);
        Cast(yOutputLocal, yAccumFp32, CAST_RINT, tile_outputs);
        DataCopyPad(yGm[row, slice], yOutputLocal);
    }
}
```

B 的 BF16/FP16 local tensor在 4 行之间保留；每行重新 Cast 后原地 Mul，避免额外
32 KiB FP32 B 副本。这样保留现有 expand 的归约顺序，同时将 B 的 GM 流量最多
降低到原来的 1/4。

### 2.3 实现路径选择

- [x] AscendC Kernel（Vector + reduction）
- [ ] CATLASS 模板库
- [ ] ACLNN 封装

虽然数学上包含两个小矩阵乘，但每行动态选择不同权重，rank 固定为 16，且需要
in-place slice add。直接调用常规 GEMM 仍需先做动态分组、workspace permutation
和 scatter。第一版采用 AscendC Vector kernel，在不增加动态 shape/workspace 的
前提下利用现有 expert-sorted 局部性。若 4-row 复用仍达不到目标，再评估带显式
group offsets 的 CATLASS/Grouped GEMM v3。

## 3. Tiling 策略

### 3.1 TilingData

```cpp
struct MoeLoraBgmvFusedTilingData {
    uint32_t numRows;
    uint32_t inputHiddenDim;
    uint32_t outputHiddenDim;
    uint32_t outputFullDim;
    uint32_t sliceOffset;
    uint32_t rowsPerCore;
    uint32_t blockDim;
    uint32_t groupRows;          // 4
    uint32_t outputTileRows;     // 512
    uint32_t rank;               // 16
    float scale;
};
```

dtype 和 index dtype 通过不同模板 kernel 入口选择，不在 Device 热路径分支。

### 3.2 Block 级切分

Host 通过 `ACL_DEV_ATTR_VECTOR_CORE_NUM` 获取 AIV 数量，不硬编码核数：

```text
desiredCoreNum = min(vectorCoreNum, ceil(M / 4))
rowsPerCore = align_up(ceil(M / desiredCoreNum), 4)
blockDim = ceil(M / rowsPerCore)
```

每个 Core 处理连续且以 4-row 对齐的区间。这样只会在 Core 边界拆开少量
combined-index group，同时保证不同 Core 写入不同的 y 行，不需要 atomic 或核间同步。

Python 集成层仅在 `M >= 512` 时选择新算子；小 shape 继续走现有 kernel，避免
4-row 分组判断和更大 UB 初始化增加 decode 延迟。该阈值是命名常量，最终根据
1/2/8/32/128/256/512/1024-token msprof 结果调整。

### 3.3 UB 级切分

第一版约束 `H <= 2048`、`O <= 2048`。Shrink 一次保存最多 4 行 X；Expand 将
输出按 512 元素切分，B tile 固定最多 8192 元素。

```text
H_aligned = align_up(H, 16)
outputTile = min(512, remaining_output)
weightTileElements = 512 * 16 = 8192
```

#### UB 分配表

`H_a = align_up(H, 16)`、`O_t = 512`、`W_t = max(H_a, 8192)`：

| Buffer | 单位大小 | 数量 | 最大字节 | 阶段复用 |
|---|---:|---:|---:|---|
| `xQueue` | `4 * H_a * 2` | 1 | 16,384 | 搬入 4 行 FP16/BF16 X |
| `weightQueue` | `W_t * 2` | 1 | 16,384 | A rank row / B output tile |
| `xFp32Buffer` | `4 * H_a * 4` | 1 | 32,768 | shrink X；expand rank duplicate |
| `weightFp32Buffer` | `W_t * 4` | 1 | 32,768 | A/B Cast、Mul、Reduce |
| `rankBuffer` | `4 * 16 * 4` | 1 | 256 | FP32 shrink 中间结果 |
| `yInputQueue` | `O_t * 2` | 1 | 1,024 | 原 y slice |
| `yOutputQueue` | `O_t * 2` | 1 | 1,024 | 更新后的 y slice |
| `yInputFp32` | `O_t * 4` | 1 | 2,048 | y 升精度 |
| `yAccumFp32` | `O_t * 4` | 1 | 2,048 | expand 归约结果 + y |
| **总计** |  |  | **104,704** | 小于 910B 每核 UB |

FP16 和 BF16 的元素大小相同，因此 UB 公式一致：

```text
ubBytes = 24 * H_a + 6 * max(H_a, 8192) + 12 * O_t + 256
bufferCoefficient(FP16) = 24 bytes/input-column + 12 bytes/output-element
bufferCoefficient(BF16) = 24 bytes/input-column + 12 bytes/output-element
fixed/reusable weight region = 6 * max(H_a, 8192) + 256 bytes
```

最大 shape `H=2048, O_t=512` 使用 104,704 bytes。Host 查询
`ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`；当 CANN 9.0 在 910B3 返回 0 时，不把 0
当成真实容量，而依赖上述已封顶的 shape/UB 证明。若平台返回非零 UB，则必须验证
`ubBytes <= queriedUbBytes`，否则拒绝调用并由 Python 回退旧路径。

所有 GM 到 UB、UB 到 GM 搬运使用 `DataCopyPad`，尾部不足 32B 时由 pad 参数处理。

## 4. Workspace 与图捕获

算子不需要 workspace。输出 y 由调用方提供并原地更新；rank 中间结果只存在于 UB。

所有 shape、blockDim 和 buffer 大小只依赖 Tensor shape；Device 读取 indices 值只
决定走 4-row 或 1-row 分支，不产生动态输出 shape，也不执行 Host `.item()`，因此
支持 ACL Graph capture/replay。

## 5. 性能规划

### 5.1 基线

性能比较必须包括相同 fused routing 前缀：

```text
baseline: moe_lora_build_combined_idx
          + bgmv_shrink + rank GM tensor + bgmv_expand

candidate: moe_lora_build_combined_idx
           + moe_lora_bgmv_fused
```

W13 和 W2 分别比较，并继续运行完整 W13 -> SiLU/Mul -> W2 真实权重路径。

### 5.2 收益来源

1. 取消 `[M, 16]` FP32 rank tensor 的 GM 写回、读回和一次 kernel launch。
2. 连续 4 行 index 相同时，A/B 权重 GM 搬运最多降到原来的 1/4。
3. X 在 shrink 的 16 个 rank 归约中常驻 FP32 UB，不重复从 GM 搬运。
4. rank=16 编译期特化，移除运行时 rank 分支。
5. mixed index 自动使用 1-row 路径，不额外构造 permutation/workspace。

### 5.3 预期与门禁

- 1024-token BGMV shrink+expand 基线：3121.991 us，占全路径 91.7%。
- 第一阶段目标：BGMV 降低 10%--20%，即节省约 312--624 us。
- 对应完整路径目标：3406.804 us -> 3095--2783 us，端到端提升约 9%--18%。
- 128/256/512/1024-token 中至少 3 个 case 提升，且 1024-token 提升不少于 8%，
  才默认启用新路径。
- 1/2/8/32-token 必须保持旧路径，端到端不得回退。

预期是访存与 Vector 计算混合受限。4-row 路径主要减少 A/B GM 流量；Cast、Mul、
Reduce 指令数量基本不变，因此 4 倍权重流量下降不等于 4 倍 kernel 加速。

## 6. Kernel 实现要点

1. `TPipe` 在 kernel 入口创建并以指针传入类，避免 Scalar 常量折叠受阻。
2. `DataCopyPad` 的长度和 repeat 参数均由 Host 已验证的 uint32 shape 派生。
3. FP16/BF16 在 `Mul`、`ReduceSum` 和 `Add` 前 Cast 到 FP32。
4. `ReduceSum` 后重新 Cast weight local，再处理下一行，禁止直接复用被归约的源 tensor。
5. Expand 复用现有 rank=16 的 `BlockReduceSum + PairReduceSum` 指令组合。
6. 高维 `Copy` 的 `repeatTime <= 4`，不会超过 255。
7. 不向 kernel 传右值；Host 将 `scale` 转为 FP32 局部变量后捕获并传入。
8. Core 范围按 4 行对齐，不同 Core 不写同一个 y cache line。

## 7. Python 集成策略

`PunicaWrapperNPU.add_lora_fused_moe` 在每个 A/B slice 上判断：

```text
use_fused = (
    not fully_sharded
    and not mul_routed_weight
    and local_rank == 16
    and full_rank == 16
    and x2d.shape[0] >= FUSED_BGMV_MIN_ROWS
    and x2d.shape[1] <= 2048
    and out_size <= 2048
)
```

满足时直接调用 `moe_lora_bgmv_fused` 更新 y slice；否则保留现有 shrink、TP 通信、
routed-weight multiply 和 expand 路径。该策略不改变 `fully_sharded` 语义。

## 8. 验证计划

### 8.1 精度

- 至少 30 个 shape/dtype/index 组合。
- FP16/BF16，int32/int64 index。
- 全有效、全 `-1`、mixed index、连续 4 行相同 index、完全不分组。
- 非 32B 对齐的 H/O 尾部。
- slice offset、in-place add、scale 非 1。
- 与 `bgmv_shrink + bgmv_expand` 严格 parity，并与 PyTorch FP32 参考做 allclose。
- ACL Graph capture/replay 覆盖 512-row 和 8192-row。

### 8.2 性能

- 真实 Qwen3.5-35B-A3B router/expert 权重。
- tokens：1、2、8、32、128、256、512、1024，top-k=8，rank=16。
- 记录 fused kernel、旧 shrink/expand 和完整 W13/W2 路径时间。
- 每条路径独立 3 轮，奇偶轮交换顺序，以 msprof active step 中位数比较。

## 9. 实现检查清单

- [x] 工程骨架已创建。
- [x] 接口、数学语义和回退条件已定义。
- [x] Block/UB 两级 tiling、UB 分配表和 coefficient 已推导。
- [x] FP16/BF16 FP32 计算路径已定义。
- [x] 无 workspace、静态 shape、ACL Graph 约束已定义。
- [ ] 统一测试用例文档已生成。
- [ ] Host、Kernel、注册和 Python wrapper 已实现。
- [ ] 编译与基本功能测试通过。
- [ ] 中文接口 README 已生成。
- [ ] 至少 30 项精度测试通过。
- [ ] 真实权重 msprof 门禁通过。

## 10. 参考实现

- `csrc/kernels/bgmv_shrink.cpp`
- `csrc/kernels/bgmv_expand.cpp`
- `csrc/ops/moe_lora_build_combined_idx/design.md`
- `vllm_ascend/lora/punica_npu.py::add_lora_fused_moe`
- PyTorch 标杆：按行选择 A/B 后执行 FP32 `matmul`，再 Cast/加回 y。
