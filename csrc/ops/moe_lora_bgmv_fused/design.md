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
token 都重新从 GM 搬入相同 expert/adapter 的 A/B 权重。融合实现把 rank 中间
结果留在 UB，并采用三级路由：

1. `rank == 16 && H <= 4096 && O <= 4096` 使用已实测的 fast 模板；连续
   8/4/2 行 index 相同时复用 A/B 权重，mixed index 使用 1-row 分支。
2. `rank in {8, 16, 32, 64}`、`1 <= H/O <= 16384` 使用 group1 Generic 单
   kernel；它不依赖 index 排序，覆盖 fast 之外的合法融合 shape。
3. 不支持的 rank/dtype/shape，以及需要通信或中间 routed-weight multiply 的
   模式，保留现有 `bgmv_shrink + bgmv_expand_slice` split 路径。

Fast 中 `H <= 2048` 使用 group8 模板；`2048 < H <= 4096` 使用 group4 模板，
把 DeepSeek-V4-Flash 的宽输入控制在 910B 单核 UB 上限内。Generic 是覆盖兜底，
不替代 fast，也不改变 split 路径承担的通信语义。

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
| `x` | FP16/BF16 | `[M, H]` | contiguous，`M >= 0`，`0 < H <= 16384` |
| `lora_a` | 与 `x` 相同 | `[L, R, H]` | contiguous，`R in {8, 16, 32, 64}` |
| `lora_b` | 与 `x` 相同 | `[L, O, R]` | contiguous，`0 < O <= 16384` |
| `indices` | int32/int64 | `[M]` | contiguous，值为任意负数或 `[0, L)` |
| `y` | 与 `x` 相同 | `[M, Y]` | contiguous，in-place 更新 |
| `slice_offset` | int64 | scalar | `>= 0` |
| `slice_size` | int64 | scalar | `== O`，`slice_offset + O <= Y` |
| `scale` | double/float | scalar | Host 转为 FP32 左值后传入 kernel |

Fast 固定 `R=16` 且 `H/O <= 4096`；Generic 对 `R` 做 8/16/32/64 四个编译期
实例化，并支持 `H/O <= 16384`。以下条件由 Python 调用层直接回退到现有
`bgmv_shrink + 通信/权重乘 + bgmv_expand_slice` 路径：

- `fully_sharded=True`；
- `mul_routed_weight=True`；
- `local_rank != full_rank`；
- rank、dtype、index dtype 或 shape 不在上述集合内；
- fused op 不可用。

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

其中 `k in [0, R)`、`o in [0, O)`。FP16/BF16 输入全部 Cast 到 FP32 计算，
最终按 `CAST_RINT` 转回输出 dtype。

### 2.1 Fast：8/4/2/1-row 分组路径

AllGather MoE 的 routed rows 已按 expert 排序；相同 adapter 下，同一 expert 的
`combined_idx` 连续。Kernel 每次最多观察连续 8 行，并依次尝试 8/4/2/1-row：

```cpp
if (H % 8 == 0 && indices[row:row+8] 全部相同) {
    if (index0 >= 0) {
        ProcessGroup<8>(row, index0);
    }
    row += 8;
} else if (H % 8 == 0 && indices[row:row+4] 全部相同) {
    if (index0 >= 0) {
        ProcessGroup<4>(row, index0);
    }
    row += 4;
} else if (H % 8 == 0 && indices[row:row+2] 全部相同) {
    if (index0 >= 0) {
        ProcessGroup<2>(row, index0);
    }
    row += 2;
} else {
    ProcessGroup<1>(row, index0);
    row += 1;
}
```

该判断只影响性能，不改变结果；不要求 `indices` 全局有序。FP32 X 每行起始地址
必须 32B 对齐，因此 `H` 不是 8 的倍数时也逐行处理。负 index 的整组直接跳过，
mixed index 逐行处理。Generic 固定 group1，不执行相邻 index 探测。

### 2.2 Fast AscendC API 序列

#### Shrink 阶段

```cpp
DataCopyPad(xLocal, xGm[row:row+R, :]);
Cast(xFp32, xLocal, CAST_NONE, R * H);
for (rank = 0; rank < 16; ++rank) {
    DataCopyPad(weightLocal, aGm[index, rank, :]);
    Cast(weightFp32, weightLocal, CAST_NONE, H);
    for (r = 0; r < R; ++r) {
        Mul(productFp32, xFp32[r * H], weightFp32, H);
        ReduceSum(rankLocal[r, rank], productFp32, reduceTmp, H);
    }
}
Muls(rankLocal, rankLocal, scale, R * 16);
```

`R` 为 8、4 或 1。X 每行只从 GM 搬一次；分组快速路径中每个 A rank row 每组
只从 GM 搬一次并 Cast 一次。独立的 `productFp32` 接收 Mul 和 Reduce 结果，保留
`weightFp32` 供组内后续行复用。仅 `M < 2048 && H == 2048` 使用不分配 product
buffer 的 inplace variant，每行重新 Cast 权重，以降低 shrink-heavy 小规模开销。

#### Expand 阶段

输出按 512 个元素切 tile。每个 tile 对应 `512 * 16 = 8192` 个连续 B 元素：

```cpp
for (output_begin = 0; output_begin < O; output_begin += 512) {
    DataCopyPad(weightLocal, bGm[index, output_begin, 0]);
    Cast(weightFp32, weightLocal, CAST_NONE, tile_outputs * 16);
    for (r = 0; r < R; ++r) {
        Copy(rankDup, rankLocal[r], rank=16, repeat=4);
        Mul(productFp32, rankDup, weightFp32,
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

B 的 BF16/FP16 local tensor 和 FP32 Cast 结果在 8/4/2 行之间保留；每行只把
Mul/Reduce 结果写入独立的 FP32 product buffer。这样在额外使用 32 KiB UB 的
前提下保留现有 expand 归约顺序，并将 B 的 GM 流量和 Cast 指令最多降到原来的
1/8。inplace variant 保留 8/4-row GM 权重复用，但每行重新 Cast。

### 2.3 Generic 单 kernel API 序列

Generic 使用 `K_TILE = 8192`、`B_TILE = 8192` 和
`O_TILE = 8192 / R`，因此 `H/O <= 16384` 时 Shrink 最多两个 K tile，Expand
根据 rank 每次处理 1024/512/256/128 个输出。每行独立读取 index；任意负 index
直接跳过，y 保持不变。

Shrink 对当前 `H_tile` 只搬入并 Cast 一次 X；A 按 rank 批量 DMA，批量大小为：

```text
rankBatch = min(R, max(1, 8192 / H_tile))
```

```cpp
for (h_begin = 0; h_begin < H; h_begin += 8192) {
    H_tile = min(8192, H - h_begin);
    DataCopyPad(xLocal, xGm[row, h_begin:h_begin+H_tile]);
    Cast(xFp32, xLocal, CAST_NONE, H_tile);
    for (rank_begin = 0; rank_begin < R; rank_begin += rankBatch) {
        DataCopyPad(weightLocal, aGm[index, rank_begin:rank_begin+rankBatch,
                                     h_begin:h_begin+H_tile]);
        for (k = 0; k < currentRankBatch; ++k) {
            Cast(weightProductFp32, weightLocal[k], CAST_NONE, H_tile);
            Mul(weightProductFp32, xFp32, weightProductFp32, H_tile);
            ReduceSum(rankPartial[h_tile, rank_begin+k],
                      weightProductFp32, reduceTmp, H_tile);
        }
    }
}
Add(rankPartial[0], rankPartial[0], rankPartial[1], R);  // 仅两片时
Muls(rankLocal, rankPartial[0], scale, R);
```

Expand 中 B 每片不超过 8192 个元素。rank 向量复制为 64-float pattern 后，
`Mul(..., src0RepStride=0)` 复用；归约树按编译期 rank 特化：

| rank | `O_TILE` | Expand 归约 |
|---:|---:|---|
| 8 | 1024 | `BlockReduceSum` |
| 16 | 512 | `BlockReduceSum + PairReduceSum` |
| 32 | 256 | `BlockReduceSum + 2 x PairReduceSum` |
| 64 | 128 | `2 x BlockReduceSum` |

尾 B tile 先把 product 补到 64-float repeat 边界；每一级 PairReduce 前也显式
补零到该级输入边界，不能依赖 `DataCopyPad(isPad=false)` 清理未写 UB。每行进入
Expand 时一次搬入完整 y slice，各 B tile 在 UB 内按偏移 Cast 到 FP32、Add、
`CAST_RINT`，循环结束后一次搬回完整 slice，避免 rank32/64 的碎片化 y DMA。

第一版所有 queue 都是真实单缓冲：`weightQueue` 的 slot 数为 1，完整 y slice
使用单个 `TQueBind<VECIN, VECOUT, 1>` 原地复用。只有实现显式
prologue/steady-state/epilogue 预取后，才对双缓冲版本单独做 profiler A/B；不能
只把 queue 数改为 2 并宣称存在搬算重叠。

### 2.4 实现路径选择

- [x] AscendC Kernel（Vector + reduction）
- [ ] CATLASS 模板库
- [ ] ACLNN 封装

虽然数学上包含两个小矩阵乘，但每行动态选择不同权重，rank 最大仅 64，且需要
in-place slice add。直接调用常规 GEMM 仍需先做动态分组、workspace permutation
和 scatter。Fast 与 Generic 均采用 AscendC Vector kernel：Fast 利用
expert-sorted 局部性，Generic 用 group1 保证通用覆盖。若大 M 分组场景仍达不到
目标，再评估带显式 group offsets 的 CATLASS/Grouped GEMM v3。

## 3. Tiling 策略

### 3.1 TilingData

```cpp
struct MoeLoraBgmvFusedTilingData {
    uint32_t numRows;
    uint32_t inputHiddenDim;
    uint32_t outputHiddenDim;
    uint32_t outputFullDim;
    uint32_t sliceOffset;
    uint32_t coreNum;
    uint32_t rowsPerCore;
    uint32_t blockDim;
    uint32_t groupRows;          // fast 为 8/4，Generic 为 1
    uint32_t outputTileRows;     // fast 为 512，Generic 为 8192/rank
    uint32_t rank;               // 8/16/32/64
    float scale;
};
```

dtype、index dtype 和 Generic rank 通过不同模板 kernel 入口选择，不在 Device
热路径分支。

### 3.2 Block 级切分

Host 通过 `ACL_DEV_ATTR_VECTOR_CORE_NUM` 获取 AIV 数量，不硬编码核数。`M == 0`
直接返回 y，不启动 kernel。`0 < M <= 1024` 时优先保证小/中 M 的 AIV 利用率：

```text
coreNum = min(M, vectorCoreNum)
q = M / coreNum
r = M % coreNum
rowCount(blockIdx) = q + (blockIdx < r ? 1 : 0)
rowBegin(blockIdx) = blockIdx * q + min(blockIdx, r)
blockDim = coreNum
```

该 quotient/remainder 映射保证任意两核最多相差 1 行。`M > 1024` 的 fast 路径
继续使用 group 对齐策略，避免为负载均衡切断过多 expert run：

```text
groupRows = H <= 2048 ? 8 : 4
desiredCoreNum = min(vectorCoreNum, ceil(M / groupRows))
rowsPerCore = align_up(ceil(M / desiredCoreNum), groupRows)
blockDim = ceil(M / rowsPerCore)
```

`M > 1024` 的 Generic 固定 `groupRows=1`，按连续 `rowsPerCore` 区间切分。所有
策略都保证不同 Core 写入不同 y 行，不需要 atomic 或核间同步。

AscendC launcher 在融合算子内部进一步选择编译期特化入口：

```text
rank == 16 且 H/O <= 4096  -> fast
  H <= 2048               -> group8 template
  2048 < H <= 4096        -> group4 template
  M < 2048 且 H == 2048   -> inplace variant，不分配 productFp32Buffer
  其它 fast shape          -> reuse variant，A/B 每组只 Cast 一次
其它受支持 rank/H/O         -> Generic group1 template
其它                         -> 不调用 fused op，由 Python 走 split
```

### 3.3 Fast UB 级切分

当前约束 `H <= 4096`、`O <= 4096`。Shrink 在窄输入模板一次保存最多 8 行 X，
宽输入模板一次保存最多 4 行 X；Expand 始终按 512 元素切分，B tile 固定最多
8192 元素，因此增大 O 不增加 UB 峰值。

```text
H_aligned = align_up(H, 16)
outputTile = min(512, remaining_output)
weightTileElements = 512 * 16 = 8192
```

#### UB 分配表

`G = H <= 2048 ? 8 : 4`、`H_a = align_up(H, 16)`、`O_t = 512`、
`W_t = max(H_a, 8192)`：

| Buffer | 单位大小 | 数量 | 最大字节 | 阶段复用 |
|---|---:|---:|---:|---|
| `indicesQueue` | `G * sizeof(index_t)` | 1 | 64 | 搬入 int32/int64 index |
| `xQueue` | `G * H_a * 2` | 1 | 32,768 | 搬入 G 行 FP16/BF16 X |
| `weightQueue` | `W_t * 2` | 1 | 16,384 | A rank row / B output tile |
| `xFp32Buffer` | `G * H_a * 4` | 1 | 65,536 | shrink X；expand rank duplicate |
| `weightFp32Buffer` | `W_t * 4` | 1 | 32,768 | 组内复用 A/B FP32 权重 |
| `productFp32Buffer` | `W_t * 4` | 1 | 32,768 | 仅 reuse variant：Mul 与 Reduce scratch |
| `rankBuffer` | `G * 16 * 4` | 1 | 512 | FP32 shrink 中间结果 |
| `reduceTmpBuffer` | 256 | 1 | 256 | FP32 `ReduceSum` 临时区 |
| `yInputQueue` | `O_t * 2` | 1 | 1,024 | 原 y slice |
| `yOutputQueue` | `O_t * 2` | 1 | 1,024 | 更新后的 y slice |
| `yInputFp32` | `O_t * 4` | 1 | 2,048 | y 升精度 |
| `yAccumFp32` | `O_t * 4` | 1 | 2,048 | expand 归约结果 + y |
| **group8 inplace 总计** |  |  | **154,432** | `H=2048` |
| **group8 reuse 总计** |  |  | **187,200** | `H=2048` |
| **group4 inplace 总计** |  |  | **154,144** | `H=4096` |
| **group4 reuse 总计** |  |  | **186,912** | `H=4096`，小于 910B 每核 192 KiB UB |

FP16 和 BF16 的元素大小相同，因此 UB 公式一致：

```text
inplaceUbBytes(int64 index) = 6 * G * H_a + 6 * max(H_a, 8192) + 12 * O_t + 72 * G + 256
reuseUbBytes(int64 index) = 6 * G * H_a + 10 * max(H_a, 8192) + 12 * O_t + 72 * G + 256
```

`H=4096, G=4, O_t=512` 时，inplace/reuse variant 分别使用
154,144/186,912 bytes，reuse 距离 192 KiB UB 上限剩余 9,696 bytes。Host 按
`M/H` 选择对应公式并查询
`ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`；当 CANN 9.0 在 910B3 返回 0 时，不把 0
当成真实容量，而依赖上述已封顶的 shape/UB 证明。若平台返回非零 UB，则必须验证
`ubBytes <= queriedUbBytes`，否则拒绝调用。

### 3.4 Generic UB 级切分

Generic 令 `P = 8192 / rank`，K/B tile 均固定为 8192 elements。所有 queue
slot 数为 1，FP16/BF16 的元素大小相同，int32/int64 单个 index 对齐后均占 32B：

| Buffer | 数量 | bytes/个 |
|---|---:|---:|
| `indicesQueue` | 1 | `A32(sizeof(index_t)) = 32` |
| `xQueue` | 1 | `A32(8192 * 2) = 16,384` |
| `weightQueue` | 1 | `A32(8192 * 2) = 16,384` |
| `yQueue` (`TQueBind<VECIN,VECOUT>`) | 1 | `A32(O * 2)` |
| `xFp32Buffer` | 1 | `A32(8192 * 4) = 32,768` |
| `weightProductFp32Buffer` | 1 | `A32(8192 * 4) = 32,768` |
| `rankPartialBuffer` | 1 | `A32(2 * rank * 4)` |
| `reduceTmpBuffer` | 1 | `A32(512) = 512` |
| `yInputFp32Buffer` | 1 | `A32(P * 4)` |
| `yAccumFp32Buffer` | 1 | `A32(P * 4)` |

其中 `A32(x) = align_up(x, 32)`。四个 rank 的各项天然满足 32B 对齐，总公式为：

```text
genericUbBytes(rank, O) = 98,848 + 8 * rank + 8 * (8192 / rank) + 2 * O
```

下表按最大 `O=16384` 计算；较小输出按实际 O 申请更少 UB：

| rank | `P = O_TILE` | 最大 Generic UB | 距 192 KiB 余量 |
|---:|---:|---:|---:|
| 8 | 1024 | 139,872 B | 56,736 B |
| 16 | 512 | 135,840 B | 60,768 B |
| 32 | 256 | 133,920 B | 62,688 B |
| 64 | 128 | 133,152 B | 63,456 B |

Host 必须按该逐项公式检查 Generic UB，不能复用 fast 的 group4/group8 公式。
单缓冲是第一版正式契约；双缓冲仅在实现真实预取流水并完成单独 A/B 后进入设计。

所有 GM 到 UB、UB 到 GM 搬运使用 `DataCopyPad`，尾部不足 32B 时由 pad 参数处理。

## 4. Workspace 与图捕获

算子不需要 workspace。输出 y 由调用方提供并原地更新；rank 中间结果只存在于 UB。

所有 shape、blockDim 和 buffer 大小只依赖 Tensor shape；Device 读取 indices 值只
决定 fast 的 8/4/2/1-row 分支或 Generic 当前行是否跳过，不产生动态输出 shape，
也不执行 Host `.item()`，因此支持 ACL Graph capture/replay。

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

1. 取消 `[M, R]` FP32 rank tensor 的 GM 写回、读回和一次 kernel launch。
2. 连续 8 行 index 相同时，A/B 权重 GM 搬运最多降到原来的 1/8；短分组保留
   4/2-row 路径。
3. X 在 shrink 的 R 个 rank 归约中常驻 FP32 UB，不重复从 GM 搬运。
4. Fast rank16 和 Generic rank8/16/32/64 都是编译期特化，移除 Device 热路径的
   runtime rank 分支。
5. 小/中 M quotient/remainder 分核减少空闲 AIV；mixed index 和 Generic group1
   不额外构造 permutation/workspace。
6. Generic 的 A rankBatch 减少窄 H 下的 DMA/Scalar 提交次数，8192-element K/B
   tile 控制 UB 峰值。

### 5.3 预期与门禁

以下是 fast 第一阶段的历史基线与门禁，保留用于回归对照，不作为尚未实测
Generic 的收益结论：

- 1024-token BGMV shrink+expand 基线：3121.991 us，占全路径 91.7%。
- 第一阶段目标：BGMV 降低 10%--20%，即节省约 312--624 us。
- 对应完整路径目标：3406.804 us -> 3095--2783 us，端到端提升约 9%--18%。
- 128/256/512/1024-token 中至少 3 个 case 提升，且 1024-token 提升不少于 8%，
  才默认启用新路径。
- 旧门禁让 1/2/8/32-token 保持 split。quotient/remainder fast 的真实 Qwen
  权重代理、top-k=6 同步筛选中，M=6--1020 的已测点在两个 DeepSeek-V4 目标
  shape 上已全部
  优于 split，因此新契约不再对 rank16 fast 设置 M 门槛；该筛选不替代完整
  msprof、ACL Graph 和端到端验证，也不能推导 Generic 一定更快。

预期是访存与 Vector 计算混合受限。8/4/2-row 路径主要减少 A/B GM 流量；Cast、
Mul、Reduce 指令数量基本不变，因此权重流量下降不会线性转化为 kernel 加速。
Generic 第一版保持单缓冲，后续双缓冲只有在 profiler 证明显式预取与 Vector
计算重叠且端到端提升后才可合入。

## 6. Kernel 实现要点

1. `TPipe` 在 kernel 入口创建并以指针传入类，避免 Scalar 常量折叠受阻。
2. `DataCopyPad` 的长度和 repeat 参数均由 Host 已验证的 uint32 shape 派生。
3. FP16/BF16 在 `Mul`、`ReduceSum` 和 `Add` 前 Cast 到 FP32。
4. reuse variant 的 `ReduceSum` 使用独立 product tensor；inplace variant 在每行
   计算前重新 Cast，不复用被归约的 tensor。
5. Fast Expand 复用现有 rank16 的 `BlockReduceSum + PairReduceSum`；Generic
   按 rank 使用 1--3 级归约，并在每一级显式补零尾部。
6. Generic 所有显式 vector repeat 不超过 128，低于 uint8 的 255 上限。
7. 不向 kernel 传右值；Host 将 `scale` 转为 FP32 局部变量后捕获并传入。
8. `M<=1024` 使用 quotient/remainder 行范围；大 M fast 按 group 对齐。不同 Core
   不写同一行 y。
9. shape/offset 乘法在 Host 先转为 uint64；正 index 范围由上游 combined-index
   契约保证，任意负 index 在 Device 跳过。

## 7. Python 集成策略

`PunicaWrapperNPU.add_lora_fused_moe` 在每个 A/B slice 上按以下顺序判断：

```text
can_fuse_semantics = (
    moe_lora_bgmv_fused is available
    and not fully_sharded
    and not mul_routed_weight
    and local_rank == full_rank
    and dtype in {FP16, BF16}
    and index_dtype in {int32, int64}
)

if not can_fuse_semantics:
    split
elif full_rank == 16 and H <= 4096 and O <= 4096:
    fast
elif (full_rank in {8, 16, 32, 64}
      and 1 <= H <= 16384 and 1 <= O <= 16384):
    Generic group1
else:
    split
```

Fast/Generic 都直接更新 y slice。Split 保留 shrink 后的 TP 通信、routed-weight
multiply 和 expand，因而是所有不支持 shape/模式的最终语义兜底。

历史生产路由对 fast 使用 `M>=512`，并对 `O>2048` 使用更保守的 `M>=1024`：
910B3 的 256->4096 实测中，768 行融合慢于 split，960/1152 行开始略有收益。
quotient/remainder 的上述真实权重筛选支持移除 rank16 fast 的 M 门槛；完整
msprof/ACL Graph 验证仍是生产放量门禁。旧门槛及其 768/960/1152 结果仅作为
回归历史，不得改写成 Generic 的实测收益。

## 8. 验证计划

### 8.1 精度

- rank 8/16/32/64 x FP16/BF16 x int32/int64 index。
- M 覆盖 0/1/2/3/47/48/49/511/512/1023/1024。
- H 覆盖 1/8191/8192/8193/16384；O 对每个 rank 覆盖
  `O_TILE-1/O_TILE/O_TILE+1/16384`。
- 全有效、全负、mixed、alternating、expert-sorted，以及连续 8/4/2 行相同 index。
- K/B tile 尾块逐级补零、非零 slice offset、in-place add、scale 为 0/负值。
- 与 `bgmv_shrink + bgmv_expand` 严格 parity，并与 PyTorch FP32 参考做 allclose。
- `fully_sharded`、`mul_routed_weight`、rank mismatch、unsupported dtype/shape 和
  fused op unavailable 必须断言走 split。
- ACL Graph capture/replay 覆盖 fast/Generic；尾块单独运行 mssanitizer。

### 8.2 性能

- 真实 Qwen3.5-35B-A3B router/expert 权重。
- Qwen 回归使用 top-k=8；DeepSeek-V4-Flash 目标 shape 使用 top-k=6，覆盖
  768/960/1152/1536/3072 routed rows。
- 记录 fused kernel、旧 shrink/expand 和完整 W13/W2 路径时间。
- 每条路径独立 3 轮，奇偶轮交换顺序，以 msprof active step 中位数比较。
- Generic 对 rank8/16/32/64 分别覆盖 K/B 整 tile 和尾 tile；M 覆盖
  1/48/512/1024/3072，并与同 shape split 路径逐项 A/B。
- 分别记录 MTE2、Vector、Scalar 时间；只有明确实现
  prologue/steady-state/epilogue 后才增加双缓冲候选，且必须与单缓冲比较。

## 9. 实现检查清单

- [x] 工程骨架已创建。
- [x] 接口、数学语义和回退条件已定义。
- [x] Block/UB 两级 tiling、UB 分配表和 coefficient 已推导。
- [x] FP16/BF16 FP32 计算路径已定义。
- [x] 无 workspace、静态 shape、ACL Graph 约束已定义。
- [x] 统一测试用例文档已生成。
- [x] Fast Host、Kernel、注册和 Python wrapper 已实现。
- [x] Fast 编译与基本功能测试通过。
- [ ] 中文接口 README 已生成。
- [x] Fast 至少 30 项精度测试通过（52/52）。
- [x] Fast 真实权重 torch_npu.profiler 门禁通过（22 case；生产路由 19 fused、
  3 fallback、0 回退）。
- [ ] Generic Host/Kernel/路由实现、编译与完整矩阵精度测试通过。
- [ ] Generic mssanitizer、ACL Graph capture/replay 和真实权重 profiler A/B 通过。

## 10. 参考实现

- `csrc/kernels/bgmv_shrink.cpp`
- `csrc/kernels/bgmv_expand.cpp`
- `csrc/ops/moe_lora_build_combined_idx/design.md`
- `vllm_ascend/lora/punica_npu.py::add_lora_fused_moe`
- PyTorch 标杆：按行选择 A/B 后执行 FP32 `matmul`，再 Cast/加回 y。
