# `moe_lora_bgmv_fused` 用例设计文档

## 1. 算子标杆

PyTorch 参考实现：

```python
import torch


def moe_lora_bgmv_fused_reference(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    indices: torch.Tensor,
    y: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    scale: float,
) -> torch.Tensor:
    safe_indices = indices.clamp_min(0).to(torch.long)
    selected_a = lora_a[safe_indices].float()
    selected_b = lora_b[safe_indices].float()
    rank_out = torch.bmm(
        x.float().unsqueeze(1), selected_a.transpose(1, 2)
    ).squeeze(1)
    rank_out.mul_(scale)
    delta = torch.bmm(
        rank_out.unsqueeze(1), selected_b.transpose(1, 2)
    ).squeeze(1)
    delta.masked_fill_(indices.lt(0).unsqueeze(1), 0.0)
    expected = y.clone()
    output_slice = expected[:, slice_offset : slice_offset + slice_size]
    output_slice.copy_((output_slice.float() + delta).to(y.dtype))
    return expected
```

NPU 调用方式：

```python
actual = torch.ops._C_ascend.moe_lora_bgmv_fused(
    x,
    lora_a,
    lora_b,
    indices,
    y,
    slice_offset,
    slice_size,
    scale,
)
```

---

## 2. 用例说明

### 2.1 测试配置

所有 `TEST_SHAPES`、`GROUPED_SHAPES` 和 `GENERAL_SHAPES` 均执行
FP16/BF16 x int32/int64 的完整笛卡尔积。`rank` 已包含在 shape 字段中，保证
Fast、GenericGrouped 和 Generic group1 的编译期实例都覆盖对应 dtype/index
组合。

```python
SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
SUPPORTED_INDEX_DTYPES = [torch.int32, torch.int64]
SUPPORTED_RANKS = [8, 16, 32, 64]

# 字段：category, description,
#       (M, rank, H, O, Y, slice_offset, index_pattern, scale,
#        expected_route)
TEST_SHAPES = [
    ("Fast", "TP8 W13 4096->256", (48, 16, 4096, 256, 512, 256, "expert_sorted", 1.0, "fast")),
    ("Fast", "TP8 W2 256->4096", (48, 16, 256, 4096, 4096, 0, "expert_sorted", 1.0, "fast")),
    ("Fast", "rank16 maximum fast H/O", (512, 16, 4096, 4096, 4096, 0, "group8_same", 0.5, "fast")),
    ("Generic", "rank8 small-M output tile", (47, 8, 4096, 1024, 1024, 0, "alternating", 1.0, "generic")),
    ("Generic", "rank16 H above fast range", (49, 16, 4097, 512, 544, 16, "mixed", -0.5, "generic")),
    ("Generic", "rank32 K tile boundary", (48, 32, 8192, 256, 256, 0, "expert_sorted", 1.0, "generic")),
    ("Generic", "rank64 maximum H", (3, 64, 16384, 128, 128, 0, "single", 1.0, "generic")),
    ("Generic", "small-M unsorted nonzero slice", (64, 32, 257, 513, 577, 32, "mixed", 0.25, "generic")),
    ("Fast", "large-M group-aligned W13", (3072, 16, 4096, 256, 256, 0, "expert_sorted", 1.0, "fast")),
    ("Fast", "large-M group-aligned W2", (3072, 16, 256, 4096, 4096, 0, "expert_sorted", 1.0, "fast")),
]

# Grouped 必须让单个 Core 实际拿到 2/4 行；M 很小时 coreNum=M，只会执行 G1。
GROUPED_SHAPES = [
    ("Grouped", "rank8 maximum UB group4", (520, 8, 4096, 4096, 4096, 0, "group4_same", 1.0, "generic_grouped")),
    ("Grouped", "rank32 group2", (400, 32, 4096, 4096, 4096, 0, "group2_same", 0.5, "generic_grouped")),
    ("Grouped", "rank64 group4 plus core tail", (401, 64, 4095, 4096, 4128, 16, "group4_same", -0.5, "generic_grouped")),
    ("Grouped", "G4 G2 negative G1 mixed", (400, 32, 257, 513, 577, 32, "grouped_mixed", 0.25, "generic_grouped")),
    ("Grouped", "different negative indices", (400, 64, 17, 129, 145, 8, "mixed_negative", 1.0, "generic_grouped")),
    ("Grouped", "strided y O tail", (520, 8, 4096, 4095, 4127, 16, "group4_same", 1.0, "generic_grouped")),
]

# 5 个 H 边界 + 11 个 M 边界 + 每个 rank 的 4 个 O 边界 = 32 个 shape。
GENERAL_SHAPES = [
    # H/K tile 边界；rank32 强制进入 Generic。
    ("KBoundary", "minimum H", (3, 32, 1, 257, 257, 0, "mixed", 1.0, "generic")),
    ("KBoundary", "K_TILE-1", (3, 32, 8191, 257, 257, 0, "all_negative", 1.0, "generic")),
    ("KBoundary", "exact K_TILE", (3, 32, 8192, 257, 257, 0, "single", 0.0, "generic")),
    ("KBoundary", "K_TILE+1", (3, 32, 8193, 257, 273, 8, "mixed", -0.5, "generic")),
    ("KBoundary", "maximum H", (3, 32, 16384, 257, 257, 0, "alternating", 1.0, "generic")),

    # quotient/remainder 与 1024 分界；M=0 必须不启动 kernel。
    ("MBoundary", "empty rows", (0, 16, 64, 64, 64, 0, "empty", 1.0, "noop")),
    ("MBoundary", "M=1", (1, 16, 64, 64, 64, 0, "single", 1.0, "fast")),
    ("MBoundary", "M=2", (2, 16, 64, 64, 64, 0, "group2_same", 1.0, "fast")),
    ("MBoundary", "M=3", (3, 16, 64, 64, 64, 0, "mixed", -0.5, "fast")),
    ("MBoundary", "M=47", (47, 16, 64, 64, 64, 0, "alternating", 1.0, "fast")),
    ("MBoundary", "M=48", (48, 16, 64, 64, 64, 0, "expert_sorted", 1.0, "fast")),
    ("MBoundary", "M=49", (49, 16, 64, 64, 64, 0, "all_negative", 1.0, "fast")),
    ("MBoundary", "M=511", (511, 16, 64, 64, 64, 0, "group4_same", 1.0, "fast")),
    ("MBoundary", "M=512", (512, 16, 64, 64, 64, 0, "group8_same", 1.0, "fast")),
    ("MBoundary", "M=1023", (1023, 16, 64, 64, 64, 0, "expert_sorted", 1.0, "fast")),
    ("MBoundary", "M=1024", (1024, 16, 64, 64, 64, 0, "expert_sorted", 1.0, "fast")),

    # 对每个 rank，P=8192/rank；覆盖 P-1/P/P+1 和最大 O。
    ("OBoundary", "rank8 P-1", (3, 8, 64, 1023, 1023, 0, "mixed", 1.0, "generic_grouped")),
    ("OBoundary", "rank8 P", (3, 8, 64, 1024, 1024, 0, "single", 1.0, "generic_grouped")),
    ("OBoundary", "rank8 P+1", (3, 8, 64, 1025, 1041, 8, "all_negative", -0.5, "generic_grouped")),
    ("OBoundary", "rank8 max O", (1, 8, 64, 16384, 16384, 0, "single", 1.0, "generic")),
    ("OBoundary", "rank16 P-1", (3, 16, 4097, 511, 511, 0, "mixed", 1.0, "generic")),
    ("OBoundary", "rank16 P", (3, 16, 4097, 512, 512, 0, "single", 1.0, "generic")),
    ("OBoundary", "rank16 P+1", (3, 16, 4097, 513, 529, 8, "all_negative", 0.0, "generic")),
    ("OBoundary", "rank16 max O", (1, 16, 64, 16384, 16384, 0, "single", 1.0, "generic")),
    ("OBoundary", "rank32 P-1", (3, 32, 64, 255, 255, 0, "mixed", 1.0, "generic_grouped")),
    ("OBoundary", "rank32 P", (3, 32, 64, 256, 256, 0, "single", 1.0, "generic_grouped")),
    ("OBoundary", "rank32 P+1", (3, 32, 64, 257, 273, 8, "all_negative", -0.5, "generic_grouped")),
    ("OBoundary", "rank32 max O", (1, 32, 64, 16384, 16384, 0, "single", 1.0, "generic")),
    ("OBoundary", "rank64 P-1", (3, 64, 64, 127, 127, 0, "mixed", 1.0, "generic_grouped")),
    ("OBoundary", "rank64 P", (3, 64, 64, 128, 128, 0, "single", 1.0, "generic_grouped")),
    ("OBoundary", "rank64 P+1", (3, 64, 64, 129, 145, 8, "all_negative", 0.0, "generic_grouped")),
    ("OBoundary", "rank64 max O", (1, 64, 64, 16384, 16384, 0, "single", 1.0, "generic")),
]

# Python 路由用例不直接调用 fused op；每项都必须断言执行 split 路径。
ROUTING_FALLBACK_CASES = [
    ("fully_sharded", {"fully_sharded": True}),
    ("routed_weight", {"mul_routed_weight": True}),
    ("rank_mismatch", {"local_rank": 8, "full_rank": 16}),
    ("unsupported_rank", {"local_rank": 4, "full_rank": 4}),
    ("unsupported_data_dtype", {"dtype": torch.float32}),
    ("unsupported_index_dtype", {"index_dtype": torch.int16}),
    ("H_above_max", {"H": 16385}),
    ("O_above_max", {"O": 16385}),
    ("fused_op_unavailable", {"moe_lora_bgmv_fused": None}),
]

# 定向值已嵌入上述 shape，不再与全部 shape 做额外笛卡尔积。
BOUNDARY_VALUES = [
    ("indices", "all disabled", -1),
    ("indices", "minimum valid weight", 0),
    ("indices", "maximum valid weight", "num_weights - 1"),
    ("scale", "zero scale", 0.0),
    ("scale", "negative scale", -0.5),
    ("slice_offset", "nonzero and last valid slice", "Y - O"),
]
```

### 2.2 用例覆盖统计

| 类别 | 基础 shape/条件数 | data dtype 数 | index dtype 数 | 设计用例数 |
|---|---:|---:|---:|---:|
| 典型 Fast/GenericGrouped/Generic | 10 | 2 | 2 | 40 |
| GenericGrouped G4/G2/G1 定向 | 6 | 2 | 2 | 24 |
| H/M/O 泛化边界 | 32 | 2 | 2 | 128 |
| Python split 路由 | 9 | 定向 | 定向 | 9 |
| **合计** | **57** | - | - | **201** |

现有 fast 回归历史仍为 52/52 精度通过；22-case 真实权重 profiler 的历史生产
路由为 19 fused、3 fallback、0 回退。上表是新增 Generic/三级路由的设计矩阵，
在实际执行前不得写成已通过结果。

---

## 3. 使用说明

### 生成测试数据示例

```python
def build_indices(
    rows: int,
    pattern: str,
    dtype: torch.dtype,
    num_weights: int = 256,
) -> torch.Tensor:
    if pattern == "empty":
        values = torch.empty((0,), dtype=torch.int64)
    elif pattern == "all_negative":
        values = torch.full((rows,), -1, dtype=torch.int64)
    elif pattern == "group8_same" or pattern == "expert_sorted":
        values = torch.arange((rows + 7) // 8).repeat_interleave(8)[:rows]
    elif pattern == "group4_same":
        values = torch.arange((rows + 3) // 4).repeat_interleave(4)[:rows]
    elif pattern == "group2_same":
        values = torch.arange((rows + 1) // 2).repeat_interleave(2)[:rows]
    elif pattern == "grouped_mixed":
        # 每 8 行依次触发 G4、G2、negative G1、positive G1。
        base = torch.tensor([0, 0, 0, 0, 1, 1, -1, 2])
        values = base.repeat((rows + 7) // 8)[:rows]
    elif pattern == "mixed_negative":
        # 不同负值必须分别跳过，后两行仍可触发 positive G2。
        base = torch.tensor([-1, -2, 0, 0])
        values = base.repeat((rows + 3) // 4)[:rows]
    elif pattern == "alternating":
        values = torch.arange(rows, dtype=torch.int64).remainder(4)
    elif pattern == "mixed":
        values = torch.arange(rows, dtype=torch.int64).remainder(3)
        values[::5] = -1
    else:
        values = torch.zeros(rows, dtype=torch.int64)
    valid = values.ge(0)
    values[valid] = values[valid].remainder(num_weights)
    return values.to(dtype=dtype, device="npu")
```

### 注意事项

1. Fast 契约为 `rank=16 && H/O<=4096`；非 fast 的 `rank in {8,32,64}`、
   `H/O<=4096`、grouped GM stride 可由 uint32 字节表示，并满足 rank8
   `M>=512` 或 rank32/64 `M>=384` 时进入 GenericGrouped；门槛以下及其余
   `rank in {8,16,32,64}`、`H/O<=16384` 进入 Generic group1。`M<=1024`
   的 fast 用例必须额外核对 quotient/remainder 的 rowBegin/rowCount，`M>1024`
   核对 group 对齐范围。
2. `KBoundary` 验证 8192 K tile；`OBoundary` 验证 8192 B 元素 tile。尾 B tile、
   BlockReduce 输出和每一级 PairReduce 输入的未写位置都必须显式补零。
3. all-negative 行的 y 必须逐 bit 保持不变；alternating/mixed 不允许错误复用相邻
   expert 权重。非零 slice 之外的 y 前缀和后缀也必须逐 bit 保持不变。
4. `fully_sharded`、`mul_routed_weight` 和 `local_rank != full_rank` 需要保留
   shrink/通信或权重乘/expand 的原始调用顺序，不能只检查最终数值。
5. 大 shape 使用最少的 `num_weights` 避免构造无关巨型权重；正 index 最大值仍需
   覆盖 `num_weights-1`。M=0 断言无 kernel launch。
6. Generic 尾块矩阵通过后单独运行 mssanitizer；Fast、GenericGrouped 和 Generic
   group1 各覆盖一次 ACL Graph capture/replay。真实权重性能结果另行记录，不能
   由精度矩阵推导。
