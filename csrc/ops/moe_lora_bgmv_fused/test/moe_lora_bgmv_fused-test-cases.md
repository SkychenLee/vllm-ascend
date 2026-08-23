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

```python
SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
SUPPORTED_INDEX_DTYPES = [torch.int32, torch.int64]

# 字段：category, description,
#       (M, H, O, Y, slice_offset, index_dtype, index_pattern, scale)
TEST_SHAPES = [
    ("Grouped", "single 4-row group", (4, 64, 64, 64, 0, "int32", "group4_same", 1.0)),
    ("Grouped", "two groups with slice", (8, 128, 96, 160, 32, "int64", "group4_same", 0.5)),
    ("MoE", "small W2-like", (16, 512, 512, 512, 0, "int32", "expert_sorted", 1.0)),
    ("MoE", "Qwen W13 dimensions", (32, 2048, 1024, 1024, 0, "int32", "expert_sorted", 1.0)),
    ("MoE", "Qwen W2 dimensions", (32, 512, 2048, 2048, 0, "int32", "expert_sorted", 1.0)),
    ("Mixed", "mixed adapters inside group", (64, 768, 1000, 1100, 64, "int64", "mixed_within4", 0.25)),
    ("Grouped", "128 W13 rows", (128, 2048, 1024, 1024, 0, "int32", "expert_sorted", 1.0)),
    ("Fallback", "no consecutive equal index", (256, 512, 2048, 2048, 0, "int64", "alternating", 1.0)),
]

GENERAL_SHAPES = [
    ("Small", "minimum supported dimensions", (1, 17, 17, 17, 0, "int32", "single", 1.0)),
    ("Small", "unaligned H/O and slice", (3, 17, 19, 23, 2, "int64", "mixed_within4", 0.75)),
    ("Boundary", "all rows disabled", (4, 31, 33, 40, 3, "int32", "all_negative", 1.0)),
    ("Boundary", "tail row after one group", (5, 2048, 2048, 2048, 0, "int64", "group4_same", 1.0)),
    ("Large", "integration threshold W13", (512, 2048, 1024, 1024, 0, "int32", "expert_sorted", 1.0)),
    ("Large", "1024-row W2", (1024, 512, 2048, 2048, 0, "int32", "expert_sorted", 1.0)),
    ("Large", "4096-row W13", (4096, 2048, 1024, 1024, 0, "int32", "expert_sorted", 1.0)),
    ("Large", "8192-row W2", (8192, 512, 2048, 2048, 0, "int32", "expert_sorted", 1.0)),
]

BOUNDARY_VALUES = [
    ("indices", "all disabled", -1),
    ("indices", "minimum valid weight", 0),
    ("indices", "maximum valid weight", "num_weights - 1"),
    ("scale", "zero scale", 0.0),
    ("scale", "negative scale", -0.5),
    ("slice_offset", "last valid slice", "Y - O"),
]
```

### 2.2 用例覆盖统计

| 类别 | Shape 数量 | 边界值数量 | dtype 数量 | 总用例数 |
|---|---:|---:|---:|---:|
| 常规形状 | 8 | 0 | 2 | 16 |
| 泛化形状 | 8 | 0 | 2 | 16 |
| 边界值 | 0 | 6 | 2 | 12 |
| **合计** | **16** | **6** | **2** | **44** |

---

## 3. 使用说明

### 生成测试数据示例

```python
def build_indices(rows: int, pattern: str, dtype: torch.dtype) -> torch.Tensor:
    if pattern == "all_negative":
        values = torch.full((rows,), -1, dtype=torch.int64)
    elif pattern == "group4_same":
        values = torch.arange((rows + 3) // 4).repeat_interleave(4)[:rows]
    elif pattern == "alternating":
        values = torch.arange(rows, dtype=torch.int64).remainder(4)
    elif pattern == "mixed_within4":
        values = torch.arange(rows, dtype=torch.int64).remainder(3)
        values[::5] = -1
    elif pattern == "expert_sorted":
        values = torch.arange((rows + 7) // 8).repeat_interleave(8)[:rows]
    else:
        values = torch.zeros(rows, dtype=torch.int64)
    return values.to(dtype=dtype, device="npu")
```

### 注意事项

1. 每个 shape 同时覆盖 FP16 和 BF16；index dtype 按 shape 配置，并在边界回归中
   交换 int32/int64，确保四种 dtype 组合均被执行。
2. 精度用例使用较小 `num_weights`，性能用例使用 Qwen3.5-35B-A3B 的 256 experts。
3. `expert_sorted` 用于命中 4-row 快速路径；`alternating` 和 `mixed_within4` 用于验证
   1-row 回退路径。
4. 大 shape 只用于真实模型和性能评测；常规精度回归不重复构造超大 CPU gather。
