# `moe_lora_build_combined_idx` 用例设计文档

## 1. 算子标杆

PyTorch 参考实现：

```python
import torch


def reference(
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_indices: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    top_k = topk_ids.shape[1]
    inv_perm = torch.argsort(torch.abs(expanded_row_idx))
    expert_per_row = topk_ids.reshape(-1)[inv_perm].to(torch.int64)
    orig_token = (inv_perm // top_k).clamp_(
        max=token_lora_indices.numel() - 1
    )
    lora_per_row = token_lora_indices[orig_token]
    safe_lora = lora_per_row.clamp(min=0)
    enabled = (lora_per_row >= 0) & adapter_enabled[safe_lora].bool()
    return torch.where(
        enabled,
        safe_lora * num_experts + expert_per_row,
        torch.full_like(lora_per_row, -1),
    ).to(torch.int32).contiguous()
```

NPU 正式算子调用：

```python
actual = torch.ops._C_ascend.moe_lora_build_combined_idx(
    expanded_row_idx,
    topk_ids,
    token_lora_indices,
    adapter_enabled,
    num_experts,
)
```

---

## 2. 用例说明

### 2.1 测试配置

```python
# 本算子的计算 dtype 固定；这里遍历 adapter_enabled 支持的两种存储类型。
SUPPORTED_DTYPES = [torch.bool, torch.int8]

# 每项参数为 (tokens, top_k, num_experts, max_loras)。
TEST_SHAPES = [
    ("Decode", "1 token, DeepSeek top-k",       (1, 6, 160, 16)),
    ("Decode", "2 tokens",                     (2, 6, 160, 16)),
    ("Decode", "8 tokens",                     (8, 6, 160, 16)),
    ("Decode", "32 tokens",                   (32, 6, 160, 16)),
    ("Prefill", "128 tokens",                (128, 6, 160, 16)),
    ("Prefill", "512 tokens",                (512, 6, 160, 16)),
    ("TopK", "top-k 1",                       (32, 1, 256, 4)),
    ("TopK", "top-k 2",                       (32, 2, 64, 8)),
    ("TopK", "top-k 8",                       (32, 8, 256, 16)),
    ("Experts", "small expert table",          (16, 2, 8, 4)),
]

GENERAL_SHAPES = [
    ("Small", "single pair",                    (1, 1, 8, 1)),
    ("Small", "non-aligned 5 pairs",            (5, 1, 8, 4)),
    ("Small", "non-aligned 18 pairs",           (3, 6, 160, 4)),
    ("Large", "2048-token prefill",          (2048, 6, 160, 16)),
    ("Large", "4096-token prefill",          (4096, 6, 160, 16)),
    ("Capacity", "mapping capacity exceeds T",  (17, 2, 64, 16)),
]

BOUNDARY_VALUES = [
    ("all_no_lora", "all token_lora_indices are -1"),
    ("all_disabled", "all adapter_enabled values are zero"),
    ("all_enabled", "all adapter_enabled values are nonzero"),
    ("max_ids", "use max_loras-1 and num_experts-1"),
    ("negative_expanded", "encode nonzero sorted positions as negative"),
]
```

### 2.2 用例覆盖统计

| 类别 | Shape数量 | 边界值数量 | dtype数量 | 总用例数 |
|---|---:|---:|---:|---:|
| 常规形状 | 10 | 0 | 2 | 20 |
| 泛化形状 | 6 | 0 | 2 | 12 |
| 边界值 | 0 | 5 | 2 | 10 |
| **总计** | **16** | **5** | **2** | **42** |

---

## 3. 使用说明

### 生成测试数据示例

```python
def make_case(
    tokens: int,
    top_k: int,
    num_experts: int,
    max_loras: int,
    adapter_dtype: torch.dtype,
    *,
    capacity_extra: int = 0,
    seed: int = 20260822,
    device: str = "npu",
):
    generator = torch.Generator().manual_seed(seed)
    num_pairs = tokens * top_k
    expanded = torch.randperm(
        num_pairs, generator=generator, dtype=torch.int64
    ).to(torch.int32)
    if num_pairs > 1:
        negative_mask = (torch.arange(num_pairs) % 2 == 1) & (expanded != 0)
        expanded[negative_mask] = -expanded[negative_mask]
    topk_ids = torch.randint(
        0,
        num_experts,
        (tokens, top_k),
        generator=generator,
        dtype=torch.int32,
    )
    token_lora = torch.randint(
        -1,
        max_loras,
        (tokens + capacity_extra,),
        generator=generator,
        dtype=torch.int64,
    )
    adapter_enabled = torch.randint(
        0,
        2,
        (max_loras,),
        generator=generator,
        dtype=torch.int8,
    ).to(adapter_dtype)
    return tuple(
        tensor.to(device)
        for tensor in (expanded, topk_ids, token_lora, adapter_enabled)
    )
```

### 注意事项

1. `abs(expanded_row_idx)` 必须是 `[0, tokens * top_k)` 的排列；重复和越界值属于非法输入。
2. `token_lora_indices` 可以大于 token 数，以覆盖 vLLM 预分配 capacity 的真实布局。
3. 参考实现和正式算子必须逐元素严格相等，不使用浮点容差。
4. 性能用例生成输入后应在计时区外完成输出分配和正确性检查。
