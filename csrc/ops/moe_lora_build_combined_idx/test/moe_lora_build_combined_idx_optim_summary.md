# `moe_lora_build_combined_idx` 性能优化报告

## 测试口径

- 平台：Ascend 910B3，CANN 9.0，物理 NPU 7。
- Checkpoint：`/home/models/Qwen3.5-35B-A3B`。
- 模型配置：hidden=2048、expert intermediate=512、experts=256、top-k=8、BF16、LoRA rank=16。
- Router、W13、W2 均读取 layer-0 真实 checkpoint 权重；A/B 是真实 expert 权重子块，
  不是训练得到的 LoRA adapter。
- Router 输入由 checkpoint 权重行确定性构造，不是线上抓取的真实激活。
- 每条路径独立采集 3 轮，中位数作为结果；奇偶轮交换 custom/baseline 顺序。
- 标杆路径包含 AiCPU argsort、W13/W2 分别构造 int64 combined_idx 和原 int64 BGMV；
  最终路径包含一次 AscendC fused routing、W13/W2 复用 int32 combined_idx 和 int32 BGMV。

## 排查发现

1. `bgmv_expand::CopyInX` 每个 routed row 最多执行 64 次标量
   `GetValue/SetValue` 来复制 rank 向量，属于 Scalar 指令开销；已改为一次高维
   `AscendC::Copy`，GM 到 UB 同时统一改为 `DataCopyPad`。
2. `bgmv_shrink` 在 rank 维逐行搬运 W、Cast、Mul、ReduceSum。1024-token 实测中
   shrink 占最终全路径 64.1%，是最大的剩余热点。
3. 给 shrink 的 W 队列增加双缓冲没有改变 Vector/Scalar/MTE2 周期，却额外占用
   约 23 KiB UB；已回退。
4. 将 rank-16 的 8 行归约批量化没有降低 Scalar 比例，反而增加 X 复制，
   shrink 回退约 0.49%；已回退。
5. fused routing 已消除 AiCPU argsort 和两次 combined_idx 构造。它是小 shape 的
   主要收益来源；大 shape 中 routing 仅占 5.2%，继续只优化 routing 的上限很低。

## 优化前基线

原始融合实现与同轮 PyTorch 标杆的真实权重全路径结果如下。

| Shape `[tokens, top-k]` | 原始融合实现 (us) | 原始标杆 (us) | 加速比 |
|---|---:|---:|---:|
| `[1, 8]` | 41.357 | 144.831 | 3.502x |
| `[2, 8]` | 44.389 | 148.135 | 3.337x |
| `[8, 8]` | 85.074 | 204.825 | 2.408x |
| `[32, 8]` | 173.292 | 317.099 | 1.830x |
| `[128, 8]` | 482.110 | 706.806 | 1.466x |
| `[256, 8]` | 906.422 | 1244.433 | 1.373x |
| `[512, 8]` | 1726.215 | 2294.598 | 1.329x |
| `[1024, 8]` | 3349.227 | 4468.261 | 1.334x |

8 个 case 的平均对标加速比为 2.072x，全部优于标杆。

## 迭代历史

### 第 1 轮：消除 expand 的标量 rank 复制

- 代码修改：`CopyInX` 使用 `DataCopyPad` 搬入 X，并用一次高维
  `AscendC::Copy` 取代每 token 最多 64 次标量 `GetValue/SetValue`。
- 精度结果：52/52 PASS，BGMV int32/int64 parity 为 0。
- 1024-token 定点结果：expand 940.027 us -> 937.507 us，提升 0.27%；
  全路径为 3408.444 us。未改动的 shrink 同轮漂移到 2192.088 us，因此不能用
  全路径绝对值归因该改动。
- 决策：保留。代码更直接地使用 Vector 搬运，且定点结果有微小正收益。

### 第 2 轮：shrink W 双缓冲

- 代码修改：W 队列改为双缓冲，尝试重叠 GM 搬运与 Vector 计算。
- 精度结果：52/52 PASS。
- 1024-token 结果：shrink 2128.983 us；与原始 2129.859 us 等价。Profiler 的
  Vector/Scalar/MTE2 周期几乎逐项一致，无法证明流水重叠收益，并额外占用约
  23 KiB UB。
- 决策：回退。

### 第 3 轮：shrink 8-rank 批量归约

- 代码修改：尝试把 rank-16 的多行点积合并为批量 AR 归约。
- 精度结果：52/52 PASS。
- 1024-token 结果：shrink 2139.323 us，较原始约回退 0.44%--0.49%；Scalar
  比例没有下降，额外 X 复制抵消了潜在收益。
- 决策：回退。

三轮迭代已达到性能优化流程上限。最终代码只保留第 1 轮的 expand 搬运修改；
shrink 保留原实现以及此前已验证的 int32/int64 index 模板支持。

## 最终精度

| 类别 | 通过数 | 失败数 | 判定 |
|---|---:|---:|---|
| Routing 严格相等 | 42 | 0 | mismatch=0 |
| BGMV index 兼容 | 8 | 0 | int32/int64 parity=0 |
| ACL Graph capture/replay | 2 | 0 | mismatch=0 |
| **合计** | **52** | **0** | **PASS** |

覆盖 FP16/BF16、shrink/expand、int32/int64 index、bool/int8 adapter flag、
1-token decode、4096-token prefill 和 ACL Graph capture/replay。

## 最终性能对比

下表中的“原始融合”与“最终融合”来自两个独立采集批次；正的变化表示最终更快。
由于同批标杆也发生 0.4%--6.1% 漂移，原始到最终的绝对变化只用于展示，不能全部
归因于代码。可靠的新增代码收益以第 1 轮同口径的 expand 定点结果为准。

| Shape | 原始融合 (us) | 最终融合 (us) | 原始到最终 | 最终标杆 (us) | 最终加速比 |
|---|---:|---:|---:|---:|---:|
| `[1, 8]` | 41.357 | 41.969 | -1.46% | 146.071 | 3.480x |
| `[2, 8]` | 44.389 | 44.777 | -0.87% | 155.079 | 3.463x |
| `[8, 8]` | 85.074 | 90.866 | -6.37% | 218.197 | 2.401x |
| `[32, 8]` | 173.292 | 181.952 | -4.76% | 336.623 | 1.850x |
| `[128, 8]` | 482.110 | 489.654 | -1.54% | 716.387 | 1.463x |
| `[256, 8]` | 906.422 | 915.430 | -0.98% | 1261.053 | 1.378x |
| `[512, 8]` | 1726.215 | 1730.094 | -0.22% | 2303.870 | 1.332x |
| `[1024, 8]` | 3349.227 | 3406.804 | -1.69% | 4535.691 | 1.331x |

- 最终 8 个 case 全部优于标杆，平均加速比为 **2.087x**。
- 原始平均对标加速比为 2.072x；最终提高到 2.087x，但约 0.7% 的变化处于跨批次
  性能噪声量级，不能声称为稳定端到端增益。
- 原始融合时间合计 6808.086 us，最终合计 6901.546 us，跨批次绝对值回退 1.35%；
  同期标杆也整体变慢，因此它不是对 expand 修改的有效 A/B 结论。

### 1024-token 最终热点

| 部分 | 时间 (us) | 占最终全路径 |
|---|---:|---:|
| Fused routing | 175.688 | 5.2% |
| BGMV shrink | 2182.624 | 64.1% |
| BGMV expand | 939.367 | 27.6% |
| Other | 108.934 | 3.2% |
| **合计** | **3406.804** | **100.0%** |

BGMV shrink+expand 合计 3121.991 us，占 91.7%。当前大 shape 已从“argsort/调度
受限”转为“BGMV Vector 计算、归约和权重访问受限”。

## 结论与剩余空间

1. 融合算子的主要价值已经兑现：消除 AiCPU argsort、减少小算子调度、只生成一次
   int32 combined_idx 并让 W13/W2 复用。最终真实权重全路径平均为 2.087x，
   1-token 为 3.480x，1024-token 为 1.331x。
2. 本轮可归因的新增收益很小：expand 定点提升约 0.27%，完整 8-case 复测没有证明
   稳定的端到端增益。双缓冲和批量归约均无效并已回退，没有留下性能负资产。
3. 1024-token 的 91.7% 时间在 BGMV，继续优化 routing 的理论上限只有约 5.2%；
   下一阶段必须重构 shrink/expand 的计算映射，而不是继续做标量级微调。
4. 若目标是端到端再提升 10%，1024-token 至少要从 BGMV 中减少约 341 us，等价于
   shrink+expand 提升约 10.9%。这需要验证 Cube/批量 GEMM、权重布局重排或跨 token
   合批方案；当前逐 token Vector dot/reduce 框架很难达到该数量级。
5. 当前测试是 AllGather、非 EP 路径。`fully_sharded` 的通信融合需要在多卡 HCCL
   环境单独开发和测量，不能由这组单卡数据外推收益。
