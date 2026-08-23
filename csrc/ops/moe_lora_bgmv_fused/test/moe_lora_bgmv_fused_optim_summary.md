# `moe_lora_bgmv_fused` 性能优化报告

## 排查发现

### 已处理问题

1. 4-row kernel 对连续 expert-sorted 路由的权重复用不足。第 1 轮扩展为
   8-row 快速路径，同时保留 4-row 和 1-row fallback。
2. 8-row kernel 虽减少 A/B 的 GM 搬运，但每个 routed row 仍重复把同一权重
   Cast 到 FP32。第 2 轮增加独立 `productFp32Buffer`，使组内 A/B 只 Cast
   一次，Mul/Reduce 写独立 scratch。
3. FP32 权重复用把 UB 占用从 154,432 B 提高到 187,200 B，小 W13 的
   shrink-heavy 场景会回退。第 3 轮生成编译期特化的 inplace/reuse 两套入口：
   `M < 2048 && H == 2048` 使用 inplace，其余 shape 使用 FP32 reuse。

### 已确认无问题

- Host 从设备查询 Vector Core 数量，blockDim 和 rows-per-core 未硬编码。
- GM 与 UB 之间全部使用 `DataCopyPad`，非 32B 对齐尾块已由精度用例覆盖。
- FP16/BF16 在 Mul、Reduce 和 Add 前均升到 FP32，最终才 Cast 回输出 dtype。
- `TPipe` 在 kernel 入口创建后传入类，避免阻止 Scalar 常量折叠。
- 无 workspace、无 Host `.item()`，不同 core 写不同 y 行，不需要 atomic。

### 未继续处理

- reuse variant 最大使用 187,200 B UB，距离 910B 每 Vector Core 192 KiB
  上限仅余 9,408 B，不适合再引入常规 double buffer。
- mixed/alternating index 只能走 1-row fallback。进一步优化需要显式 group
  offsets 或 Grouped GEMM/CATLASS，属于接口和数据流扩展，未纳入本次三轮迭代。
- EP、`fully_sharded=True`、routed-weight multiply 和 rank != 16 继续走原有
  shrink/expand 路径，本算子不改变这些语义。

## 测试口径

- 硬件：同一张空闲 Ascend 910B3（物理 NPU 0）。
- checkpoint：`/home/models/Qwen3.5-35B-A3B`。
- Router fingerprint：`d55bd73f2cfdb0bd`。
- W13 A/B fingerprint：`7c98ae45ea311629` / `55db3ee58409e318`。
- W2 A/B fingerprint：`181af914084c3f26` / `71c0edc469d32fc8`。
- checkpoint 没有 LoRA adapter；A/B 使用真实 expert 权重的 rank-16 子块，
  用于保持真实 dtype、shape、数值分布和访存形态。
- baseline 为备份提交 `f6df1ca`；final 为当前分支最终源码。两者使用相同
  JSONL、相同 checkpoint、相同卡和相同 separate 标杆。
- 每条路径独立 3 轮并交换执行顺序，固定 warmup=5、active=5、repeat=1，
  报告取三轮中位数。
- 物理 NPU 7 和 NPU 6 上出现外部并发负载的批次已排除，未计入下表。

## 优化前基线

同卡重编译 `f6df1ca` 后，4-row baseline 的平均 separate/fused 加速比为
1.398x；10/10 case 均优于 separate。

## 迭代历史

### 第 1 轮：8-row 权重复用

- 代码修改：连续 8 行 index 相同时复用 A/B 搬运，保留 4-row fallback。
- 精度结果：44/44 通过。
- 历史性能结果：平均 separate/fused 加速比 1.974x。
- 决策：保留。

### 第 2 轮：组内 FP32 权重复用

- 代码修改：增加独立 FP32 product scratch，A/B 每组只 Cast 一次。
- 精度结果：44/44 通过。
- 历史性能结果：平均 separate/fused 加速比 2.002x；相对第 1 轮 7/10
  case 改善，平均绝对改善 14.17%；相对最初 4-row 9/10 case 改善，
  平均绝对改善 22.12%。
- 决策：保留；继续修复小 W13 回退。

### 第 3 轮：双 kernel 选择

- 代码修改：生成 inplace/reuse 两套 FP16/BF16、int32/int64 入口；小 W13
  使用 154,432 B inplace variant，其余使用 187,200 B reuse variant。
- 精度结果：44/44 通过，覆盖 1/4/8-row、非对齐尾块、负 index、slice 和
  inplace alias。
- 同卡性能结果：相对 4-row baseline 9/10 case 改善，平均每 case 提升
  19.15%，总耗时从 4,886.390 us 降到 3,988.288 us（提升 22.52%）。
  唯一回退为最小 W13 Case 0 的 0.56%，处于微小抖动范围。
- 决策：保留并停止迭代；三轮上限已到。

## 最终性能对比

表中 baseline、final 和 separate 均来自物理 NPU 0 的受控采集。

| Case | Shape | DType | 4-row baseline(us) | Final fused(us) | Baseline -> final | Separate(us) | Final vs separate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 W13 | [1024, 2048] | BF16 | 188.760 | 189.824 | -0.56% | 213.060 | 1.122x |
| 1 W13 | [2048, 2048] | BF16 | 315.390 | 282.718 | +11.56% | 419.528 | 1.484x |
| 2 W13 | [4096, 2048] | BF16 | 540.007 | 435.521 | +23.99% | 823.220 | 1.890x |
| 3 W13 | [8192, 2048] | BF16 | 947.115 | 764.111 | +23.95% | 1632.585 | 2.137x |
| 4 W2 | [1024, 512] | BF16 | 181.540 | 172.063 | +5.51% | 187.484 | 1.090x |
| 5 W2 | [2048, 512] | BF16 | 297.134 | 260.569 | +14.03% | 366.715 | 1.407x |
| 6 W2 | [4096, 512] | BF16 | 506.218 | 390.272 | +29.71% | 719.218 | 1.843x |
| 7 W2 | [8192, 512] | BF16 | 864.165 | 669.237 | +29.13% | 1424.753 | 2.129x |
| 8 W13 | [4096, 2048] | FP16 | 540.887 | 434.869 | +24.38% | 821.864 | 1.890x |
| 9 W2 | [4096, 512] | FP16 | 505.174 | 389.104 | +29.83% | 717.862 | 1.845x |

### 汇总

| 指标 | 4-row baseline | Final |
| --- | ---: | ---: |
| 平均 separate/fused 加速比 | 1.398x | 1.684x |
| BF16 平均 separate/fused 加速比 | 1.380x | 1.638x |
| FP16 平均 separate/fused 加速比 | 1.472x | 1.867x |
| 优于 separate 的 case | 10/10 | 10/10 |

## 结论

1. 该算子是权重搬运、FP32 Cast 和 Vector Mul/Reduce 混合受限；同一 expert
   的 routed rows 增多后，权重搬运与 Cast 复用收益明显放大。
2. 最终版本相对 4-row baseline 的平均每 case 提升为 19.15%，按 10 case
   总耗时计算提升 22.52%；中大 W2 的提升最高，达到约 29%。
3. 最小 W13 的 expert group 较短，8-row/reuse 收益有限；inplace variant 把它
   恢复到 4-row baseline 附近，最终仅有 0.56% 微小回退。
4. 最终版本对 separate 的 10/10 case 全胜，平均加速比 1.684x；FP16 与 BF16
   均通过真实权重精度门禁。
5. 剩余优化空间主要在 1-row fallback 和显式 group offsets/Grouped GEMM，
   需要扩大接口与上游路由数据流，不能由现有 kernel 内局部调优安全解决。
