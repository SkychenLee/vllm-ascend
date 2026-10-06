# TP8＋EP8＋LoRA 本轮优化与整网实测（2026-10-07）

本轮源码与原生算子已安装到 `ds-ltc-v25-ep:{installed_repo}`。使用物理卡 8–15 完成同参数基线、优化版正式性能测试及独立 profile。两个正式测试均成功完成 8 个请求、65536 输入和 32768 输出 token，无 prefix cache 命中和抢占。

| 指标 | 基线 | 优化后 | 变化 |
|---|---:|---:|---:|
| 端到端输出吞吐 | 259.43 token/s | 278.56 token/s | +7.37% |
| 完成全部请求耗时 | 126.31s | 117.63s | −6.87% |
| 平均 TTFT | 17.55s | 13.48s | −23.17% |
| 平均 TPOT | 23.08ms | 20.11ms | −12.90% |
| DSpark draft token 接受率 | 85.87% | 82.72% | −3.15 个百分点 |
| 同形状主模型 decode 图均值 | 81.61ms | 66.40ms | −18.64% |

正式输入为固定 token-ID 随机 prompt，每请求精确 8192 输入＋4096 输出、8 并发，`ignore_eos`。模型 DeepSeek-V4-Flash-0731-w8a8，LoRA lora1/r16/max_loras3，AllGather TP8/EP8/DP1，显式 V1 DSpark5/greedy，FULL_DECODE_ONLY＋npugraph_ex，max_num_batched_tokens16384、max_num_seqs8、max_model_len128000、gpu_memory_utilization0.93。同一 seed2026100702、prompt bytes 和服务参数，fresh engine，profile 测试与正式计时分开。

两组输出 token 序列均未逐 token 等同；接受率也有变化。本轮按用户要求优先优化性能，没有完成模型质量/精度等价验证，不能将吞吐差值全部归因某一算子。每个条件仅一次正式测试，不宣称统计稳定性。LoRA 仅有部分 tensor 的零值采样，未证明整套 adapter 全零或已训练质量。

## 已实现与实际命中

- W13 普通浮点双 A shrink：共享 X 读取与 FP32 转换，单 kernel 计算 gate/up，采用跨核步进的行分配；保留原 FP32 reduction 顺序、两个 B expand 和 W2。H4096/r16/BF16或FP16、M≤384，排除 fully-sharded/INT8/310P 等不匹配情况。
- EP 路由与 adapter/expert 索引融合：本场景每层 34 个恢复 kernel→1、10 个索引 kernel→1；合法 guard miss 保留原路径。
- 普通 EP SwiGLU＋W8A8 动态量化融合：本场景 routed limit0、R288；保留供 W2 LoRA 使用的浮点激活。无 clipping 的生产边界为 R≤288，正 clipping 为 R≤384。
- 显式 V1 DSpark 跳过未使用的普通 MTP stash：省去每 rank 512MiB 常驻 buffer 和专用写入/通信；DSpark aux hidden、draft 权重及投机流程保留。普通 MTP、非 speculative、V2 和自动 runner 仍保留原 buffer。
- 窄范围 DSpark5 target eager Prefill 使用 Cube W13 双 A：gather/mask＋现有 Cube BMM（FP32 输出）＋scatter。Decode、draft、profile/dummy 和其它不匹配场景走原路径；没有新增 Decode Cube kernel。
- Cube gather/mask arena 复用：每 rank 精确减少112MiB，full-alias where 未观察到 clone。首次未复用版本 KV126909<128000，未跑分；修复后实测 KV128277、10.32GiB，保留原128k与0.93参数并通过正式长输入测试。
- 编译缓存隔离：routing、activation、pair、MTP 四开关，以及 speculative method、runner 与派生 DSpark skip 条件均纳入对应缓存因子；更换配置须重新启动 engine/capture。

末5个同形状 target 图 T48/R288 的 rank0 profile：每图43次 pair 共1.899ms，W2原单 shrink43次共2.854ms，B expand129次共4.052ms；routed BGMV 总13.942→8.805ms；EP恢复5.914→0.492ms；索引1.782→0.322ms；activation＋quant1.263→0.839ms；两次完整512MiB MTP TensorMove→0。间隔中 draftT40/R240、3层、无LoRA primitive，均值40.769→37.687ms。算子总和可能重叠，不能相加为 critical-path 节省。Cube 在8个rank全部43层均有真实 prepare_accept/pair_submitted，具体 BMM trace 见 profile报告。

## 验证与复现

安装后261项相关单元检查通过；arena 修复后59项安装后针对性检查通过。非零真机 pair/activation 对旧算子逐位一致，包含 task_queue1、npugraph_ex fullgraph、NPUGraph 变化输入/权重/索引回放及 inactive/alias/blocked/Meta 边界。Cube 新旧arena五组非零多波/跨层/路由变化场景逐位一致；对 legacy vector BGMV 的最大绝对差约1.64e−7，属于独立容差验证。

启动脚本（在一个终端前台运行）：

```sh
docker exec ds-ltc-v25-ep sh {experiment_root}/ds_16_v25_optimized.sh
```

服务 ready 后在另一终端执行 warmup，然后完整场景：

```sh
docker exec ds-ltc-v25-ep python {experiment_root}/bench/benchmark_dspark_lora.py --mode smoke --seed 2026100701 --output-dir {experiment_root}/bench/reproduce-smoke
docker exec ds-ltc-v25-ep python {experiment_root}/bench/benchmark_dspark_lora.py --mode full --seed 2026100702 --output-dir {experiment_root}/bench/reproduce-full
```

同一 live engine 不重复 full seed，以免 prefix caching 使比较失效；重复正式对照使用 fresh engine 或新的独立 seed。当前测试服务已关闭。容器重建时需核对模型及 adapter 的 NFS 子挂载可见性，修复记录为 [mount-restoration.json](mount-restoration.json)。

- [完整设计与全部限制](../../design.md)
- [正式基线结果](bench/baseline-full/result.json) / [优化结果](bench/optimized-full/result.json)
- [参数与性能比较](bench/optimized-comparison.json) / [输出差异记录](bench/optimized-output-diagnostic.json)
- [安装源码/库哈希与备份](runtime-build/installed-source-binaries-manifest.json)
- [生产入口验证](analysis/production-binding-validation.json) / [Cube arena 验证](analysis/cube-scratch-reuse-validation.json)
- [后续 decode 候选分析](analysis/next-decode-candidates.md)：W2 H2048 单 A 跨核步进行分配、共享 gate/up A 融合尚未接入；需另外构建和实测。
