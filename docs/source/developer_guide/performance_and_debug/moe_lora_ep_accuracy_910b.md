# W8A8 MoE LoRA：单流 EP 精度对照（910B）

## 目的与范围

后续隔离变更：`MOE_LORA_PREFILL_CLIPPED_SWIGLU_ENABLED` 已默认关闭，
DSV4 MoE LoRA 的 EP/非 EP prefill 均回退到显式 clamp + cat + `npu_swiglu`，decode 不变。
下文真实权重分数来自 `cd9a79b2c` 的旧开关状态（当时 prefill clipped-SwiGLU 开启），
不是关闭该算子后的精度结果；该开关变更尚未重新进行整模型精度评测，运行中服务也未自动重启。
开关变更的 52 项量化 MoE LoRA UT 已通过，覆盖默认关闭与 decode 路径不变；
在远端独立进程测试源码快照，未覆盖安装目录，日志为
`ltc-v25:/tmp/lora-disable-clipped-XZ4slE/pytest.log`。

先比较非 EP recover AllGather + BGMV 与 EP AllGather + BGMV，精度问题明确后再优化性能。
本记录使用 DeepSeek-V4-Flash-0731 W8A8、8 张 910B3、v1 runner；不启用 MoE LoRA 双流、
DSV4 DSA overlap、shared-expert overlap、DSpark、单 LoRA GMM 或 composite GMM 快速路径。

`lora1` 是零增量 adapter：完整检查 70,932 个 tensor，35,466 个 A 非零、所有 B 均为零。
因此其数学增量为零，但仍经过普通 LoRA 计算路径，没有按零权重特殊跳过。
`news2026` 使用真实 rank-32 社区权重，故服务的 `max-lora-rank` 必须为 32。

当前 DSA 实现在 compressor 被 LoRA 包装时保留基座引用。这轮 `news2026` 分数只用于
同一实现的 EP/非 EP 对照，不代表社区 adapter 的完整 target-module 精度，也不是官方 benchmark 分数。

## 本轮发现与修复

1. EP 专家已经按完整 local expert 放置（MoE TP=1），上游却拒绝 EP 与 fully-sharded LoRA 同时使用。
   仅在专家层分配权重时复制 LoRA 配置并关闭该层的 rank 分片；dense/shared-expert 仍保持原 fully-sharded 设置。
2. v1 调度器的 eager fallback 丢失 `has_lora/num_active_loras`，使 prefill 误走无 MoE LoRA 路径。
   现在在 eager 返回前恢复实际调度状态；decode 图键及其补齐后的 LoRA count 不变。
3. DSV4 配置的 `swiglu_limit=10` 没有传入 routed experts，导致裁剪语义与 checkpoint 配置不符。
   现在 EP/非 EP 同时透传该参数，保留非 EP 的 recover + BGMV 实现，不混用修复前后的基线。
4. 融合 GMM 的裁剪 Float 属性被按 double 传递/读取，且 NZ 接口的 `DFX_IN` 缓存键漏掉 `limited`。
   修复 Float 转换、tiling 读取和执行器缓存键；仅修改 Python 或仅重编译 torch binding 不足以生效，
   必须同时更新 custom OPP 的 `libcust_opapi.so` 与 `libcust_opmaster_rt2.0.so`。

裁剪反例：固定投影值 gate=up=20，连续使用相同形状切换 limit，
旧版本均返回 400；完整修复后的真实 NPU 输出如下。

| limit | 期望值 | NPU 反量化输出 |
| --- | ---: | ---: |
| 0 | 400 | 400 |
| 5 | 24.832679 | 24.832678 |
| 10 | 99.995458 | 99.995461 |
| 30 | 400 | 400 |

新回归在旧库上 limit=5/10 失败，limit=0/30 通过，能够检测裁剪失效和跨参数缓存误复用。

## 对照方法与历史数据的限制

- 同一批 120 道已有 TRUE/FALSE 题；每题依次请求 ds、lora1、ds_repeat、news2026。
- temperature=0、seed=0、关闭 thinking 与 prefix caching；无效答案计错，保存逐题内容和 top-20 logprobs。
- 优先串行排除动态批次组成差异；额外用固定中英文/数学/代码提示生成 64 token 检查 decode。
- 精度比较同时记录首 token、整段序列、共同前缀概率差；首 token 概率不能代表整个 decode。
- 两侧使用相同确定性通信配置；这些设置仅用于精度诊断，不作为性能结论。

修复 eager 状态之前，确定性首 token 得分为：

| 模式 | ds | lora1 | ds 重复 | news2026 |
| --- | ---: | ---: | ---: | ---: |
| 非 EP | 71/120 | 71/120 | 71/120 | 71/120 |
| EP | 71/120 | 71/120 | 71/120 | 70/120 |

**这些是旧路径诊断数据，不是有效的 MoE LoRA 精度基线。**
当时 prefill 的 MoE LoRA 可能被漏算；ds/lora1 的首 token 概率完全一致不能证明适配正确。
旧 EP 串行 decode 中 ds 重复 4/4 序列一致、空 lora1 仅 2/4 一致，
正是继续定位该问题的线索。news2026 的旧输出差异也不能证明所有 target modules 都参与。

## 本轮完整修复后的真实权重结果

2026-09-06，双方各完成 480 次首 token 请求和 16 次 64-token 请求，全部返回有效响应。
相同 120 道题的结果如下；这是已有题集的回归分数，不是官方 news2026 benchmark 分数。

| 请求 | 非 EP | EP | EP/非 EP 首 token 一致 |
| --- | ---: | ---: | ---: |
| ds（无活动 adapter，融合基座路径） | 76/120（63.33%） | 71/120（59.17%） | 113/120 |
| lora1（零增量，普通 BGMV） | 71/120（59.17%） | 71/120（59.17%） | 112/120 |
| news2026（非零，部分 target 覆盖） | 74/120（61.67%） | 74/120（61.67%） | 112/120 |

用户指定的非 EP recover AllGather + BGMV 基线对应 `lora1` 行，不能用融合 `ds` 行替代。
两个 adapter 的总分虽然相同，但各有 8 道题答案变化，均为 4 道由对变错、4 道由错变对。
因此不能据此宣称 EP 数值等价或精度问题全部解决。融合 ds 在 EP 下少答对 5 题，
说明 EP 基座本身也需要继续定位，而不只是 LoRA routing。

同模式内，ds 重复请求在双方都是 120/120 首 token 及概率完全一致；
ds 与零 lora1 则分别只有非 EP 111/120、EP 114/120 首 token 一致。

64-token 连续生成检查：

| 比较 | 非 EP 内完整序列一致 | EP 内完整序列一致 | EP/非 EP 完整序列一致 |
| --- | ---: | ---: | ---: |
| ds 与自身重复 | 4/4 | 4/4 | ds 为 1/4 |
| ds 与零 lora1 | 2/4 | 1/4 | lora1 为 1/4 |
| news2026 | 不做重复对照 | 不做重复对照 | 3/4 |

这些只有 4 个短提示，且输出截断在 64 token；它们检查数值/序列稳定性，不代表完整回答的任务精度。
非 EP 的 17×23 均回答 391，但格式不同；这类措辞变化也会被严格序列检查记为不一致。

**当前结论：基线已重建、四类明确代码错误已修复，但精度验收未通过，不进入性能优化。**
没有针对零 B 权重跳过 LoRA，也没有把相同总分作为通过条件。

## 剩余数值问题的隔离证据

- 零 BGMV 增量在同一投影边界上逐元素为零，64 组 dispatch/routing oracle 检查通过；
  这只覆盖已测算子输入，不等于证明整模型所有权重加载和计算均正确。
- 基座融合 GMM/SwiGLU/quant 保留内部 FP32 中间值；普通 LoRA 路径先输出 BF16 的 W13，
  注入 BGMV 后再激活/量化，两者并非同一舍入边界。
  使用本轮最终算子，128 行、limit=10 的固定随机输入中，262,144 个量化元素有 3,078 个不同，
  最大差为 2 个 INT8 刻度，scale 最大绝对差为 0.00397259。
  该反例证明路径差异存在，但尚不能把整模型所有差异都归因于它。
- 尝试 INT8 GMM 直接输出 FP32 被当前 CANN 拒绝（161002，量化输出仅支持
  int8/int32/float16/bfloat16）。不能仅改 `output_dtype` 来实现 FP32 对齐。
- EP 与 TP 还改变 W2 动态量化的分片范围和求和顺序。后续应以相同输入/共同生成前缀，
  定位第一个发生偏差的层，分别比较 W13、激活量化、W2 和 combine 的结果；
  在模型级证据建立之前，不将这些差异定性为全部可接受的浮点误差。

## 通信重复性隔离

通过 vLLM 自身 PyHcclCommunicator，固定 BF16 输入、改变各 rank 到达顺序，各重复 20 次：

| 输入形状 | 默认 AIV 最大重复绝对差 | HCCL 确定性最大重复绝对差 |
| --- | ---: | ---: |
| 8 × 4096 | 0.125 | 0 |
| 32 × 4096 | 0.125 | 0 |
| 128 × 4096 | 0.125 | 0 |
| 2048 × 4096 | 0.1875 | 0 |

这证明通信是一个独立波动来源，不代表模型数值等价。参考
[项目确定性说明](../../faqs.md#15-how-to-generate-deterministic-results-when-using-vllm-ascend)
及 [CANN 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_DETERMINISTIC.md)。
确定性也不等于批次不变性；并发请求仍须与 ds 自身重复对照比较。

## 验证状态与证据

- 局部 EP 配置、量化 MoE 与 LoRA UT：99 项通过。
- v1 eager/graph 元数据 UT：2 项通过（涵盖零/单/多 adapter）。
- DSV4 裁剪参数透传 UT：3 项通过。
- 精度比较脚本 CPU 回归：9 项通过。
- 零增量 NPU BGMV：6 项通过，rank16/32 × 1/8/128 行；
  active zero-B 与同一基座边界逐元素相等，nonzero-B 对照改变输出，无 adapter 行不变。
- 实际 NPU dispatch routing：64 组通过，覆盖 8 个 EP rank、1/2/8/16/128/512/513/2048 token、
  无 adapter、禁用 adapter、空本地专家及融合/回退边界，与 CPU oracle 和 recover 逐元素相等。
- 最终算子/配置合并回归：77 项通过（routing 64 + 零 BGMV 6 + 模型参数 3 + 裁剪 4）。
  裁剪回归每个用例内连续切换 limit，避免 pytest 多进程拆分参数用例后漏检执行器缓存键。
- clamp + SwiGLU 与 clipped SwiGLU 在 1/2/8/16/128 行样本上的输出、量化值及 scale 完全一致；
  不以更换这两个算子掩盖真实问题。
- max-model-len=256000 是启动配置，不代表已验证 256k 输入精度；本轮不做吞吐性能验收。
- 本地专项 ruff 和 git diff --check 通过；完整 format.sh ci 已尝试，但因本机缺少 pre-commit 未能运行检查。

本轮直接使用真实权重，未采用 dummy 作为验收依据。FULL_DECODE_ONLY 图模式与 EP AllGather
完成实际请求；flashcomm1、MTP/DSpark、DSA overlap、shared-expert overlap 和 MoE LoRA 双流
按精度隔离范围关闭。未验证 AlltoAll、并发 8 请求精度、长上下文精度和吞吐。

证据目录：`ltc-v25:/home/ltc_vllm/ep_accuracy_20260906/`。
旧结果保留在 `*_deterministic_*.json`；`noep_statefix` 因发现共同裁剪问题被主动停止，不用于基线。
算子证据见 `fused_clamp_cachefixed.log`、`clamp_regression_before_retry.log`、
`zero_delta_model_config_tests.log`、`state_unit_tests_retry.log`、`routing_check_retry.log`。
正式新对照使用 `*_precisionfix_*.json`；产物哈希记录在 `precisionfix_artifact_hashes.json`。
最终回归见 `final_unit_tests.log`、`final_state_tests.log`、`final_operator_tests_cache_sequence.log`；
数值边界见 `mlp_boundaries_precisionfix.log`、`fp32_boundary_support.log`。
所有临时 printf 诊断已移除后才开始新模型对照；旧二进制已备份在 `binary_backup/`。
远端运行目录仍是旧分支上的已同步工作区，不以其旧 git HEAD 冒充本轮源码版本。

## 复现命令

以下在远端 `/workspace` 执行。EP 模式额外增加 `--enable-expert-parallel`，其余参数保持一致。
确定性对照双方都设置下面四个开关；保留原通信对照时不要仅对一侧设置。

```bash
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=512
export HCCL_DETERMINISTIC=true LCCL_DETERMINISTIC=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0

vllm serve /data/models/DeepSeek-V4-Flash-0731-w8a8 \
  --safetensors-load-strategy prefetch --served-model-name ds --port 8888 \
  --max-model-len 256000 --max-num-batched-tokens 16384 --max-num-seqs 8 \
  --gpu-memory-utilization 0.93 --tensor-parallel-size 8 --data-parallel-size 1 \
  --quantization ascend --block-size 32 --enable-chunked-prefill \
  --no-enable-prefix-caching --async-scheduling \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-lora --fully-sharded-loras --max-lora-rank 32 --max-loras 3 \
  --lora-modules \
    lora1=/data/models/DeepSeek-V4-Flash-0731-lora-rank16 \
    news2026=/data/models/deepseek-v4-flash-news2026-lora \
  --all2all-backend allgather_reducescatter \
  --additional-config '{"enable_moe_lora_dual_stream":false,"enable_flashcomm1":false,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false,"multistream_dsv4_dsa_overlap":false}'
```

评测脚本不依赖 NPU 或 vLLM 客户端包：

```bash
python benchmarks/scripts/check_ep_lora_accuracy.py \
  --questions-from-report /tmp/noep_dsa_single_empty_lora_serial120.json \
  --label noep --workers 1 --out noep.json
# 切换为相同配置的 EP 服务后，使用 --label ep --out ep.json 再测。
python benchmarks/scripts/check_ep_lora_accuracy.py \
  --compare noep.json ep.json --out comparison.json
```

HF mapper 仅保留在远端社区权重加载环境，本次修改不把 mapper 加入 Git 提交。
