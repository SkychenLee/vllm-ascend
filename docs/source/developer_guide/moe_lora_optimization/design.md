# AllGather W8A8 MoE＋LoRA 优化设计与算子使用约束

更新日期：2026-10-07，Asia/Shanghai。此前实验对象：ds-ltc-v25-ep 容器、fully_sharede_lora_ok 分支及当时已安装候选。本文从 2026-10-06 设计扩展而来，适用于 157_moe_ep_and_fully_sharede_lora 发布副本。旧实验的证据保留在独立目录；公开材料范围与路径占位符见 [发布说明](README.md)。

本文汇总此前接入和本轮完成的全部相关优化，逐项说明融合方法、使用条件、原生接口限制、回退/报错、图与缓存约束，以及已有验证。这里的“已实现”指代码已接入，不表示全部模型、形状、硬件或 speculative 配置均已验证。

快速定位：[优化总表](#33-全部优化状态) · [路由与索引](#5-路由与索引融合的全部限制) · [INT8融合](#6-非ep-fully-sharded-int8融合族) · [激活融合](#7-普通ep-swigluint8动态量化) · [Cube](#8-cube-w13-a-prefill) · [MTP与DSpark](#11-deepseek-v4-普通mtp-stash-与-dspark) · [配置](#12-配置使用矩阵与示例) · [验证](#13-精度图与性能证据)。

## 1. 最新 single-adapter 状态与此前路径快照

最新 single-adapter 实现合约、可移植配置与证据见 [single-adapter 专项文档](../moe_lora_single_adapter.md)。2026-10-07 最终 fully-sharded `VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER=0/1` 对照，OFF/ON 输出吞吐为 **316.02/378.05 tok/s（+19.63%）**，平均 TTFT 为 7.497/7.512 s，平均 TPOT 为 19.120/16.313 ms，DSpark acceptance 为 75.30%/89.33%。两条件各一次测量，acceptance 也发生变化；这是条件性端到端结果，不建立纯 kernel 因果收益或模型质量等价。

**fully-sharded Cube 已新增独立实验路径。** [专项文档](../moe_lora_fully_sharded_cube.md)说明 W13/W2 A 投影的分组矩阵乘、内存边界、使用条件与验证。下文第 8 节的既有 Cube 仍仅支持受限普通 TP＋EP LoRA。其余路径描述与第 13–16 节的实验、验收状态保留此前发布快照；其中“本轮/当前”指当时的 EP 优化实验，最新 fully-sharded 状态以相应专项文档为准。

当前有两条主计算路径和一个受限 Prefill 替换路径：

- **TP＋EP、非 fully-sharded LoRA**：基座 W8A8 INT8 GMM，LoRA A/B 浮点。本轮接入 AllGather EP 路由融合、W13/W2 indices 复用、SwiGLU＋量化、H4096/r16 paired Vector A、Decode graph 和 DeepSeek V4 普通 MTP stash 优化。
- **非 EP、TP fully-sharded LoRA**：基座与 LoRA 共用 INT8 激活，LoRA A 融合反量化；保留 paired A、旧 W13 B＋激活＋量化、grouped 大行数和 large recover。这些 INT8 LoRA 融合不能通过开关搬到普通 EP。
- **Cube Prefill**：现有 gather/mask＋PyTorch BMM＋scatter，仅替换受限 TP8＋EP8、BF16/r16 的 W13 双 A。此次窄范围新增显式 V1 DSpark5 target eager Prefill 支持；没有新增 Decode Cube kernel。

**EP＋fully_sharded_loras=True 仍不支持。** 新 activation 生产行数限制为 unclipped R≤288、positive-clipped R≤384。普通 MTP stash skip 已实现，仅在 optimization flag=1、显式 V1、normalized DSpark 时省 allocation/copy/专用 gather；V2、runner 自动选择(None)、ordinary MTP 和 non-spec 继续保留 buffer。Cube DSpark5 新 guard 不改变 draft eager 行为和 target Decode graph。

真实模型/LoRA路径已恢复。已完成同一启动配置下 **8192 输入＋4096 输出、8 并发、max_num_batched_tokens=16384、DSpark5** 正式基线：8/8 成功、32768 输出 tokens、259.432696 output tok/s。正式 optimized 8/8成功，278.559021 output tok/s，相对基线+7.37%；两组输出token序列不相同、DSpark接受率85.87%→82.72%，质量/精度等价尚未验证。算子 micro 不能代替整网收益。非 EP 的相同场景仍无本轮可比吞吐。

| 项目 | 当前快照 |
| --- | --- |
| 宿主仓库 | {repo_root} |
| 容器安装仓库 | {installed_repo}；上游 {upstream_repo} |
| host 分支/历史基线 | fully_sharede_lora_ok / 658005654；工作区实现与 installed manifest 才是完整快照 |
| 安装版本 | editable vLLM 0.25.1、vLLM Ascend 0.1.dev1+g658005654 |
| 运行环境 | Python3.12、aarch64、Torch2.10.0+cpu、torch_npu2.10.0.post4 |
| 构建 | fresh Release、SOC_VERSION=Ascend910_9391、CANN toolkit latest |
| 正式运行设备 | 用户指定物理 8–15；单卡算子实验在停服后分配的 12 等卡 |
| 启动脚本 | {site_launch_script} |
| 模型 | {model_path} |
| 模型关键维度 | BF16、43 target layers、HC multiplier4、H4096/I2048/E_global256/K6 |
| 本轮实际服务配置 | TP8/EP8、DP1、显式 V1、DSpark5、LoRA r16/3slots、FULL_DECODE_ONLY/capture≤48 |

Torch 的 +cpu 是 wheel 标记，NPU 执行来自 torch_npu。当前 fresh native 与 11 个生产源文件已安装，manifest 记录 source/binary hash 与回滚位置，安装动作未使用 NPU：

- [本轮部署与备份清单](evidence/2026-10-07/runtime-build/installed-source-binaries-manifest.json)。
- binding SHA256：`f7355731daa346a7fd7ff87bdaaddbceaea402fb391bcd7339644e8a45c7a674`。
- 两个 kernel 库位置 SHA256 均为 `ff5266af525ae36ea604856c9ddc81fd9bda7bbffc50560a9b441a1f2a9c6894`；[安装核对](evidence/2026-10-07/runtime-build/install-libraries.json)。
- 首次installed Python/CPU UT 261 passed；[记录](evidence/2026-10-07/analysis/installed-unit-tests.log)、[执行清单](evidence/2026-10-07/analysis/installed-unit-tests.json)。随后arena复用已原子同步installed同SHA，新增focused installed 59 UT PASS；[记录](evidence/2026-10-07/analysis/installed-cube-reuse-unit-tests.log)、[清单](evidence/2026-10-07/analysis/installed-cube-reuse-unit-tests.json)。CPU/Fake记录与第13.6节installed NPU数值/图验证分列。

下文凡“历史”证据均指 2026-10-06 或更早独立实验；本轮 source 接入、fresh installation、原生 POD probe、正式 Torch NPU graph 和有权重整网分别标明，不互相替代。

## 2. 维度与共用合约

| 符号 | 含义 |
| --- | --- |
| T | 当前阶段实际 token 行数，可能包括 graph padding、投机验证 token |
| K / M | top-k / 原始 token-expert pair 数 M=T×K |
| R | dispatch 容量行数，可能含无效尾行；不是实际本地有效行数 |
| H / I | 专家 hidden / 本地 intermediate 维度 |
| r / r_local | 完整 / TP切分后的 LoRA rank |
| E_local / E_global | 本地专家数 / 全局 expert_map 长度 |
| L / G | adapter槽位数 / L×E_local 个adapter-expert group |
| N_slots | 非EP recover的token_lora_indices长度，可能是buffer容量或短token映射表，不是adapter数 |
| D / Q_D | BF16或FP16 / 舍入并存储为D |
| q / s | INT8激活 / FP32每行反量化scale |

当前 EP 实测 H4096/I2048/r16/E_local32/E_global256/K6。**8并发不代表每次 T=8。** 当前 DSpark5 的 target 验证 query_len=6，8 请求 steady graph 为 T48/M288/R288，prepared hidden `[48,4096]`、top-k `[48,6]`、slots `[48]`；dispatch 容量 `[288,4096]`，activation `[288,2048]`。本地有效行数通常小于容量，不能用 `sum(expert_tokens)` 替代 R 来判定融合。

真实 profile 已确认 routed activation **limit0、SwiGLU dim=-1**；模型 config 的 limit10 作用于 shared experts，不能推导 routed 也 clip10。本轮 noClip R288 guard 正好覆盖该 target；positive clip R384 是另一测量区域。相邻 eager draft 观测 T40/R240、3 draft layers，未观测 BGMV/SGMV。图 padding、低并发、投机 acceptance 会产生其他 T/R，须按实际 shape 判断。假如其他 speculative 配置产生 T64/R384，EP routing/pair 可仍命中，而 noClip activation 会回旧链。

普通动态对称量化意图为 scale=amax(abs(activated))/127、q=round_even(activated×127/amax)，zero row 输出q0/scale0。实数公式等价不保证低精度/INT8 bitwise相同，融合要保留原dtype舍入边界、CANN算术次序、top-k权重和activation alpha/beta。

### 2.1 硬件、注册与物理格式

自定义接口属于 torch.ops._C_ascend。路由、INT8与新activation提供NPU和Meta实现；旧浮点void shrink并非全部有独立Meta注册。没有原生CPU kernel；Python helper/上游 torchops 的CPU或fallback行为不能当作native CPU实现。

direct kernels 受 VLLM_ENABLE_ATB_AND_DIRECT_KERNELS 控制。当前主CMake对 ascend310p.*、ascend950 不编译注册这组direct库。已测环境为当前Ascend910_9391/A3，不能从源码存在推导310P/A5支持。

| 算子族 | 实际storage合约 |
| --- | --- |
| 新EP routing、activation_quant、pair 特化 | 同NPU、contiguous；runtime C++明确只接受format0/2 |
| recover、combined、旧INT8/grouped | 同NPU、逻辑contiguous；没有统一physical-format guard，调用方保证普通base storage |
| 旧浮点BGMV/SGMV | caller保证正确dtype/shape、连续base storage和合法slot；旧binding检查不完整 |
| Cube组合 | 普通连续BF16矩阵、对应weight/stream布局；没有新EP/pair核那种统一descriptor保护 |

**is_contiguous() 不足以证明raw storage为ND。** 新EP/activation/pair 的 eager Python先检查格式，否则旧路径；compile时跳过FakeTensor descriptor查询，真实执行仍由C++拒绝blocked。同一个用ND编译的callable后换NZ也报错，不自动换图或fallback。

允许format0是代码合约；Oct6 EP/activation请求0的fixture被规范化为2，不能将其当作这两族的实际format0测试。本轮正式pair验证记录x/indices为2、两组A为0，pair的实际format0权重已测。VLLM_ASCEND_ENABLE_NZ不能保证所有metadata/activation已经是base storage。

原生kernel使用当前NPU stream，host主要依据静态metadata/设备资源决策；不为路由payload读取CPU值。Cube counts回读是明确例外，见第8节。

## 3. 计算流程和优化总表

### 3.1 普通 TP＋EP AllGather

~~~mermaid
flowchart TD
    A["浮点hidden ＋对齐adapter slots"] --> B["AllGather / InitRouting：本地expert容量行"]
    B --> C["EP metadata恢复：local expert ＋ slot"]
    C --> D["同forward生成一次combined indices"]
    B --> E["动态量化 ＋ base INT8 GMM1"]
    E --> F["W13浮点LoRA A/B → gate/up"]
    D --> F
    F --> G["SwiGLU ＋ optional clipping"]
    G --> H["浮点activated → W2 LoRA"]
    G --> I["INT8 activated ＋scale → base GMM2"]
    D --> H
    H --> J["相加、combine、最终TP reduce"]
    I --> J
~~~

新activation_quant融合G的激活与量化，同时生成H/I的两种输入，**保留浮点W2语义**。Cube符合条件时只替换W13 A，W13 B、W2、base GMM不被融合进Cube。

### 3.2 非 EP fully-sharded

~~~text
INT8 x + scale
  ├─ base INT8 GMM1 → 浮点gate/up
  └─ paired INT8 A shrink（融合反量化）→ FP32[R,2*r_local]
      → TP rank通信，保留rank-major paired layout
      → 双B + base add + clip + SwiGLU + optional topk + quant
      → INT8 activated + scale
           ├─ base GMM2
           └─ W2 INT8 A shrink → TP通信 → B add
~~~

该模式W2本来消费INT8＋scale，旧融合不返回浮点activated；直接搬到普通EP会改变W2 LoRA输入语义。

### 3.3 全部优化状态

| 项目 | 融合/减少的工作 | 适用范围 | 状态与控制 |
| --- | --- | --- | --- |
| AllGather EP兼容及默认backend | hidden/slots对齐、固定通信结构、恢复本地expert | LoRA-enabled TP＋EP | 已接入；默认allgather，非法配置启动报错 |
| 无效容量行清零 | 防止未写尾行/NaN污染 | AllGather EP | 正确性基础；不单独宣称加速 |
| recover small/sort/large | 通用abs/sort/gather替换 | 非EP fully-sharded | 既有自动分派，无env gate |
| combined_indices native | mask与adapter/expert编码 | 多路径helper | 既有融合；FUSED_ROUTING默认0 |
| single-adapter metadata | recover＋mask＋combined-index融合，兼容混合adapter | 非EP＋fully-sharded AllGather，受限small/large rows | 已接入；SINGLE_ADAPTER默认0，[合约与证据](../moe_lora_single_adapter.md) |
| fully-sharded Cube A | W13双A合并GMM，W2 A GMM，FP32投影 | A3、非EP TP8、单adapter、eager prefill、R16384..98304 | 已接入；FULLY_SHARDED_CUBE默认0，[合约与证据](../moe_lora_fully_sharded_cube.md) |
| W13/W2 indices复用 | 同forward少做一次编码 | 普通AllGather EP，同A前两维 | 已接入；flag0也复用framework结果 |
| EP routing native | sort/mask/map/slot恢复单kernel | M≤384/map≤512 | 已接入；FUSED_ROUTING默认0 |
| INT8 A shrink及paired W13 A | 反量化、FP32 reduction、双A共读输入/rank-major通信 | 非EP fully-sharded | 既有核心native自动布局路径 |
| 旧W13 B/activation/quant | 双B、base add、clip、SwiGLU、topk、quant | 非EP fully-sharded、I≤256 | 既有fusion/shape分派 |
| grouped大行数A/B/W2 | 组内weight缓存、宽B列tile | 非EP fully-sharded | 保留实验；GROUPED_PREFILL默认0，历史目标W2更慢 |
| Cube W13 A Prefill | 双A pack、64行tile、Cube BMM | 原non-spec或显式V1 DSpark5 target eager | 新窄guard已接入；W13_CUBE_PREFILL默认0 |
| activation_quant | clip/SwiGLU/amax/quant、保留float W2 | 普通EP、I2048、noClip≤288/clip≤384 | 扩大生产guard已接入；FUSED_ACTIVATION默认0 |
| floating W13 A pair | gate/up共享X及Cast、一次launch、跨核row步进 | 普通非fully-sharded、H4096/r16/M≤384 | 新特化已接入；FUSED_SHRINK_PAIR默认0，其他native shape仍旧双launch |
| LoRA Decode graph | bucket、定长mapping、真实Prefill排除 | 符合descriptor的请求 | graph配置控制，fresh启动重新capture |
| compiler/AOT cache隔离 | 图算子与MTP allocation分支隔离 | 编译/捕获 | 6个动态flags（含SINGLE_ADAPTER、FULLY_SHARDED_CUBE）＋MTP derived skip＋normalized method |
| MTP row writer | 避免已观测whole-buffer copies | 保留普通buffer的DeepSeek V4路径 | COPY默认0，flag1 shape适配时scatter |
| V1 DSpark skip普通MTP | allocation/copy/专用TP gather | 显式V1＋normalizedDSpark＋COPY1 | 已接入；V2/None/nonSpec/ordinaryMTP严格保留 |
| whole-weight BF16 cache | 缓解weight读开销 | 独立prototype | 未接入，无production开关 |
| device-only Decode Cube | 固定metadata/gather/BMM/scatter融合 | 未来候选 | 未实现，无production schema/开关 |

## 4. AllGather EP 兼容与错误边界

additional_config["moe_lora_ep_backend"]默认精确字符串allgather，只接受allgather/alltoall；不是通用all2all_backend，也不存在gallgather取值。

LoRA-enabled engine且EP size>1时固定选择此backend，包括base-only batch、profile、dummy、capture；不随活跃tokens切MC2。非EP/EP size1仍AllGather，无LoRA配置的engine保留原硬件/阈值选择。**不是全仓非LoRA MoE都改成AllGather默认。**

| 必须条件 | 不满足时 |
| --- | --- |
| DP=PCP=DCP=1 | 启动ValueError，不自动换AlltoAll |
| expert placement=linear | mix/其他placement不支持 |
| 无enable_eplb/dynamic_eplb/redundant experts | 不支持上游与additional冗余布局 |
| 无expert_map_path/expert_map_record_path | 当前AllGather EP LoRA不支持 |
| 非fully_sharded_loras | EP＋fully-sharded被上游assert拒绝 |
| 禁用VLLM_ASCEND_ENABLE_FUSED_MC2 | 当前LoRA不能插入MC2/FusedMC2单算子 |
| 活跃量化MoE LoRA为W8A8 | registry其他QuantType报NotImplementedError |
| W1/W2及其scale lists各一tensor，scale存在 | 不支持动态EPLB逐expert list |
| 无weight scale-bias / antiquant offset | 明确不支持 |
| 普通EP输入BF16/FP16浮点 | dispatch提前量化关闭；INT8/已有dynamic_scale报错 |

PP1是Cube额外限制，不是上述基本配置检查的统一条件。想使用旧backend需显式设置alltoall，而非期待错误后自动降级。

prepare对hidden和prepared_slots执行相同truncate/pad/split/gather/unpad，pad slot=-1；不覆盖dense Punica mapping。EP本地专家保存完整rank，不能再次按全局TP切分。shared experts仍用dense LoRA。AllGather在routed＋shared最终合并后执行TP reduce，避免重复reduce。

InitRouting提供连续global专家范围和global→local map。expanded负值是exclusion，不能abs。无效容量行在dispatch、GMM1后、combine前masked_fill成0；NaN×0不能替代。空本地dispatch跳过空kernel但保留collectives。scoped context在finally清理上一batch路由状态。

源码：[配置](../../../../vllm_ascend/ascend_config.py)、[通信](../../../../vllm_ascend/ascend_forward_context.py)、[包装器](../../../../vllm_ascend/lora/fused_moe.py)、[prepare](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py)、[dispatch](../../../../vllm_ascend/ops/fused_moe/token_dispatcher.py)、[量化链](../../../../vllm_ascend/lora/quant_moe.py)。

## 5. 路由与索引融合的全部限制

### 5.1 非EP moe_lora_recover

~~~text
moe_lora_recover(expanded[M], topk[T,K], slots[N_slots], top_k)
    -> experts[M] INT64, slots_out[M] INT64
for source p:
    d = abs(expanded[p])
    experts[d] = int64(flat_topk[p])
    slots_out[d] = slots[min(p//K,N_slots-1)]
~~~

expanded/topk为INT32或INT64，slots为INT64，同NPU且连续；K>0、M=T×K，M非0时slots非空；M0不launch。Python对stride输入先contiguous。

**数据前提：abs(expanded)为[0,M)完整唯一排列。** 负值是带符号合法位置，-1可表示目的1，不是EP exclusion。expert/slot payload保留完整INT64，包括>2^53，不经FP32转换。

生产分派：context非空、fully_sharded is True、use_ep is False、正Python int K、正确维度/dtype、同NPU、schema存在、M≤2^24。use_ep=False是严格身份。**不受FUSED_ROUTING控制，不用于普通EP。**

| 分支顺序 | 实际条件 | 融合方法/资源 |
| --- | --- | --- |
| Large优先 | 8161≤M≤262144、UB≥114720B、vector cores可查询 | tile512，两launch：source scatter→destination pack；64B/row scratch，最多16MiB |
| Small | 未进Large，M+T≤1024，UB足够 | 单AIV/launch，全部metadata进UB，inverse scatter，整块写出 |
| Sort | Small未选，160≤M≤8160，K≤M、M%K=0，UB足够 | 单AIV/launch，有限FP32位置key排序，整数payload gather |
| framework | 超预算/资源不满足 | 保留abs＋argsort＋index_select，不无限增scratch |

Small UB=align32(M×index_bytes)+align32(M×expert_bytes)+align32(min(N_slots,ceil(M/K))×8)+2×align32(M×8)，另留4096B。成本是M+T≤1024，不能写成“所有≤1024行走Small”。

Sort UB=24×align32(M)+4096B，API上限8160；192KiB UB下实际最多8000，8001–8160回退。FP32只承载有界位置key；payload按两个uint32 gather，INT32专家符号扩展。nextafter-up reciprocal的token floor只在有界domain精确，不能推广任意整数除法。

Large每目的row独占64B scratch，只写有效16B的expert/slot，避免跨core相邻INT64写竞争；第二launch compact。scratch未整体初始化，**重复、缺洞、越界排列均不在语义保证内**；分支跳过越界不构成坏数据兼容。

native错误dtype/dim/device/contiguity是TORCH_CHECK；资源不足才内部fallback。没有统一native行数接受上限等于262144这一说法：超large预算还可走framework。Python的2^24是caller边界。

源码：[binding](../../../../csrc/moe_lora_recover.cpp)、[validation](../../../../csrc/moe_lora_recover_validation.h)、[small](../../../../csrc/kernels/moe_lora_recover_small.cpp)、[sort](../../../../csrc/kernels/moe_lora_recover_sort.cpp)、[large](../../../../csrc/kernels/moe_lora_recover_large.cpp)。

### 5.2 combined_indices及forward内复用

~~~text
moe_lora_combined_indices(experts[R], slots[R], enabled[L], E_local)
    -> combined[R] INT64
if 0<=s<L and enabled[s]!=0: combined = s*E_local + expert
else: combined = -1
~~~

| 条件 | 限制 |
| --- | --- |
| native输入 | experts INT32/64，slots INT64，enabled INT32；1D、同NPU、contiguous |
| 静态参数 | E_local>0、1≤L≤64；native没有单独R硬cap，R0不launch |
| Python选路 | FUSED_ROUTING=1、schema存在、R≤98304及上述条件 |
| payload | expert不做值域检查；INT64乘加二补码wrap；enabled非0均active |
| 无效slot | native超大正slot返回-1，旧fallback可能index error；producer必须-1或有效slot |

每core负责8行对齐区间，tile128，先读enabled到UB，输出INT64编码；无sort/D2H/top-k权重。缺schema、flag0或合法guard miss走原mask/where。

同forward复用：普通AllGather EP、非fully-sharded，W13/W2 A stack前两维adapter/expert相同，生成一次传给两侧。flag0也能复用framework结果。**不跨step/cache adapter indices**，graph replay必须重新读取变化mapping/enabled，热加载不能保留旧编号。

源码：[helper](../../../../vllm_ascend/lora/combined_indices.py)、[binding](../../../../csrc/moe_lora_combined_indices.cpp)、[kernel](../../../../csrc/kernels/moe_lora_combined_indices.cpp)、[复用主链](../../../../vllm_ascend/lora/quant_moe.py)。

### 5.3 EP metadata恢复

~~~text
moe_lora_ep_routing(expanded[M], topk[T,K], prepared_slots[T],
                    expert_map[E_global], R)
    -> local_experts[R] INT64, adapter_slots[R] INT64
~~~

native：M≤384、1≤E_global≤512、0≤R≤M、K>0；expanded/topk/map为INT32/64、slots INT64；shape匹配、同NPU、contiguous、实际format0/2。R0无launch，但仍检查基础metadata。Python R0更早返回，不能推导native接受任意坏metadata。

Python基础helper先验证形状、capacity及R≤2^24；fusion需FUSED_ROUTING=1、schema、上述小范围和format条件。大M/大map/旧库等合法miss回旧链；非法基础shape/capacity ValueError；直接native超界TORCH_CHECK。

单AIV单launch UB inverse scatter：

1. 输出初始-1；d=expanded[p]，d<0或d≥R跳过，**绝不abs sentinel**。
2. seen[d]=1；global expert必须在map范围且map[g]≥0，才写local expert和prepared_slots[p//K]。
3. 找seen目的前缀第一个hole，此行及以后全-1，复现旧sorted ordinal==destination；已seen但非法expert只使本行invalid。
4. 非负目的行必须唯一；**重复不支持，也不做duplicate检测**。INT64 slot完整保留。

替代旧valid mask→FP32有界key→argsort→gather→map→slot恢复→where。worst UB20992B，无Cube。旧fallback选择不保证CANN支持任意blocked integer Cast/Identity/TransData；compile遇blocked则runtime拒绝而非fallback。

源码：[恢复helper](../../../../vllm_ascend/lora/routing.py)、[binding](../../../../csrc/moe_lora_ep_routing.cpp)、[kernel](../../../../csrc/kernels/moe_lora_ep_routing.cpp)。

## 6. 非EP fully-sharded INT8融合族

### 6.1 A shrink、paired布局及输入合约

allgather_int8只在AllGather＋fully_sharded=True/use_ep=False。输入允许D且无scale，或INT8且FP32 scale同设备，scale形状为输入除末维或附加1；INT8还需明确D output_dtype。错误scale/dtype ValueError。

~~~text
z[i,a] = sum_h (float(q[i,h])*scale[i]) * float(A[group(i),a,h])
~~~

kernel内融合反量化和FP32 reduction，避免物化完整浮点X。inactive INT8行写0，可用empty输出；浮点shrink skip行，caller需zeros。

| schema | 输入/输出 | native限制 |
| --- | --- | --- |
| bgmv_shrink_int8 | q[R,H] INT8，A[G,r,H] D，ids[R] INT64，s[R] FP32；写y[R,r] FP32 | 0<r≤512，H/G正，R/H/G≤INT32_MAX |
| bgmv_shrink_int8_pair | A[2,G,r_local,H]；写y[R,2r_local] | 0<r_local≤512，其他同上 |
| bgmv_shrink_int8_pair_grouped | paired A，另有order[R] INT64 | H≤4096、H%64=0、2r_local×H≤16384 |

同NPU、contiguous、shape严格，y不与输入overlap。旧ops没有新EP那种physical-format guard。

paired A存储[2,L,E_local,r_local,H]，loader/reset使用连续view，非EP fully-sharded、W13两片、无shared-MoE-LoRA时建立；max_lora_rank必须被专家内部TP size整除。不逐forward pack，不热加载更换graph指针。

r_local<r时低rank结果all-gather dim0后view成[shards,R,2r_local]；本地rank等于完整rank时相应all-reduce。直接rank-major供B消费，省去最后维gather后的转换。无packed A则双独立shrink；普通EP不适用这一rank切分。

### 6.2 旧B/add/activation/quant融合

~~~text
moe_lora_expand_swiglu_quant(base, gate, up, Bg, Bu, ids, topk?, limit)
moe_lora_expand_swiglu_quant_pair(base, paired, Bg, Bu, ids, topk?, limit)
    -> INT8[R,I], FP32[R]
~~~

native输入base[R,2I] D，gate/up[R,r] FP32，Bg/Bu[G,I,r]同D，ids[R]INT64，optional topk[R]FP32。0<I≤8192、0<r≤512、G正、R/G≤INT32_MAX；limit finite且≥0。paired[shards,R,2r_local]FP32，shards×r_local=r，R>0时(2R-1)×r_local×4≤UINT32_MAX。

~~~text
gate = Q_D(base_gate + B_gate*z_gate)
up   = Q_D(base_up + B_up*z_up)
if limit>0:
    gate=Q_D(min(gate,limit))
    up=Q_D(clamp(up,-limit,+limit))
act=Q_D(silu(gate)*up)
if topk: act=Q_D(act*topk)
act → amax/quant → q,scale
~~~

只写q/scale，无adapter仍计算base激活。先FP32 round到INT32再转换INT8，避免双舍入。

**生产自动启用：** AllGather＋非EP fully-sharded、payload.fusion=True、activation严格silu、W13两片、r≥8且2次幂、I%32=0、I≤256、I×r≤8192。无独立env flag；guard miss原W13＋activation＋quant。当前EP I2048不命中。

W2使用同INT8＋scale做A，TP通信后B加回down_out；fully-sharded输出offset按tp_rank×B_local_width。core INT8/旧B在部分调用点没有hasattr fallback，schema缺失会lookup/error，必须安装匹配native库。

### 6.3 grouped大行数与W2列tile

GROUPED_PREFILL默认0，import时保存常量；运行时改env不生效。分派按静态行数，**没有显式is_prefill判断**。

| Python入口条件 | 要求 |
| --- | --- |
| 主分支 | 已进入旧fully-sharded非EP INT8 W13 fusion，有packed A |
| R/G | 8192≤R≤2^24、R≥8G、0<G<2^24 |
| A | INT8 NPU、0<H≤4096、H%64=0、2r_local×H≤16384 |
| B | r∈8/16/32、0<I≤256、I%32=0 |
| native | paired grouped A、paired grouped fusedB、grouped expand存在 |

只排序薄combined metadata，无效id以G排最后，FP32有界keys构造order，inverse=argsort(order.float())。**完整X不排序搬运。** grouped A按order间接读X，binding将薄scale做index_select(scale,order)保证配对；小paired恢复原序再TP通信。grouped B复用组内B，q/scale恢复原序后GMM2；W2复用routing。

~~~text
moe_lora_expand_swiglu_quant_pair_grouped(base, paired, Bg, Bu,
                                         ids, order, topk?, limit)
bgmv_expand_grouped(x, B, ids, order, y, offset)
bgmv_expand_grouped_tiled(x, B, ids, order, y, offset)
~~~

grouped fusedB的native I≤256、I%32=0、r∈8/16/32。grouped expand输入x[R,r]FP32、B[G,width,r]和y[R,D_out]同D；r∈8/16/32/64、0<width≤8192；width/offset/D_out均16对齐、offset≥0、offset+width≤D_out；rows/groups/D_out≤INT32_MAX，y不alias输入。

width×r≤8192选narrow，超出需tiled schema把UB B cache限制到8192元素；缺tiled则普通BGMV。仅input_scale非空的INT8 LoRA路径中，其他B rank用bounded torch fallback，tile_rows=max(1,2^21/(width×r))。order必须合法完整permutation，native只查metadata不核验全部值。

**历史目标grouped W2更慢，保持GROUPED_PREFILL=0作为推荐。** 精度通过不表示性能改善，窄/宽分组实验保留，不能列为当前整网收益。

源码：[grouped](../../../../vllm_ascend/lora/grouped_prefill.py)、[Punica](../../../../vllm_ascend/lora/punica_npu.py)、[bindings](../../../../csrc/moe_lora_int8.cpp)、[kernels](../../../../csrc/kernels/moe_lora_int8.cpp)、[历史实验说明](../moe_lora_v25_optimization.md)。

## 7. 普通EP SwiGLU＋INT8动态量化

~~~text
moe_lora_activation_quant(Tensor input, float limit=0., int core_limit=0)
    -> (Tensor activated, Tensor q, Tensor s)
input[R,4096] D -> activated[R,2048] D, q[R,2048] INT8, s[R] FP32
~~~

在 W13 LoRA 加完后融合；列布局为 gate2048 拼 up2048，不能交错。浮点 activated 送 W2 LoRA，q/s 送 base GMM2，不能删除浮点输出来借用 fully-sharded 的 INT8 W2 语义。

| 条件 | Python实际guard | native接口 |
| --- | --- | --- |
| 路径 | AllGather、use_ep为真、非fully-sharded | caller保证语义；native只知道输入张量 |
| flag/schema | FUSED_ACTIVATION=1、schema存在 | NPU/Meta注册，无CPU kernel |
| activation | 严格silu、topk_scales=None | 没有topk/alpha/beta参数 |
| shape | NPU D、2D、宽4096；limit0时1≤R≤288，positive limit时1≤R≤384 | 0≤R≤384、宽4096 |
| limit | finite、0..65504 | 同上；负值/NaN/Inf报错 |
| storage | contiguous、eager physical0/2 | contiguous、runtime physical0/2 |
| cores | 生产不传调参值 | core_limit整数0..64，0默认；R0不launch |

noClip R289..384 的合法生产输入回旧链，虽 native API 支持。positive clip384 不能据此推导所有 workload都收益；新区域依据独立 operator测量选择。不同 I、GELU/TANH/SWIGLUOAI/交错OAI/STEP、非None topk scales、共享expert带alpha/bias的clipped激活均保留原 activation＋scales＋quant，不改变 activation 数学。

融合与精度方法：

1. D→FP32；limit>0时gate上限clip、up对称clip，各自Q_D再回FP32，保留clamp存储边界。
2. FP32 gate/(1+exp(-gate))×up，Q_D得到返回floating。
3. quant基于舍入后的floating，不能使用未舍入FP32 act。
4. scale使用vector Muls(amax,float(1/127))，multiplier使用vector Div(127,amax)；scalar改写曾导致INT8差1/scale bit差，不采用。
5. FP32 round-even→INT32→INT8；zero row输出q0/s0。
6. scale以精确4B DataCopyPad写出，源32B对齐，避免覆写相邻scale。

默认 cores=min(available,R)，R≤48再cap24；显式 core_limit>0 用 min(available,R,core_limit)。单launch整行处理、跨核row步进，无跨core reduction；UB43136B，每行输入8192B、输出4096＋2048＋4B。这是 Vector 融合，不是 Cube。BF16需CCE220+，实际支持以direct-kernel构建与schema为准。

FakeTensor/Meta不读运行时descriptor；compile Python跳过格式查询，NPU native仍强制格式0/2。eager blocked输入保留旧路径；compiled blocked输入明确拒绝，不动态fallback。输入无NaN/Inf值域guard；bitwise结论只覆盖已记录的finite/tiny/zero/clamp/exp边界。

本轮 [20case测量](evidence/2026-10-07/analysis/activation-rows-result.json) 覆盖 BF16/FP16、R48/96/192/288/384、limits0/10，每case8次变化输入图replay，三输出raw bits全相同。BF16 R288 noClip 19.0214→16.3302µs（1.1648×）；clip10 71.4884→17.6040µs（4.0609×）。当前真实 routed limit0，不能用clip10收益解释整网。该20case实验加载历史正式binding，native算术本轮未改；fresh组合binding另完成BF16/R288/limit0真实guardhit、fullgraph npugraph_ex＋outerNPUGraph，3次变化输入三输出rawbit exact，[生产验证](evidence/2026-10-07/analysis/production-binding-validation.json)。

源码：[Python gate](../../../../vllm_ascend/lora/activation.py)、[binding](../../../../csrc/moe_lora_activation.cpp)、[kernel](../../../../csrc/kernels/moe_lora_activation.cpp)。

## 8. Cube W13 A Prefill

真实实现为cube_moe.py的gather/mask＋aten.bmm.dtype_out(...,FP32)＋scatter，矩阵乘走现有Cube。**没有已接入grouped_w13_cube C++算子，也没有Decode Cube kernel。** 不融合W13 B、W2、base GMM或整层。V1 DSpark5支持此次仅扩大静态启动条件，不把counts回读带入图或draft。

### 8.1 全部静态条件

| 项目 | 条件 |
| --- | --- |
| 开关/量化 | W13_CUBE_PREFILL=1、基座W8A8 |
| graph | 原non-spec：enforce_eager，或FULL_DECODE_ONLY且sizes非空、min≥1、max≤8；V1 DSpark5：必须FULL_DECODE_ONLY、sizes非空、min≥1、max≤48，即使enforce_eager=True也不豁免这一spec guard |
| speculative | speculative_config=None，或显式V1（VLLM_USE_V2_MODEL_RUNNER is False）＋use_dspark()＋num_speculative_tokens==5；其他method、V2、runnerNone、DSpark非5都不支持 |
| 并行 | TP8、EP8、DP1、PP1、PCP1、DCP1 |
| engine | max_num_seqs≤8、max_num_batched_tokens≤16384 |
| backend/overlap | allgather，multistream_overlap_shared_expert=False |
| layer | use_ep=True、非fully-sharded、W13两片、H4096、E_local32、topk6 |
| LoRA | max_lora_rank16、BF16 |

I2048和历史SP=False是测量条件，**不是Cube static gate**；Cube只计算A，不依赖B的I，代码也没有额外SP=False硬guard。

### 8.2 runtime与融合方法

实际 runtime 需 comm_type=ALLGATHER、not allgather_int8；`supports_cube_target_prefill_context` 要求 context非None、cudagraph_runtime_mode=NONE、in_profile_run is False、attn_metadata非None、is_draft_model is False。缺字段/None保守回原路径；FULL/PIECEWISE图、profile、无attention的dummy、draft先回退，不读counts。此 helper 已在 quant_moe 的真实caller接入，context 不能仅凭Punica.is_prefill判定。

当前 V1 DSpark5 配置 target Decode容量最高R288，draft观测R240；两者亦低于prepare的R6144门槛，即使runtime为NONE也不会回读counts。真实 target Prefill（包括chunked Prefill最后1token）由runner强制NONE，只有大R/单已加载adapter等完整guard满足才可接受。大shape且具attention metadata的NONE非profile warmup原则上可能通过runtime helper；通常loaded-slot/single-adapter挡掉dummy，不能宣称所有warmup一律排除。若以FULL等graph context调用 opaque MoE，helper先拒绝prepare，CPU counts/fixed scratch不进入捕获；[focused Fake/context UT](evidence/2026-10-07/analysis/cube-prefill-dspark5-compatibility.md)记录caller验证。

X[R,4096]BF16 contiguous、6144≤R≤98304；同NPU的INT64 counts[32]，group_list_type=1；counts非负、512≤sum(counts)≤R；同一个已加载正adapter覆盖全batch；scratch同stream。base混合、多adapter、未加载slot不进入。

A gate/up同shape[L,32,16,4096]BF16 contiguous。每层pack为[32,4096,32]，rank16＋16。expert-major有效prefix按64行tile，最多1567 tiles、每wave224、最多7waves：

~~~text
counts回读CPU → CPU tile plan → H2D metadata
gather [wave,64,4096]，padding mask=0
select A [wave,4096,32]
Cube BMM FP32 [wave,64,32]
gate/up两半scatter回[R+1,16]，无效写sink，返回前R行
~~~

counts.cpu().tolist()是实际同步，最大plan传输37608B；必须连同gather/scatter/plan成本计入，不能只报BMM。不可移入小Decode或capture。

prepare合法guard miss返回None→BGMV；**routing已接受后**若input identity、generation、slot、combined metadata或stream发生不匹配，ValueError，不无声fallback。combined值依赖expert-major producer和single-adapter合约，不逐值检查。

pack key包含slot/generation、storage pointer/offset/shape/stride/dtype/device。set_lora/reset_lora hooks失效重建；直接绕过hook原地改A内容无法由key检测，无content checksum。routing每batch重建，shared scratch仅同device/stream串行复用，未验证多模型并行调用。

### 8.3 持久内存

| 项目 | 大小 |
| --- | --- |
| gathered/tiles | 现共享同一112MiB arena；首次候选为两份各112MiB |
| tile_weights | 56MiB |
| 双outputs/product/values | 12.000122 / 1.75 / 0.875MiB |
| metadata/indices/mask/scalars等 | 约0.2688MiB |
| shared scratch/设备 | 191,778,186B，182.893930MiB；首次候选309,218,698B |
| pack/层 | 8,388,608B，8MiB |
| 43层pack＋shared | 552,488,330B，526.893930MiB；首次候选669,928,842B |

不含原LoRA/base权重、KV/graph pool/CANN workspace、allocator rounding、CPU plan/H2D staging及其他临时量，不能作为总模型显存。

首次安装的双buffer Cube合计638.893930MiB/rank，抵消V1 DSpark COPY1省普通MTP512MiB后仍净增126.893930MiB。真实Cube1启动KV **126909tokens/10.21GiB**，低于原max_model_len128000；controller拒绝进入bench并安全停止，不能记为有效optimized性能结果。baseline为128438/10.33GiB，仅余438tokens（按日志平均bytes/token估约36.1MiB）。

本轮随后合并gathered/tiles：`torch.index_select(..., out=gathered)` 后同stream逐元素 `torch.where(valid.unsqueeze(-1), gathered, zero, out=gathered)`，BMM消费同arena；不存在跨元素读写依赖，scatter先完成提交、下一wave才复用。保留product/outputs/packing/stream/loadercache，省112MiB（117440512B，117.44MB十进制），新Cube合计526.893930MiB，常驻净增 **14.893930MiB/rank**。新arena通过51host-source CPUUT及device12/taskqueue1独立真机验证，随后已原子同步installed同SHA `8a9e5397f3402f4aab5a2c3806bee8b2d2735942a106ef5d85ddb7ab26ec7186`，installed focused59UT PASS。[sourcehash](evidence/2026-10-07/analysis/cube-scratch-reuse-source-manifest.json)记录验证时源码；[更新后的部署manifest](evidence/2026-10-07/runtime-build/installed-source-binaries-manifest.json)保留原baseline与失败中间版本backup，native库未变，不为纯Python内存复用重建。

[真实arena复用验证](evidence/2026-10-07/analysis/cube-scratch-reuse-validation.json)在未改installed模块的独立old/new source实例上比较同一BMM，5cases含240tiles/两waves、两layer共享arena/独立pack、counts变化、slotinvalidate/reload、大→小、NaNpadding/negativecapacity/canary，两输出old/new全部rawbit0。完整[224,64,4096]where全alias，Torch可见allocated峰值增量0、aten::clone0；这不声称所有CANN内部workspace都由Torch profiler统计。Cube相对legacyBGMV因reduction次序最大abs1.64e-7，16行/每caseCPUFP64样本最大abs1.12e-7，单列容差比较而非bitexact。

新的常驻净增小于baseline平均余量只是预算，不保证实际KV容量；删除compiler整缓冲copy的峰值/graph pool收益也不能账面保证。修复后原参数重启实测KV为128277 tokens、10.32GiB，满足max_model_len128000（util0.93保持）；短请求8/8通过，8个rank全部43层实际命中Cube，正式长Prefill/8K4K场景正在测试。[独立内存审阅](evidence/2026-10-07/analysis/pair-native-safety-review.md)与[内存修复说明](evidence/2026-10-07/analysis/cube-scratch-reuse-review.md)。

源码：[Cube](../../../../vllm_ascend/lora/cube_moe.py)、[调用](../../../../vllm_ascend/lora/punica_npu.py)、[runtime gate](../../../../vllm_ascend/lora/quant_moe.py)。详细生命周期见 [历史·专项审计](notes/design-notes-cube-mtp.txt)。

## 9. 浮点基础LoRA、W13 paired Vector 与基座融合边界

### 9.1 既有浮点BGMV/SGMV

普通dense/shared LoRA按Punica.is_prefill选择SGMV/BGMV；routed MoE使用预展开per-row BGMV，即使Prefill也不自动用dense segment SGMV。Cube或新pair选中时仅替换W13 A来源，B expand与W2仍保留原逻辑。

310P或max_lora_rank≥128时普通浮点wrapper选上游torchops，否则native Vector；这不替代显式INT8调用，也不表示所有rank<128都合法。native expand支持8/16/32/64。

| schema | intended输入/语义 | 条件与限制 |
| --- | --- | --- |
| bgmv_shrink(x,A,ids,y,scale) | x/A同D，y FP32[R,r]，ids INT64[R]；FP32 reduction | H>r>0，负slot跳过；caller zeros；H>11776分tile |
| bgmv_expand(x,B,ids,y,offset,size) | x FP32，B/y同D，负slot保持y | rank8/16/32/64，r≤size，offset≥0，offset+size≤ywidth |
| sgmv_shrink(x,A,ids,seq_len,y,scale) | seq长度/adapter连续segment映射，y FP32 | caller保证合法segment、常规块对齐shape/storage |
| sgmv_expand(x,B,ids,seq_len,y,offset,size) | segment B delta加到y | rank及slice约束；不能从BGMV tail测试推导任意SGMV oddshape |
| bgmv_shrink_pair(x,A0,A1,ids,y_pair,scale) | y_pair FP32[2,R,r] | 第9.2节特化单launch；其他合法shape保留两次旧launch |

旧float binding没有完整验证全部dtype/device/contiguity/index正值上限/physical format。intended contract不是安全通用API。expand native固定原y＋delta，wrapper的add_inputs没有传给native，不是任选覆盖写。

### 9.2 本轮新增：H4096/r16 gate/up A 单launch

公开 schema 未改：

~~~text
bgmv_shrink_pair(Tensor x, Tensor weight0, Tensor weight1,
                Tensor indices, Tensor(a!) y_pair, float scale) -> ()
~~~

pair公共metadata contract：x[M,H]、A0/A1[slots,r,H]同BF16/FP16且同shape，indices[M]INT64，y_pair[2,M,r]FP32；同NPU、contiguous、slots>0、H>r>0、M/H≤INT32_MAX、r≤UINT32_MAX/4、y不overlap输入。M0不launch。非负indices须是合法A槽位，native不逐值检查；所有负值（含INT64_MIN）保留原输出位型。

| Python生产快速路径guard | 行为 |
| --- | --- |
| FUSED_SHRINK_PAIR=1、symbol存在、硬件非310P | 默认0；旧库缺symbol回旧A循环；旧库若有旧pair symbol仍可能仅双launch |
| 普通float double-A | input_scale=None、非fully-sharded、非fuse_swiglu；Cube已接受时优先Cube |
| x | NPU BF16/FP16、contiguous `[M,4096]`，1≤M≤384 |
| A | 两个同shape `[adapter_slots,E_local,16,4096]`、两个leading dim正、与x同dtype/device、contiguous |
| indices | `[M]` INT64、同device、contiguous；producer保证有效非负combined index |
| eager storage | x、A0/A1、indices physical0/2；blocked miss回原A循环 |
| compiling storage | Python不查Fake descriptor；runtime native检查x/A0/A1/indices/y全physical0/2，blocked报错 |

Python一次zero-init `[2,M,16]` FP32，native用plane-major布局；两个plane是连续slice，后续原B loop读对应slice。当前W13 A stacks可展平为[3×32,16,4096]，不用每forward pack或修改loader。W2 H2048、INT8 A、fully-sharded rank通信和旧INT8融合不命中此路径。guard不单查phase，小capacity Prefill也可能使用pair。

旧TILE_LENGTH11776下H4096只走一段ReduceSum。新kernel每row共读X一次/Cast一次，分别读取两组rank16权重，保留FP32 Mul/ReduceSum、`acc=+0; acc+=sum`、最后Muls(scale)的顺序。32个dot product与权重读取量未减少；少一次launch和一份X读取/Cast，不保证2×吞吐。rows以blockIdx开始、step=blockNum跨核步进，让局部有效prefix分散到更多vector cores；不支持任意tensor strides。UB49280B（48.125KiB），每plane每row恰64B DataCopyPad，uint64 offset，negative skip发生在unsigned转换前。publicAPI在H/rank/M不匹配特化时保持两次旧实现，不缩窄旧metadata域。

本轮 standalone POD：[24case结果](evidence/2026-10-07/analysis/shrink-pair-result-slots96.json)，BF16/FP16、M48/144/288/384、prefix/random/full、96slots，eager/变化输入/weights/indices replay rawbit exact；inactive signed-zero、INT64_MIN、非零output storage_offset/canary测试通过。该probe使用device12、task_queue_enable=0，与installed Torch binding/taskqueue1/npugraph_ex验收区分。BF16 prefixM288/36valid rows 3.5807×，randomM288约1.0104×，fullM384约0.9897×；收益依赖局部occupancy/row调度，不能套为整网倍数。fresh full库已经安装，正式Torch入口已在device12/taskqueue1、BF16/M48和288验证guardhit、eager/compiled/outerNPUGraph三次变化输入及全部weights/indices rawbit0，含allinactive→恢复；同时negativepreserve/padding123、Meta、alias/noncontiguous/blocked及cachedcompiled ND→blocked拒绝均通过。[生产结果](evidence/2026-10-07/analysis/production-binding-validation.json)、[脚本](evidence/2026-10-07/analysis/production-binding-validation.py.txt)、[日志](evidence/2026-10-07/analysis/production-binding-validation.log)。这批正式入口验证仍不是整网吞吐或完整FP16/M384生产验证。

源码：[kernel](../../../../csrc/kernels/bgmv_shrink.cpp)、[constants](../../../../csrc/kernels/bgmv_shrink_pair_config.h)、[Punica branch](../../../../vllm_ascend/lora/punica_npu.py)、[binding](../../../../csrc/torch_binding.cpp)、[Meta](../../../../csrc/torch_binding_meta.cpp)。数学/边界详见[独立review](evidence/2026-10-07/analysis/pair-native-safety-review.md)。

### 9.3 基座融合与未接入实验

仓库grouped_matmul_swiglu_quant、weight_nz/tensor_list、v2是既有base GMM1＋SwiGLU＋quant，不收LoRA A/B/mapping，不是本轮新增。moe_mlp有active LoRA时先进入quant_moe专用链，必须在GMM1和activation间加W13 delta；payload.fusion=True也不会自动使用base融合跳过LoRA。当前profile的DSpark draft可能使用既有base融合，因为draft无相应LoRA调用；不能据此推断target LoRA也走相同路径。

whole-weight BF16 cache仍是独立未接入prototype，无production开关。固定shape device-only Cube planner、gather/BMM/scatter或混合Vector/Cube Decode仍未实现，不能与本轮paired Vector或既有Prefill Cube混称。

源码：[float wrappers](../../../../vllm_ascend/lora/lora_ops.py)、[pair validation](../../../../csrc/bgmv_shrink_pair_validation.h)、[base分派](../../../../vllm_ascend/ops/fused_moe/moe_mlp.py)。

## 10. LoRA Decode graph与两层缓存

定长mapping避免unique/nonzero造成value-dependent输出shape；combined权重view为[L×E_local,...]，inactive=-1，replay重新读mapping/enabled。

- graph需mode非NONE、CompilationMode.VLLM_COMPILE、非enforce_eager。
- 实际请求只要computed_tokens<prompt_tokens就eager，包括chunked Prefill最后1token。
- uniform Decode按scheduled query_len/总tokens/spec条件判断，SP descriptor需TP整除。
- dispatcher传真实has_lora/num_active_loras；dummy capture以待捕获bucket同时传force_has_lora/force_num_active_loras，避免base graph和错误最大adapter bucket。profile仍使用max_loras；capture/warmup使用对应bucket的adapter数。
- shape/dtype/storage指针、通信结构需符合capture合约；内容可变，不擅换格式/address后期望同图适应。

npugraph_ex使用force_eager=True、inplace_pass=False、clone_input/output=False。static_kernel还受npugraph_ex、硬件、LOCAL_WORLD_SIZE和spec/sequence批次范围限制，不能设开关就宣称全静态。

### 10.1 两层cache隔离

AscendCompiler inner hash 与外层AOT compile_factors均包含 effective GROUPED_PREFILL 和六个动态bool：FUSED_ROUTING、SINGLE_ADAPTER、FULLY_SHARDED_CUBE、FUSED_ACTIVATION、FUSED_SHRINK_PAIR、OPTIMIZE_MTP_BUFFER_COPY，保留torch_npu版本及既有编译参数。GROUPED hash使用实际import-time常量，不用后来env值冒充已traced行为。新 fully-sharded Cube 虽只允许 eager prefill，仍进入两层缓存因子，避免配置身份混用。

MTP allocation分支还依赖runner/spec：inner hash新增与模型构造一致的 `skip_dspark_mtp_buffer` derived bool（COPY1＋explicitV1＋spec非None＋use_dspark）。外层已由上游官方compile_factors包含VLLM_USE_V2_MODEL_RUNNER（False/True/None分别），不新增虚拟env项。上游SpeculativeConfig原hash只用共享aux boolean等因素，DSpark/eagle3可能相同；本地幂等wrapper仅COPY1时hash `(original_hash, normalized method)`，COPY0原hash原样返回。wrapper用wraps/闭包保留original和marker，重复import/reload不重复包裹，不改config method、runner选择或模型语义。

此前 UT 验证四flag16组合、persistent cache开关miss及恢复、同aux DSpark/eagle3隔离、flag0 original hash一致性、runnerFalse/True/None、derivedskip保留/跳过、重复wrap/reload；[历史cache审计](evidence/2026-10-07/analysis/compile-cache-isolation-update.md)。该四flag记录不代表新增 SINGLE_ADAPTER 的验证，最新证据见 [专项文档](../moe_lora_single_adapter.md)。**切开关或runner/spec配置须重启engine并重新编译/capture**；隔离不能让已捕获graph/已构造buffer随env自动更新。当前正式对照禁用编译磁盘cache也不代表可以删除长期隔离。普通EP Cube只允许eager，未作为这些graph factors新增项；fully-sharded Cube使用独立因子。

DECODE_GRAPH_EVIDENCE是rank0 CPU dispatch日志，只说明选择的descriptor，不能证明设备replay完成。

源码：[runner](../../../../vllm_ascend/worker/model_runner_v1.py)、[compiler](../../../../vllm_ascend/compilation/compiler_interface.py)、[AOT patch](../../../../vllm_ascend/patch/platform/patch_compile_cache.py)。

## 11. DeepSeek V4 普通MTP stash 与 DSpark

### 11.1 保留buffer时的row writer

目标maxbatch16384、HC multiplier4×H4096、BF16的普通pre-HC buffer为512MiB/rank；T8只写256KiB。原`buffer[:T].copy_`在当前npugraph_ex functionalization下可展开为两次完整buffer TensorMove，真实本轮graph亦观测2次full-buffer moves，均值1.714136ms/replay。

保留buffer且COPY1时，helper采用既有 `torch_npu.npu_scatter_nd_update_` 原地更新前T行，buffer地址和尾部保持。不是新增自定义C++ MTP kernel，不删除MTP层或更改模型输出。

| 条件 | 行为 |
| --- | --- |
| buffer存在、COPY1 | 合法同D/device输入走row update |
| indices | 初始化持久INT64[max_tokens,1]，本例128KiB，只用前T |
| hidden | 2D、宽匹配、T≤capacity；目标D为FP16/BF16 |
| T0 | 无写入launch |
| COPY0/indices不存在/dtype或device需转换/其他不适用 | 原copy与conversion |
| shape/capacity非法 | 写前ValueError |
| bufferNone | helper no-op；不访问row indices |
| FlashComm1/SP且buffer存在 | 仍先all-gather/unpad，再写普通pre-HC residual |
| getter | 返回persistent Tensor，或第11.3节明确V1 DSpark的None |

普通eager/plain NPUGraph历史copy约9.32µs、scatter16.64µs，scatter更慢；既有helper79.188×只针对已观测compiled whole-buffer copies条件，不泛称所有MTP写入收益。COPY默认0保留旧行为。

### 11.2 DSpark消费的数据

构造后的 normalized `SpeculativeConfig.use_dspark()` 才是判断依据；用户可能按mtp接口配置DSpark，proposer注释中的mtp字样不能替代normalized method。model constructor已获得完整vllm_config.speculative_config，无需运行时NPU数据或模型权重决定skip。

V1 DSpark选择AscendDSparkProposer/DSparkDraftModel，target收集指定aux层hidden，draft main_proj/main_norm消费。本例target IDs40/41/42，对应layer boundaries41/42/43，拼接width12288，经draft projection回4096。V1普通MTP getter只有method==mtp且结果非None时rebinding；DSpark随后以aux输入覆盖相关hidden使用。draft自有forward，没有普通target `_mtp_hidden_buffer` writer。

V1 draft强制use_cuda_graph=False/eager，target graph独立判断；V1支持greedy，拒绝probabilistic draft sampling。当前V2继承上游DSparkSpeculator，有graph manager及probabilistic sampling入口，不能套用V1限制。上游V2 model_runner仅hasattr后slice getter，没有None guard，故此次严格保留V2 buffer，不修改upstream。

DSpark所需的aux收集、hc_head/norm最终输出、draft `mtp.*` 权重、projection buffer、draft KV、共享topk_indices_buffer均保留。普通MTPstash冗余不代表DSpark不使用draft MTP层或可以删除权重。

### 11.3 本轮实现：仅显式V1 DSpark跳过普通buffer

构造guard必须同时满足：

~~~python
VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY == True
and vllm.envs.VLLM_USE_V2_MODEL_RUNNER is False
and speculative_config is not None
and speculative_config.use_dspark()
~~~

此guard不限定DSpark speculative token数；Cube的DSpark5限制是单独的静态条件。env `is False` 表示明确`VLLM_USE_V2_MODEL_RUNNER=0`，不能写成not truthiness：None可能自动选择V2 DSpark。

| 配置 | buffer与forward行为 |
| --- | --- |
| COPY0，包括DSpark | 原512MiB allocation、原stash copy/gather；flag0不是disable普通MTP |
| COPY1＋显式V1＋DSpark | buffer=None、indices=None；整个pre-HC stash flatten/专用all-gather/unpad/copy分支跳过 |
| COPY1＋V2或runnerNone＋DSpark | buffer保留，合法row writer启用；避免upstreamNone slice风险 |
| COPY1＋ordinary normalizedmtp | buffer保留，pre-HC residual消费者语义不变 |
| COPY1＋non-spec/其他method | buffer保留；没有扩大为所有“不消费”配置的skip |

target aux在stash之前收集，skip只省stash服务的临时flatten/gather，不替换hidden_states，不删除其他通信。所有TP rank以同一config构造，一致跳过该collective；混合rank环境配置不属于支持合约。PP相关原return逻辑保留；此轮实际TP8/PP1条件验证优先。

模型/helper UT119项验证COPY0恢复、ordinaryMTP/nonSpec、explicitV1DSpark/V2/None、TP1/TP8、SPgather调用和aux输出保留；source-only审阅记录见[数据流与测试](evidence/2026-10-07/analysis/dspark5-target-shapes-and-mtp-buffer.md)。这些CPU/fake证据不等于真实DSpark准确性；本轮整网吞吐已完成对照，输出序列不同，未通过质量等价验证。

源码：[模型/helper](../../../../vllm_ascend/models/deepseek_v4.py)、[DSpark draft](../../../../vllm_ascend/models/deepseek_v4_dspark.py)、[proposer](../../../../vllm_ascend/spec_decode/dspark_proposer.py)、[V1](../../../../vllm_ascend/worker/model_runner_v1.py)。V2 consumer审计路径：容器 `{upstream_repo}/vllm/v1/worker/gpu/model_runner.py`。

## 12. 配置、使用矩阵与示例

下表本地开关集中注册、非敏感、严格0/1、默认0；invalid值报错。MC2/NZ/dual-stream等其他已有变量不适用“全部默认0”。应在engine启动和任何有关module import之前设置，并重启/重新capture后比较。GROUPED_PREFILL是明确的import-time常量；其余虽动态getter，在capture/构造后改env也不更新已有graph/buffer。

| 环境变量 | 功能与使用条件 |
| --- | --- |
| VLLM_ASCEND_MOE_LORA_FUSED_ROUTING | combined≤98304、EP pairs≤384/map≤512，分别guard；设1不代表两者都命中 |
| VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER | 非EP＋fully-sharded AllGather metadata融合；形状、存储与回退见[专项文档](../moe_lora_single_adapter.md) |
| VLLM_ASCEND_MOE_LORA_FULLY_SHARDED_CUBE | 非EP TP8单adapter大prefill的W13/W2 A投影；A3、FP32 GMM与回退见[专项文档](../moe_lora_fully_sharded_cube.md) |
| VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION | 普通AllGatherEP、silu、I2048、无topk scales，noClipR≤288/positiveclipR≤384 |
| VLLM_ASCEND_MOE_LORA_FUSED_SHRINK_PAIR | 普通float W13双A、H4096/r16/M≤384、ND/base0/2；不用于INT8/fully-sharded/W2 |
| VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY | 第11节显式V1 DSpark skip；其他保留buffer配置合法shape走row writer |
| VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL | import-time；非EP fully-sharded大R、窄I；历史W2更慢，推荐0 |
| VLLM_ASCEND_MOE_LORA_W13_CUBE_PREFILL | 第8节受限target eager大Prefill；原non-spec或explicitV1DSpark5，绝不进入Decode/draft graph |
| VLLM_ASCEND_DECODE_GRAPH_EVIDENCE | CPU descriptor分派日志，不是device replay完成证据 |
| VLLM_ASCEND_DISABLE_PIN_MEMORY | 驱动无法pin时workaround，不是融合加速 |

| 优化 | 普通TP＋EP | 非EP fully-sharded | 当前DSpark5 target | Prefill/Decode限制 |
| --- | --- | --- | --- | --- |
| EP routing | M≤384 | 不用 | T48/M288可命中 | 静态shape，unique destination；mapping内容可变 |
| combined/复用 | 支持guard/同布局 | helper可用 | R288可命中 | nativeR≤98304，无跨step编号cache |
| activation | noClip≤288/clip≤384、I2048 | 不用 | 实际limit0/R288可命中 | 按capacity，native上限384不同于PythonnoClip上限 |
| float pair | H4096/r16/M≤384 | 不用 | W13可命中，W2H2048不命中 | 按shape，任意负ids保留/productionzero-init |
| INT8旧融合 | 不用 | I≤256等 | 不放宽target布局 | 静态guard，不单按phase |
| grouped | 不用 | 大R、I≤256 | 依target主路径 | 默认关闭实验，无CPUcounts |
| Cube W13 A | 受限TP8EP8 | 不用 | explicitV1DSpark5 target大Prefill可命中 | actualNONE、非profile、attn非空、非draft、R≥6144、singleloadedadapter |
| MTP row writer | 模型相关 | 模型相关 | V2/runnerNone保留buffer时可用 | 特定编译条件获益，eager可更慢 |
| DSpark skip普通MTP | 模型相关 | 模型相关 | explicitV1＋COPY1 | constructor选择；V2/None/mtp/nonSpec保留 |

并入现有TP8/EP8命令的候选片段如下（真实模型/LoRA路径、dtype/rank/module列表、DSpark配置及既有参数以当前launcher为准）：

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_MOE_LORA_FUSED_ROUTING=1
export VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION=1
export VLLM_ASCEND_MOE_LORA_FUSED_SHRINK_PAIR=1
export VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY=1
export VLLM_ASCEND_MOE_LORA_W13_CUBE_PREFILL=1
export VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL=0
~~~

~~~text
--tensor-parallel-size 8
--enable-expert-parallel
--enable-lora
--max-num-seqs 8
--max-num-batched-tokens 16384
--additional-config '{"moe_lora_ep_backend":"allgather"}'
~~~

此片段本身不自动设置graph/spec。当前候选使用DSpark5、FULL_DECODE_ONLY、capture sizes非空且≤48以及完整第8节Cube条件；普通EP不加fully_sharded_loras=True。COPY1使当前explicitV1DSpark跳过普通stash；flag0恢复旧行为须新engine。为比较单项可关闭各flag，但backend/模型/adapter/spec/seed应一致。当前Cube内存预算较紧，应先完成启动capacity验收再正式8192＋4096/C8性能。准确性用户指定后优先级，不能把随机token吞吐报告成质量验证。

non-EP grouped另需fully-sharded＋窄I，不能混入普通EP。current候选/full launcher：[参数记录](evidence/2026-10-07/bench/optimized-launcher-prepared.json)、[baseline launcher来源](evidence/2026-10-07/bench/launcher-source.json)。

## 13. 精度、图与性能证据

### 13.1 历史：2026-10-06单个本地MoE层

H4096/I2048/r16/E32/L4/K6，非零合成W8A8/LoRA；global256中rank0拥有32；limit0。含routing、两个baseGMM、W13/W2、activation/quant，**不含attention、跨rank通信、scheduler和真实权重**。5samples×50graph replays，median device-event：

| T/R | baseline µs | optimized µs | speedup | 本层减时 | activation命中 |
| --- | --- | --- | --- | --- | --- |
| 1/6 | 215.109 | 126.048 | 1.707× | 41.4% | 是 |
| 2/12 | 292.108 | 196.851 | 1.484× | 32.6% | 是 |
| 4/24 | 352.284 | 251.122 | 1.403× | 28.7% | 是 |
| 8/48 | 437.067 | 321.450 | 1.360× | 26.5% | 是 |
| 64/384 | 1382.409 | 1246.398 | 1.109× | 9.8% | 否，保留路由优化 |

initial exact＋15变化input/route/slot/mask图case exact。证据：[历史·最终单层](evidence/2026-10-06/validated-layer-limit0.json)、[历史·汇总](evidence/2026-10-06/results.json)。1.36×不是整网Decode提升36%，不能乘43层推算。

### 13.2 历史：micro/helper，不能跨方法相乘

| 对象 | scope | baseline→optimized | 证据 |
| --- | --- | --- | --- |
| combined | R36/R288单op graph | 33.416→8.386 / 38.711→11.727µs | [历史·35 cases](evidence/2026-10-06/combined-indices-native.json) |
| 仅indices复用 | 旧limit10/T8合成层 | 522.032→514.084µs，1.0155× | [历史·reuse](evidence/2026-10-06/indices-reuse-layer.json) |
| EP routing | R6/R48/R384单op graph | 88.792→8.205 / 131.651→8.201 / 135.593→31.566µs | [历史·456 cases](evidence/2026-10-06/ep-routing-native/result.json) |
| activation no-clip BF16 | R6/12/24/48 | 6.139→2.809 / 8.102→3.331 / 8.245→4.582 / 10.416→5.999µs | [历史·最终策略](evidence/2026-10-06/activation-native/final-default-result.json) |
| activation clip10 | BF16/R48 | 33.860→6.472µs，5.232× | 同上；不能替代no-clip收益 |
| MTP actual helper compiled | BF16/T8/512MiB buffer | 1725.084→21.785µs，79.188× | [历史·helper](evidence/2026-10-06/profiles/mtp_copy_integrated_result.json) |

combined/EP小单op graph保留host enqueue floor。activation使用100ops/graph、20replays/sample、3samples。MTP仅特定compiled helper，不是模型token throughput。

### 13.3 历史：2026-10-06已执行验证

| 项目 | 记录 |
| --- | --- |
| UT/lint | 188 passed / passed；[历史·UT](evidence/2026-10-06/validated-unit-tests.log)、[历史·lint](evidence/2026-10-06/validated-lint.log) |
| combined | 35 cases、3图形状变化输入，整数exact |
| EP | 456 cases、8种index/topk/map dtype，sentinel/invalid/hole；不包含duplicate支持 |
| activation reference | 78cases，floating/INT8/FP32 scale三输出rawbit exact，24变化图cases |
| activation stress | 2×1200=2400变化replays，三输出bit差0 |
| compiled activation | D两种×limit0/10，R48，4cases；变化ND exact、fresh/cached blocked拒绝 |
| compiled EP | R48/R384，变化ND exact，ND已编译callable换blocked拒绝 |
| MTP helper | D两种×T1/T8×flag0/1，8组；head/tail/pointer/输出变化验证 |
| 当时全网 | 2026-10-06未执行，当时权重不可用；当前2026-10-07基线见13.5 |

证据：[历史·activation](evidence/2026-10-06/installed-activation-reference.json)、[历史·stress1](evidence/2026-10-06/activation-native/stress-result.json)、[历史·stress2](evidence/2026-10-06/activation-native/stress-core24-result.json)、[历史·compiled activation](evidence/2026-10-06/activation-native/compiled-guard-result.json)、[历史·compiled EP](evidence/2026-10-06/ep-routing-native/storage-guard-compile-result.json)、[历史·installed EP format](evidence/2026-10-06/installed-ep-format-smoke.json)。

历史文档还记录large recover85 NPU ST、grouped W2 123 NPU precision/graph＋149 host-Meta＋36 CPUdispatch；属于先前CANN9.0.1/torch_npu2.10.post2阶段，不是本轮重新执行。其目标grouped W2更慢。当前repo其他测试文件存在，不等于本轮fresh正式安装版本全部case已执行；此表不是本轮261 UT总数。

早期activation scalar/vector舍入失败、负收益、缺load_library的fixture保留，最终不通过放宽容差掩盖差异。Meta/fake device不等于真机数值/格式验证。

### 13.4 历史整网与profile边界

9月30日TP8EP8 AllGather LoRA、43层、rank0旧graph profile：device span约49.018ms，LoRA四类kernel精确和6.681ms、index/check3.488ms、routed activation/quant0.594ms、MTP整块copy1.690ms。跨stream可能重叠，不能把sum当串行critical path；[历史·归因](evidence/2026-10-06/profiles/rank0_decode_hotspots.md)。

先前会话记录的8k＋4k/C8/batch16384为eager约29.46 output tok/s、Decode graph＋Prefill Cube约129.40 output tok/s；**两组均TP8EP8、无spec/DSpark、历史零值adapter仍走完整LoRA计算**。graph和Cube共同变化，不能归因单项Cube；不能作为非零adapter质量或非EP吞吐证明。

原result在2026-10-06审阅时未挂载、未独立复查或复测。当前已恢复权重及基线仍不能把这些non-spec历史值用于宣称本轮DSpark收益。容器存量profiles在{profiles_root}/lora-8k4k-c8-{eager-r1,graph-cube-r1}，详细来源见 [历史·专项notes](notes/design-notes-cube-mtp.txt)。

当前已恢复权重；本轮仍固定模型/adapter来源与完整参数、spec配置、真实输入/输出数、成功/失败、TTFT/TPOT、aggregate output tok/s、preemption和DSpark acceptance做新engine对照。旧单层T64/R384表中activation未命中是当时R48生产guard的事实，不因新guard而重写旧测量。

### 13.5 本轮：2026-10-07有权重正式对照与profile

正式未profile full run：TP8＋EP8＋AllGather＋W8A8＋ordinaryLoRA r16＋DSpark5，devices8–15，maxbatch16384，max_model_len128000，max_loras3；本轮routing/activation/pair/MTP/Cube各优化flags0。8请求各精确8192输入/4096输出，8/8 PASS，总输出32768，prefixcache hit0、preemption0。

| 指标 | 正式baseline | optimized正式结果 |
| --- | --- | --- |
| wall（含Prefill） | 126.306362s | 117.633957s |
| effective output throughput | 259.432696tok/s | 278.559021tok/s（+7.37%） |
| TTFT mean / p50 / p95 | 17.550260 / 17.963946 / 27.730107s | 13.484378 / 13.797043 / 21.318259s |
| TPOT mean / p50 / p95 | 23.084438 / 22.815040 / 28.475438ms | 20.106955 / 19.347203 / 25.326539ms |
| DSpark proposed / accepted | 30955 / 26580 | 31910 / 26396 |
| acceptance | 85.866581% | 82.720150% |
| mean accepted length含bonus | 5.293329 | 5.136008 |
| KV capacity / maxlen | 128438 / 128000tokens | 128277 / 128000；首次双bufferCube1的126909仅失败初始化证据，不参与跑分 |

来源：[baseline汇总](evidence/2026-10-07/bench/baseline-summary.json)、[易读摘要](evidence/2026-10-07/bench/baseline-summary.txt)、[服务与计数来源](evidence/2026-10-07/bench/baseline-service-owner.json)。使用固定full seed2026100702和全新engine避免prefixcache复用。模型/adapter preflight抽样值为0，非完整adapter扫描；随机token性能与成功输出数不证明模型/LoRA质量或完整非零adapter行为。

独立profile run是8192输入/128输出/C8，含profiler开销，不能将其wall当正式full吞吐。rank0最后5个同shape target replay：T48/R288，device span均值81.614707ms；routed BGMV 258call/13.942081ms，shared SGMV258/4.427427ms，routing recovery1462/5.913537ms，slotmetadata430/1.782222ms，routedactivation+quant86/1.263201ms，ordinaryMTPfullcopy2/1.714136ms。kernel sum/union和跨stream overlap不同，不能将算子sum当critical path或预测整网收益。

真实routing为34小kernel pattern/43层，slotmetadata10kernel pattern/层；routedlimit0，sharedclip10。相邻eager draftT40/R240、3层base融合，无BGMV/SGMV；间隔还包含采样、metadata、通信、hostdispatch，不能把40.77ms均值全归因LoRA或CPU。[profile归因](evidence/2026-10-07/bench/baseline-rank0-hotspots.txt)、[完整JSON](evidence/2026-10-07/bench/baseline-rank0-hotspots.json)。

### 13.6 本轮候选验证状态

| 层次 | 已完成与边界 |
| --- | --- |
| source CPU/fake | MTP119、cache23、Cube47及关联quant8合计197；pair/activation actorfocused98另有重叠，勿直接相加 |
| fresh installed UT | 261 passed；[log](evidence/2026-10-07/analysis/installed-unit-tests.log)、[清单](evidence/2026-10-07/analysis/installed-unit-tests.json)，CPU/Fake不等于NPU |
| activation broaden | 20case、两dtype/R48..384/limits0和10，8变化replay/case，float/int8/scale rawbit exact；[结果](evidence/2026-10-07/analysis/activation-rows-result.json) |
| newpair standalone | 24case、96slots/prefix/random/full，negative/signedzero/offsetcanary/变化输入exact；POD taskqueue0，[结果](evidence/2026-10-07/analysis/shrink-pair-result-slots96.json) |
| fresh build/install | source snapshot与installed hash一致，完整binding＋两kernel位置、rollback已记录，[manifest](evidence/2026-10-07/runtime-build/installed-source-binaries-manifest.json) |
| fresh installed NPU/taskqueue1/npugraph_ex | device12、BF16pairM48/288、fullgraph=True/dynamic=False，FX＋outerNPUGraph3replays含allinactive→恢复，weights/indices变化rawbits0；Meta/negativepreserve/canary/alias/noncontiguous/blocked/cachedND换blocked检查PASS；activation BF16R288limit0 guardhit与三输出rawbits0，[结果](evidence/2026-10-07/analysis/production-binding-validation.json) |
| Cube arena复用 | 省112MiB/rank；51host-source CPUUT、5个old/new独立source真机case全rawbit0，fullwavewhere可见峰值增量0/clone0；随后installed同SHA、focused59UT PASS，[NPU结果](evidence/2026-10-07/analysis/cube-scratch-reuse-validation.json)、[installed59UT](evidence/2026-10-07/analysis/installed-cube-reuse-unit-tests.log) |
| V1DSpark skip＋CubeDSpark5真实整网 | 首次Cube1启动KV126909<128000未bench；修复后KV128277达标，正式8K4K/C8与profile均8/8 PASS，全部rank/layer命中Cube、MTP完整stashcopy消失 |
| 输出质量与非EP可比吞吐 | 本轮未验证，质量按用户后优先级；仍需非零adapter语义验收 |

候选来源/限制：[pair/activation实验](evidence/2026-10-07/analysis/paired_shrink_activation_cube_candidates.md)、[pair独立安全review](evidence/2026-10-07/analysis/pair-native-safety-review.md)、[Cube窄扩review](evidence/2026-10-07/analysis/cube-prefill-dspark5-compatibility.md)、[MTP数据流](evidence/2026-10-07/analysis/dspark5-target-shapes-and-mtp-buffer.md)、[cache隔离](evidence/2026-10-07/analysis/compile-cache-isolation-update.md)。这些notes包含编写时的source-only/旧guard描述，应以本文与installed manifest的最新快照为准。

### 13.7 本轮实测命中、收益和边界

优化正式场景使用与baseline相同的模型/adapter路径、V1 DSpark5、服务参数和full seed，5个候选开关设1；真实输入/输出数、零cache hit和零preemption均核对。来源：[优化正式结果](evidence/2026-10-07/bench/optimized-full/result.json)、[严格比较](evidence/2026-10-07/bench/optimized-comparison.json)、[输出差异](evidence/2026-10-07/bench/optimized-output-diagnostic.json)、[实测报告](evidence/2026-10-07/performance_report.md)。

末5个同形状 target replay（T48/R288）：

| 项目 | baseline每图 | optimized每图 |
| --- | --- | --- |
| target device span mean | 81.614707ms | 66.401442ms（−18.64%） |
| routed W13 shrink | 86旧single calls，7.101680ms | 43pair calls，1.899315ms |
| routed W2 shrink | 43旧single calls，2.754090ms | 43旧single calls，2.854065ms |
| routed B expand | 129calls，4.086311ms | 129calls，4.051716ms |
| routed BGMV总和 | 258calls，13.942081ms | 215calls，8.805095ms |
| shared SGMV | 258calls，4.427427ms | 258calls，4.287153ms |
| EP routing recover | 1462calls，5.913537ms | 43native calls，0.491664ms |
| combined indices | 430calls，1.782222ms | 43native calls，0.321933ms |
| routed activation＋quant | 86calls，1.263201ms | 43native calls，0.839480ms |
| ordinary MTP完整512MiB复制 | 2calls，1.714136ms | 0 |
| main AllReduce/AIV | 87calls，6.793416ms | 87calls，6.104606ms |
| 相邻target主图间隔均值 | 40.769380ms | 37.687477ms |

W13/W2旧single角色来自同stream matmul/SwiGlu锚点和源码顺序推断；自定义op输入shape在CANN task表为N/A，不能当直接shape记录。算术总和包含跨stream重叠，不能相加当critical-path收益。draft仍T40/R240、3层，无LoRA primitive；间隔含draft、采样、metadata、通信与host dispatch。Cube在8rank全部43层均有真实prepare_accept/pair_submitted日志；日志是每state一次事件，不代表全调用次数。rank0有251次BF16×BF16→FP32 tile BMM，共24.201237ms、118种S形状，输入[S,64,4096]×[S,4096,32]，全部在eager model4294967295，主Decode图49的同类BMM为0。[Cube实测trace](evidence/2026-10-07/bench/optimized-cube-profile-evidence.json)。主图提速不等于端到端同比提速。

两条件各一次正式性能测量。8个请求的输出序列均不同，第一处差异index为[2,23,0,18,3,0,3,3]；autoregressive后续差异不能直接用作误差大小。DSpark接受率下降3.1464个百分点，draft次数6191→6382，也会影响端到端耗时。没有逐flag消融，不能归因单项优化或宣称精度等价。随机token场景与部分zero采样adapter不建立训练质量。按照用户性能优先，本轮保留实测收益和差异记录，质量验证后续单列。

来源：[优化rank0详细profile](evidence/2026-10-07/bench/optimized-rank0-hotspots.json)、[同形状profile比较](evidence/2026-10-07/bench/profile-comparison.json)、[projection拆分](evidence/2026-10-07/analysis/baseline-shrink-projections.json)。本轮owned测试服务已关闭，8–15卡已释放。

## 14. 回退/报错统一表

| 触发 | 行为 |
| --- | --- |
| AllGather EP并行/placement/EPLB非法 | 启动报错，不自动换backend |
| EP fully-sharded/未注册quant/MC2/offset等 | 明确不支持，不普通非量化fallback |
| optional flag0、schema缺失、合法guard miss | 保留原链；core旧INT8 schema缺失的lookup/error例外如下 |
| direct native已检查dtype/shape/device/contiguity非法 | 对应binding报错；旧binding未完整验证，不能泛化 |
| EP routing/activation/pair特化真实storage非0/2 | runtime TORCH_CHECK；旧ops无统一physical guard |
| compiled ND后换blocked | runtime拒绝，不动态换图；pair其他native shape仍旧两launch合约 |
| eagerblocked导致新helper miss | 回原链；不保证旧CANN支持任意blocked格式 |
| activation noClipR289..384 | Python旧activation＋quant；native单独API仍可执行≤384 |
| pair非H4096/r16/M≤384生产shape | Python旧A循环；直接public pair其他合法shape双旧launch |
| recover UB/core/scratch预算miss | native内部framework |
| 非EP排列/EPunique目的/groupedpermutation/非负Aindex越界 | 超producer数据合约，无完整device value扫描，可能不安全 |
| 旧core INT8 schema缺失 | lookup/error，需匹配部署 |
| 真实Prefill/graphdescriptor不符 | runner eager/dispatcher正常选择；不得强迫Cube进入graph |
| Cube static/runtime/shape/slot合法guard miss | BGMV；profile/graph/draft/缺ctx不读counts；成功prepare后invariant变化ValueError |
| COPY0 | 普通MTPallocation/copy/gather原行为，DSpark也刷新buffer |
| COPY1＋explicitV1DSpark | 普通buffer/indicesNone、stash专用gather/copy跳过，aux与draft不变 |
| COPY1＋V2/runnerNone/ordinaryMTP/nonSpec | 保留普通buffer；合法row writer，conversion回原copy |
| MTPshape/capacity非法 | helper写前ValueError；Nonebufferhelper直接no-op |
| 修改env/config后复用已构造engine或graph | 不支持自动重构；须重启并重新编译/capture |

## 15. 复现、部署与后续设计

### 15.1 单层复现

以下在容器执行，物理15映射npu0；新输出文件，benchmark拒绝覆盖。合成单层不代替8卡整网：

~~~bash
cd {installed_repo}
export ASCEND_RT_VISIBLE_DEVICES=15
export VLLM_ASCEND_MOE_LORA_FUSED_ROUTING=1
export VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION=1
python benchmarks/benchmark_moe_lora_decode.py \
  --device 0 \
  --library {installed_repo}/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so \
  --baseline-dir {previous_experiment_root} \
  --tokens 1 2 4 8 64 --limit 0 --iterations 50 \
  --output /tmp/moe-lora-decode-new-run.json
~~~

baseline-dir包含保存的quant_moe.baseline.py/routing.baseline.py。正式binding与standalone probe勿在同进程重复注册schema。activation-native/ep-routing-native/profiles有专项脚本，库路径/参数以保存版本为准。

### 15.2 ABI、构建与发布

recover设备查询通过Torch-free POD C bridge隔离SDK std::string ABI；platform object单独_GLIBCXX_USE_CXX11_ABI=0，Torch binding保持自身ABI，不无差别加到所有C++文件。

本轮full kernel使用干净build目录。某AscendC merge生成规则将输入.o替换成linked image，重复增量link可unknown file type；使用独立目录重建，不能据此判断运行算子错误。

发布必须同时匹配Python、binding、kernel；旧pair symbol存在不证明已载新单launch特化。kernel安装位置为vllm_ascend/libvllm_ascend_kernels.so及vllm_ascend/lib/libvllm_ascend_kernels.so。本轮fresh build/安装/回滚依据[本轮manifest](evidence/2026-10-07/runtime-build/installed-source-binaries-manifest.json)，历史部署/回滚依据 [历史·manifest](evidence/2026-10-06/runtime-source-publication.json) 和runtime-backup，不只换Python就启用新schema。

### 15.3 尚未完成的工作与候选

| 候选/验收 | 目的与前置 |
| --- | --- |
| freshpair验证扩大 | 已测BF16M48/288；其他dtype/边界/M384及更多生产图组合按新扩展需要验证，不能套用已测结果 |
| 当前组合整网 | arena复用已安装并更新manifest；新engine待验收KV≥128000，原maxlen/util.93等参数不变；8192＋4096/C8/batch16384/DSpark5完整对照 |
| 实际质量 | 真实非零adapter输出与投机acceptance语义；micro rawbits不替代输出质量 |
| DSpark V2 skip普通MTP | 当前故意不实现；先处理upstream getterNone consumer，再逐runner/TP图验证 |
| nonSpec跳过普通MTP | 当前故意保留；需审计所有consumer/runner，不能只把flag1当disable |
| noClipR>288生产扩大 | native≤384可测试，但当前R384收益小，需稳定完整链收益后放宽 |
| device-onlyCube planner/gather/BMM/scatter或小DecodeCube | 当前未实现；省CPUcounts同步需多adapter、stream、workspace、graph lifecycle和FP32reduction验证 |
| sharedexpert activation/LoRA融合 | alpha/bias/limit语义单独设计；当前不自动复用routedkernel |
| nonEP整网可比性能 | 固定同模型/adapter/工作负载和TP/spec，不能套用TP＋EP结果 |

H4096/r16 paired A、V1DSpark普通MTPskip、noClip288/clip384和CubeDSpark5窄guard已实现并fresh安装，因此不再作为“未接入”候选；正式NPU入口已完成第13.6节记录的范围；整网性能、实际CubePrefill与MTPskip已验收；尚未完成输出质量等价、非EP可比吞吐及上述扩展。

## 16. 审计与源码索引

| 主题 | 文件 |
| --- | --- |
| 历史EP/INT8/grouped/float/graph完整schema与限制 | [2026-10-06核心审计](notes/design-notes-core.txt) |
| 历史recover/combined/EP/activationUB、数据、错误合约 | [2026-10-06原生审计](notes/design-notes-native.txt) |
| 历史Cube/cache/MTP/DSpark/指标来源 | [2026-10-06专项审计](notes/design-notes-cube-mtp.txt) |
| 当前Cube arena复用/CPU与NPU内存验证 | [修复review](evidence/2026-10-07/analysis/cube-scratch-reuse-review.md)、[验证](evidence/2026-10-07/analysis/cube-scratch-reuse-validation.json)、[sourcehash](evidence/2026-10-07/analysis/cube-scratch-reuse-source-manifest.json) |
| 当前pair/activation源、micro与guard演变 | [候选记录](evidence/2026-10-07/analysis/paired_shrink_activation_cube_candidates.md)、[pair安全](evidence/2026-10-07/analysis/pair-native-safety-review.md) |
| 当前MTPskip与CubeDSpark5/runtimehelper | [MTP数据流](evidence/2026-10-07/analysis/dspark5-target-shapes-and-mtp-buffer.md)、[Cube审计](evidence/2026-10-07/analysis/cube-prefill-dspark5-compatibility.md) |
| 当前cacheflag/method/derivedbool隔离 | [cache审计](evidence/2026-10-07/analysis/compile-cache-isolation-update.md) |
| 当前完整source/binary/rollback | [部署manifest](evidence/2026-10-07/runtime-build/installed-source-binaries-manifest.json)、[sourcehash](evidence/2026-10-07/runtime-build/final-source-hashes.json) |
| 当前正式baseline与profile | [baseline](evidence/2026-10-07/bench/baseline-summary.json)、[profile](evidence/2026-10-07/bench/baseline-rank0-hotspots.json) |
| 历史单层结果与安装 | [Oct6results](evidence/2026-10-06/results.json)、[Oct6manifest](evidence/2026-10-06/runtime-source-publication.json) |
| 唯一环境变量注册处 | [envs.py](../../../../vllm_ascend/envs.py) |

后续扩大shape、扩展V2/nonSpec skip或接入新kernel时，应同步更新guard/Meta、graph/cache、使用矩阵、部署清单与验证证据；输出质量、非EP可比吞吐与正式NPU扩大范围仍未完成，不提前标记通过。
