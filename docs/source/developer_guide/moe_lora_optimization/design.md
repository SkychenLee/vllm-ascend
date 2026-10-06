# AllGather W8A8 MoE＋LoRA 优化设计与算子使用约束

更新日期：2026-10-06，Asia/Shanghai。适用对象：157_moe_ep_lora 发布副本；实现与验证来源为原 fully_sharede_lora_ok 工作区和 ds-ltc-v25-ep 容器。

本文汇总此前接入和本轮完成的全部相关优化，逐项说明融合方法、使用条件、原生接口限制、回退/报错、图与缓存约束，以及已有验证。公开材料范围、路径占位符与证据校验见 [发布说明](README.md)。这里的“已实现”指代码已接入，不表示全部模型、形状、硬件或 speculative 配置均已验证。

快速定位：[优化总表](#optimization-inventory) · [路由与索引](#routing-and-indices) · [INT8融合](#fully-sharded-int8) · [新激活融合](#ep-activation-quant) · [Cube](#cube-prefill) · [MTP与DSpark](#mtp-and-dspark) · [配置](#configuration) · [验证](#validation)。

## 1. 当前状态

当前有两条不同主路径：

- **TP＋EP、非 fully-sharded LoRA**：基座为 W8A8 INT8 GMM，LoRA A/B 使用浮点。优化 AllGather EP 路由、W13/W2 索引复用、SwiGLU＋量化、Decode graph 和 DeepSeek V4 缓存写入。
- **非 EP、TP fully-sharded LoRA**：基座与 LoRA 共用 INT8 激活，LoRA A 融合反量化；保留 paired A、旧 W13 B＋激活＋量化、grouped 大行数和 large recover。
- **Cube Prefill**：仅替换受限 TP8＋EP8、BF16/rank16 配置的 W13 双 A 投影。它是现有 PyTorch BMM 组合路径，本轮没有新增 Decode Cube kernel。

**EP＋fully_sharded_loras=True 当前不支持。DSpark 有模型/proposer 接入，但现有 Cube Prefill 排除 speculative_config 非空的配置。DSpark 下跳过普通 MTP buffer 分配、写入和通信还没有实现。**

本轮缺少模型与真实 LoRA 权重。已验证算子、变化输入图回放、编译路径和合成本地 MoE 层；尚未重测 8192 输入＋4096 输出、8 并发、max_num_batched_tokens=16384 的整网性能。非 EP 的相同场景整网吞吐也没有本轮可比结果。

| 项目 | 当前快照 |
| --- | --- |
| 宿主仓库 | {repo_root} |
| 容器安装仓库 | {installed_repo} |
| 分支/基线 | fully_sharede_lora_ok / 658005654，另有工作区修改；提交号不能代表完整实现 |
| 安装版本 | editable vLLM 0.25.1、vLLM Ascend 0.1.dev1+g658005654 |
| 运行环境 | Python3.12、aarch64、Torch2.10.0+cpu、torch_npu2.10.0.post4 |
| 构建 | Release、SOC_VERSION=Ascend910_9391、CANN toolkit latest 路径 |
| 本轮计算设备 | 12–15，位于用户指定的8–15范围内 |
| 权重状态 | 原 {original_workspace} 未挂载；真实模型/LoRA权重不可用 |

Torch 的 +cpu 是宿主 wheel 标记，NPU 执行来自 torch_npu。宿主/容器19个本轮源文件一致性、原生库 hash、备份与验证见 [部署清单](evidence/2026-10-06/runtime-source-publication.json)。

- binding SHA256：360386bcd788fe7f94882743472a8ad707995954ec8e02b421e0128f6800f839。
- 两个安装位置的 kernel 库 SHA256：9be9047e3d41c1155458433de43b1bfceb6f1c3b01601279739dc9f052d9c6cf。

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

当前 EP 测量样例为 H4096/I2048/r16/E_local32/E_global256/K6。**8并发不代表每次 T=8。** 例如8请求各验证8token时，T64/M384；激活融合 R≤48 可能不命中，EP路由融合 M≤384 仍可能命中。必须依据实际阶段和padding后的形状判断。

普通动态对称量化意图为 scale=amax(abs(activated))/127、q=round_even(activated×127/amax)，zero row 输出q0/scale0。实数公式等价不保证低精度/INT8 bitwise相同，融合要保留原dtype舍入边界、CANN算术次序、top-k权重和activation alpha/beta。

### 2.1 硬件、注册与物理格式

自定义接口属于 torch.ops._C_ascend。路由、INT8与新activation提供NPU和Meta实现；旧浮点void shrink并非全部有独立Meta注册。没有原生CPU kernel；Python helper/上游 torchops 的CPU或fallback行为不能当作native CPU实现。

direct kernels 受 VLLM_ENABLE_ATB_AND_DIRECT_KERNELS 控制。当前主CMake对 ascend310p.*、ascend950 不编译注册这组direct库。已测环境为当前Ascend910_9391/A3，不能从源码存在推导310P/A5支持。

| 算子族 | 实际storage合约 |
| --- | --- |
| 新EP routing、activation_quant | 同NPU、contiguous；runtime C++明确只接受format0/2 |
| recover、combined、旧INT8/grouped | 同NPU、逻辑contiguous；没有统一physical-format guard，调用方保证普通base storage |
| 旧浮点BGMV/SGMV | caller保证正确dtype/shape、连续base storage和合法slot；旧binding检查不完整 |
| Cube组合 | 普通连续BF16矩阵、对应weight/stream布局；没有新EP核那种统一descriptor保护 |

**is_contiguous() 不足以证明raw storage为ND。** 新EP/activation的eager Python先检查格式，否则旧路径；compile时跳过FakeTensor descriptor查询，真实执行仍由C++拒绝blocked。同一个用ND编译的callable后换NZ也报错，不自动换图或fallback。

允许format0是代码合约，本轮请求0的fixture被规范化为2，实际format0未测。VLLM_ASCEND_ENABLE_NZ不能保证所有metadata/activation已经是base storage。

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

<a id="optimization-inventory"></a>

### 3.3 全部优化状态

| 项目 | 融合/减少的工作 | 适用范围 | 状态与控制 |
| --- | --- | --- | --- |
| AllGather EP兼容及默认backend | 对齐hidden/slots，固定通信结构，恢复本地expert | LoRA-enabled TP＋EP | 已接入；默认allgather |
| 无效容量行清零 | 防止未写入尾行/NaN污染 | AllGather EP | 正确性基础；不单独宣称加速 |
| recover small/sort/large | 替代通用abs/sort/gather | 非EP fully-sharded | 既有自动分派，无env gate |
| combined_indices native | mask与adapter/expert编码 | helper可用于多路径 | 既有融合；FUSED_ROUTING默认0 |
| W13/W2 indices复用 | 同forward少做一次编码 | 普通AllGather EP，同布局 | 本轮接入；独立于native开关 |
| EP routing native | sort/mask/map/slot恢复单kernel | M≤384、map≤512 | 本轮新增；FUSED_ROUTING默认0 |
| INT8 A shrink | 反量化＋FP32 A reduction | 非EP fully-sharded | 既有核心native路径 |
| paired INT8 W13 A | gate/up共读输入、保留rank-major通信 | 非EP fully-sharded，packed A | 既有自动布局优化 |
| 旧W13 B/activation/quant | 双B、base add、clip、SwiGLU、topk、quant | 非EP fully-sharded，I≤256 | 既有fusion/shape分派 |
| grouped大行数A/B/W2 | metadata排序、按组weight缓存、宽B列tile | 非EP fully-sharded | 实验保留；默认0，目标W2曾更慢 |
| Cube W13 A Prefill | 双A打包、64行tile、Cube BMM | 受限TP8EP8、非spec | 既有；W13_CUBE_PREFILL默认0 |
| 新activation_quant | clip/SwiGLU/amax/quant，保留float | 普通EP，R≤48、I2048 | 本轮新增；FUSED_ACTIVATION默认0 |
| LoRA Decode graph | 正确bucket、定长mapping；共享runner排除真实Prefill | 符合descriptor的请求；Prefill策略也作用非LoRA | 已接入，由graph配置控制 |
| compiler/AOT cache隔离 | 避免切开关复用旧算子图 | 编译/捕获 | 本轮补充三个新开关因素 |
| MTP row writer | 避免编译后的whole-buffer copies | DeepSeek V4 main buffer | 本轮新增；COPY开关默认0 |
| float bgmv_shrink_pair | 保留paired公共接口 | 无生产Python调用 | 当前两次原launch；实验/兼容入口 |
| whole-weight BF16 cache | 缓解weight读开销候选 | 独立prototype | 未接入，无production开关 |
| DSpark跳过普通MTP | 省allocation/copy/专用gather | 未来DSpark设计 | 仅分析，未实现 |

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

<a id="routing-and-indices"></a>

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

<a id="fully-sharded-int8"></a>

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

grouped fusedB的native I≤256、I%32=0、r∈8/16/32。grouped expand输入x[R,r]FP32、B[G,width,r]和y[R,output_width]同D；r∈8/16/32/64、0<width≤8192；width/offset/output_width均16对齐、offset≥0、offset+width≤output_width；rows/groups/output_width≤INT32_MAX，y不alias输入。

width×r≤8192选narrow，超出需tiled schema把UB B cache限制到8192元素；缺tiled则普通BGMV。仅input_scale非空的INT8 LoRA路径中，其他B rank用bounded torch fallback，tile_rows=max(1,2^21/(width×r))。order必须合法完整permutation，native只查metadata不核验全部值。

**历史目标grouped W2更慢，保持GROUPED_PREFILL=0作为推荐。** 精度通过不表示性能改善，窄/宽分组实验保留，不能列为当前整网收益。

源码：[grouped](../../../../vllm_ascend/lora/grouped_prefill.py)、[Punica](../../../../vllm_ascend/lora/punica_npu.py)、[bindings](../../../../csrc/moe_lora_int8.cpp)、[kernels](../../../../csrc/kernels/moe_lora_int8.cpp)、[历史实验说明](../moe_lora_v25_optimization.md)。

<a id="ep-activation-quant"></a>

## 7. 新普通EP SwiGLU＋INT8动态量化

~~~text
moe_lora_activation_quant(input[R,4096] D, limit=0., core_limit=0)
    -> activated[R,2048] D, q[R,2048] INT8, s[R] FP32
~~~

在W13 LoRA加完后融合。列布局为gate2048拼up2048，不能交错。float activated送W2 LoRA，q/s送base GMM2。

| 条件 | Python实际guard | native接口 |
| --- | --- | --- |
| 路径 | AllGather、use_ep=True、非fully-sharded | caller保证语义 |
| flag/schema | FUSED_ACTIVATION=1、schema存在 | 已注册 |
| activation | 严格silu，topk_scales=None | 没有topk/alpha/beta参数 |
| shape | NPU D、2D、1≤R≤48、宽4096 | 0≤R≤384、宽4096 |
| limit | finite、0..65504 | 同上 |
| storage | contiguous，eager0/2 | contiguous，runtime0/2 |
| cores | 不显式传调参值 | core_limit整数0..64，0默认 |

native支持R384不代表生产使用；R49/128/384走旧链。GELU/TANH/SWIGLUOAI/交错OAI/STEP、非None topk scales、不同I均原activation＋可选scales＋quant。共享expert带alpha/bias的clipped激活不可直接替换。

融合与精度方法：

1. D→FP32；limit>0时gate上限clip、up对称clip，各自Q_D再回FP32，保留clamp存储边界。
2. FP32 gate/(1+exp(-gate))×up，Q_D得到返回floating。
3. quant基于舍入后的floating，不能使用未舍入FP32 act。
4. scale使用vector Muls(amax,float(1/127))，multiplier使用vector Div(127,amax)；scalar改写曾导致INT8差1/scale bit差，不采用。
5. FP32 round-even→INT32→INT8；zero row输出q0/s0。
6. scale以精确4B DataCopyPad写出，源32B对齐，避免覆写相邻scale。

默认cores=min(available,R)，R≤48再cap24；显式core_limit>0用min(available,R,core_limit)。单launch按整行处理，无跨core reduction，UB43136B；每行输入8192B，输出4096＋2048＋4B。收益为少launch/clamp/cat/GM往返，**不是Cube**。BF16需CCE220+。

输入没有NaN/Inf数值guard，limit finite不保证输入有限；bitwise结论限记录的finite/tiny/zero/clamp/exp边界及已测后端。

源码：[Python gate](../../../../vllm_ascend/lora/activation.py)、[binding](../../../../csrc/moe_lora_activation.cpp)、[kernel](../../../../csrc/kernels/moe_lora_activation.cpp)。

<a id="cube-prefill"></a>

## 8. Cube W13 A Prefill

真实实现为cube_moe.py的gather/mask＋aten.bmm.dtype_out(...,FP32)＋scatter，矩阵乘走现有Cube。**没有已接入grouped_w13_cube C++算子。** 不融合W13 B、W2、base GMM或整层。

### 8.1 全部静态条件

| 项目 | 条件 |
| --- | --- |
| 开关/量化 | W13_CUBE_PREFILL=1、基座W8A8 |
| graph | enforce_eager，或FULL_DECODE_ONLY且sizes非空、min≥1、max≤8 |
| speculative | speculative_config=None；当前DSpark排除 |
| 并行 | TP8、EP8、DP1、PP1、PCP1、DCP1 |
| engine | max_num_seqs≤8、max_num_batched_tokens≤16384 |
| backend/overlap | allgather，multistream_overlap_shared_expert=False |
| layer | use_ep=True、非fully-sharded、W13两片、H4096、E_local32、topk6 |
| LoRA | max_lora_rank16、BF16 |

I2048和历史SP=False是测量条件，**不是Cube static gate**；Cube只计算A，不依赖B的I，代码也没有额外SP=False硬guard。

### 8.2 runtime与融合方法

实际runtime还需comm_type=ALLGATHER且not allgather_int8；真实eager需runtime_mode=NONE、in_profile_run=False、attn_metadata非空。graph/dummy/profile先保留原路径，不读counts。

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
| gathered/tiles | 各112MiB |
| tile_weights | 56MiB |
| 双outputs/product/values | 12.000122 / 1.75 / 0.875MiB |
| metadata/indices/mask/scalars等 | 约0.2688MiB |
| shared scratch/设备 | 309,218,698B，294.893930MiB |
| pack/层 | 8,388,608B，8MiB |
| 43层pack＋shared | 669,928,842B，638.893930MiB |

不含原LoRA/base权重、KV/graph pool/CANN workspace、allocator rounding、CPU plan/H2D staging及其他临时量，不能作为总模型显存。

源码：[Cube](../../../../vllm_ascend/lora/cube_moe.py)、[调用](../../../../vllm_ascend/lora/punica_npu.py)、[runtime gate](../../../../vllm_ascend/lora/quant_moe.py)。详细生命周期见 [专项审计](notes/design-notes-cube-mtp.txt)。

## 9. 浮点基础LoRA、paired实验与基座融合边界

普通dense/shared LoRA按Punica.is_prefill选择SGMV/BGMV；routed MoE使用预展开per-row BGMV，即使Prefill也不自动用dense segment SGMV。Cube选中仅替换W13 A来源。

310P或max_lora_rank≥128时普通浮点wrapper选上游torchops，否则native Vector；这不替代显式INT8调用，也不表示所有rank<128都合法。native expand支持8/16/32/64。

| schema | intended输入/语义 | 条件与限制 |
| --- | --- | --- |
| bgmv_shrink(x,A,ids,y,scale) | x/A同D，y FP32[R,r]，ids INT64[R]；FP32 reduction | H>r>0，负slot跳过；caller zeros；H>11776分tile |
| bgmv_expand(x,B,ids,y,offset,size) | x FP32，B/y同D，负slot保持y | rank8/16/32/64，r≤size，offset≥0，offset+size≤ywidth |
| sgmv_shrink(x,A,ids,seq_len,y,scale) | seq长度/adapter连续segment映射，y FP32 | caller保证合法segment、常规块对齐shape/storage |
| sgmv_expand(x,B,ids,seq_len,y,offset,size) | segment B delta加到y | rank及slice约束；不能从BGMV tail测试推导任意SGMV oddshape |
| bgmv_shrink_pair(x,A0,A1,ids,y_pair,scale) | y_pair FP32[2,R,r] | **当前两次single launch，无生产Python调用** |

旧float binding没有完整验证全部dtype/device/contiguity/index正值上限/physical format。仅intended contract不构成安全通用API。expand native固定原y＋delta，wrapper的add_inputs没有传给native，不是任选覆盖写。

pair严格检查同D、两同shape A[slots,r,H]、ids INT64[R]、y_pair FP32[2,R,r]、同NPU且contiguous、slots>0、H>r>0、rows/H≤INT32_MAX、r≤UINT32_MAX/4、输出不overlap，empty不launch；合法正slot仍由caller保证，inactive保持原位型。其保守实现顺序调用两次原bgmv_shrink_impl；不能写成“现整网双A单launch”。

仓库grouped_matmul_swiglu_quant、weight_nz/tensor_list、v2是既有基座GMM1＋SwiGLU＋quant。它们不收LoRA A/B/mapping。moe_mlp有active LoRA时先进入quant_moe专用链，必须在GMM1和activation间加W13 delta；payload.fusion=True也不自动调用这些base融合替代LoRA边界。

whole-weight BF16 cache为未接入本分支的独立prototype，无production开关。当前普通EP BF16 gate/up paired kernel是后续候选，不与INT8 paired或未接入float pair混淆。

源码：[float wrappers](../../../../vllm_ascend/lora/lora_ops.py)、[Punica](../../../../vllm_ascend/lora/punica_npu.py)、[bindings](../../../../csrc/torch_binding.cpp)、[pair validation](../../../../csrc/bgmv_shrink_pair_validation.h)、[base分派](../../../../vllm_ascend/ops/fused_moe/moe_mlp.py)。

## 10. LoRA Decode graph与两层缓存

定长mapping避免unique/nonzero造成value-dependent输出shape；combined权重view为[L×E_local,...]，inactive=-1，replay重新读mapping/enabled。

- graph需mode非NONE、CompilationMode.VLLM_COMPILE、非enforce_eager。
- 实际请求只要computed_tokens<prompt_tokens就eager，包括chunked Prefill最后1token。该检查位于共享runner，不以has_lora或是否配置LoRA为条件，因此无LoRA配置、base-only以及含真实Prefill的混合batch同样受影响；dummy capture传入显式override，保留捕获流程。
- uniform Decode按scheduled query_len/总tokens/spec条件判断，SP descriptor需TP整除。
- dispatcher传真实has_lora/num_active_loras；dummy capture以待捕获bucket同时传force_has_lora/force_num_active_loras，避免base graph和错误最大adapter bucket。profile仍使用max_loras；capture/warmup使用对应bucket的adapter数。
- shape/dtype/storage指针、通信结构需符合capture合约；内容可变，不擅换格式/address后期望同图适应。

npugraph_ex使用force_eager=True、inplace_pass=False、clone_input/output=False。static_kernel还受npugraph_ex、硬件、LOCAL_WORLD_SIZE和spec/sequence批次范围限制，不能设开关就宣称全静态。

AscendCompiler hash和外层AOT compile_factors都纳入effective GROUPED_PREFILL、新FUSED_ROUTING/FUSED_ACTIVATION/OPTIMIZE_MTP_BUFFER_COPY，保留torch_npu版本和既有编译参数。GROUPED读实际import常量。**切开关须重启engine并重新编译/capture**；cache隔离不允许改env后继续用现有图。Cube仅eager，不属于这三个新graph factors。

DECODE_GRAPH_EVIDENCE是rank0 CPU dispatch日志，只说明选择的descriptor，不能证明设备replay完成。

源码：[runner](../../../../vllm_ascend/worker/model_runner_v1.py)、[compiler](../../../../vllm_ascend/compilation/compiler_interface.py)、[AOT patch](../../../../vllm_ascend/patch/platform/patch_compile_cache.py)。

<a id="mtp-and-dspark"></a>

## 11. DeepSeek V4 MTP writer与DSpark

### 11.1 已实现：更换writer，保留buffer

目标maxbatch16384、hc_mult×H16384、BF16时普通pre-HC buffer为512MiB/相关rank；T8只写256KiB。

特定npugraph_ex functionalization将buffer[:T].copy_展开为两次完整512MiB TensorMove。本轮helper使用已有torch_npu.npu_scatter_nd_update_原地更新前T行，buffer地址和尾部保持。**不是新增自定义C++MTP kernel，也没有取消buffer或MTP层。**

| 条件 | 行为 |
| --- | --- |
| OPTIMIZE_MTP_BUFFER_COPY=1 | 合法D输入走row update |
| indices | 初始化持久INT64[max_tokens,1]，目标128KiB，仅用前T |
| hidden | 2D、宽匹配、T≤capacity；同dtype/device且目标FP16/BF16 |
| T0 | 无写入launch |
| flag0/indices不存在/dtype或device需转换/其他不适用 | 原copy与conversion |
| shape/capacity非法 | 写前ValueError |
| SP | 仍原all-gather/unpad后用helper |
| getter | 仍返回原persistent Tensor |

普通eager/plain NPUGraph曾测copy约9.32µs、scatter16.64µs，scatter更慢；收益针对已观察npugraph_ex编译条件，默认0，不泛称所有MTP写入都79×。

### 11.2 DSpark使用关系

V1的method=dspark选择AscendDSparkProposer/DSparkDraftModel；target收集指定aux层hidden，draft main_proj/main_norm消费。V1 draft强制use_cuda_graph=False/eager，target graph独立判断；V1支持greedy，拒绝probabilistic draft sampling。

当前V2继承上游DSparkSpeculator，有graph manager及probabilistic sampling实现入口，因此不能将上述V1限制套到V2。V2的实际图/采样组合本轮未做有权重整网验证。

DSpark消费aux hidden；当前main仍分配/刷新普通MTP buffer。v1 consumer仅method=mtp且getter非None时用普通pre-HC buffer，DSpark走aux。draft中mtp.*权重名字不代表可以删除；DSpark的projection buffer、draft KV、共享topk_indices_buffer和aux收集均需保留。

### 11.3 未实现：DSpark/no-spec跳过普通buffer

未来可省普通allocation/copy及专为其写入服务的gather，但必须先处理：

1. 所有TP rank一致跳过，不能部分rank省collective。
2. 普通method=mtp仍需pre-HC residual。
3. v1有None guard；当前upstream v2仅hasattr后直接slice getter，若返回None会TypeError，须先补consumer。
4. 不误删mtp.* draft权重、DSpark projection/KV、共享topk buffer。
5. 验证实际支持的v1/v2、SP/non-SP、graph、普通MTP、DSpark、无spec组合。

flag0只是旧writer，**不是“不走MTP写入”**。当前buffer仍存在，尚未新增None风险。真实MTP/DSpark全模型效果未重测。

源码：[模型/helper](../../../../vllm_ascend/models/deepseek_v4.py)、[DSpark draft](../../../../vllm_ascend/models/deepseek_v4_dspark.py)、[proposer](../../../../vllm_ascend/spec_decode/dspark_proposer.py)、[v1](../../../../vllm_ascend/worker/model_runner_v1.py)。v2 consumer位于容器 {upstream_repo}/vllm/v1/worker/gpu/model_runner.py，审计见 [Cube/MTP notes](notes/design-notes-cube-mtp.txt)。

<a id="configuration"></a>

## 12. 配置、使用矩阵与示例

下表本地开关全部集中注册、非敏感、严格0/1、默认0；invalid值报错。其他MC2/NZ/dual-stream等已有变量不适用“全部默认0”。

| 环境变量 | 功能与使用条件 |
| --- | --- |
| VLLM_ASCEND_MOE_LORA_FUSED_ROUTING | combined≤98304与EP pairs≤384分别guard，设1不代表两者都命中 |
| VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION | 普通EP、silu、R≤48/I2048、无topk scales |
| VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY | 更换DeepSeek V4普通buffer writer，不跳过buffer |
| VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL | import-time常量；非EP fully-sharded大R；历史W2更慢，建议0 |
| VLLM_ASCEND_MOE_LORA_W13_CUBE_PREFILL | 受限eagerPrefill双A，当前不支持spec/DSpark |
| VLLM_ASCEND_DECODE_GRAPH_EVIDENCE | CPU分派证据日志，不是replay完成证据 |
| VLLM_ASCEND_DISABLE_PIN_MEMORY | 驱动无法pin时workaround，不是融合加速 |

| 优化 | 普通TP＋EP | 非EP fully-sharded | DSpark target | Prefill/Decode限制 |
| --- | --- | --- | --- | --- |
| EP routing | M≤384 | 不用 | 按验证tokens判断 | 静态shape；合法mapping可graph |
| combined/复用 | 支持guard/同布局 | helper可用 | 按R判断 | R≤98304才native；无跨step缓存 |
| 新activation | R≤48/I2048 | 不用 | 大验证batch可能fallback | 实测R48 graph/compile |
| INT8旧融合 | 不用 | I≤256等 | 不放宽target布局 | 静态guard，不单按phase |
| grouped | 不用 | 大R、I≤256 | 依target主路径 | 无CPU counts，默认关闭实验 |
| Cube W13 A | 受限TP8EP8 | 不用 | **排除** | **仅actual eager，Decode不走** |
| MTP writer | 模型相关 | 模型相关 | 当前仍刷新buffer | 特定编译条件获益 |
| DSpark skip MTP | 未实现 | 未实现 | 未实现 | 未实现 |

并入已有TP8/EP8启动命令的片段如下，省略当前不可用模型/LoRA路径和既有模型参数：

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_MOE_LORA_FUSED_ROUTING=1
export VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION=1
export VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY=1
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

保留已有rank/dtype、graph、LoRA模块、DSpark/spec配置。普通EP不加入fully_sharded_loras=True。DSpark下Cube flag设1也不命中，不为Cube改变需要的投机方法。COPY可单独关闭比较，普通MTP写入仍存在。

historical Cube需non-spec＋BF16/r16/H4096/E32/topk6和第8节完整条件。non-EP grouped另需fully-sharded＋窄I，不能混入普通EP。真实graph模式用当前vLLM compilation_config配置，不假定以上片段自动开启graph。

<a id="validation"></a>

## 13. 精度、图与性能证据

### 13.1 最新单个本地MoE层

H4096/I2048/r16/E32/L4/K6，非零合成W8A8/LoRA；global256中rank0拥有32；limit0。含routing、两个baseGMM、W13/W2、activation/quant，**不含attention、跨rank通信、scheduler和真实权重**。5samples×50graph replays，median device-event：

| T/R | baseline µs | optimized µs | speedup | 本层减时 | activation命中 |
| --- | --- | --- | --- | --- | --- |
| 1/6 | 215.109 | 126.048 | 1.707× | 41.4% | 是 |
| 2/12 | 292.108 | 196.851 | 1.484× | 32.6% | 是 |
| 4/24 | 352.284 | 251.122 | 1.403× | 28.7% | 是 |
| 8/48 | 437.067 | 321.450 | 1.360× | 26.5% | 是 |
| 64/384 | 1382.409 | 1246.398 | 1.109× | 9.8% | 否，保留路由优化 |

initial exact＋15变化input/route/slot/mask图case exact。证据：[最终单层](evidence/2026-10-06/validated-layer-limit0.json)、[汇总](evidence/2026-10-06/results.json)。1.36×不是整网Decode提升36%，不能乘43层推算。

### 13.2 micro/helper，不能跨方法相乘

| 对象 | scope | baseline→optimized | 证据 |
| --- | --- | --- | --- |
| combined | R36/R288单op graph | 33.416→8.386 / 38.711→11.727µs | [35 cases](evidence/2026-10-06/combined-indices-native.json) |
| 仅indices复用 | 旧limit10/T8合成层 | 522.032→514.084µs，1.0155× | [reuse](evidence/2026-10-06/indices-reuse-layer.json) |
| EP routing | R6/R48/R384单op graph | 88.792→8.205 / 131.651→8.201 / 135.593→31.566µs | [456 cases](evidence/2026-10-06/ep-routing-native/result.json) |
| activation no-clip BF16 | R6/12/24/48 | 6.139→2.809 / 8.102→3.331 / 8.245→4.582 / 10.416→5.999µs | [最终策略](evidence/2026-10-06/activation-native/final-default-result.json) |
| activation clip10 | BF16/R48 | 33.860→6.472µs，5.232× | 同上；不能替代no-clip收益 |
| MTP actual helper compiled | BF16/T8/512MiB buffer | 1725.084→21.785µs，79.188× | [helper](evidence/2026-10-06/profiles/mtp_copy_integrated_result.json) |

combined/EP小单op graph保留host enqueue floor。activation使用100ops/graph、20replays/sample、3samples。MTP仅特定compiled helper，不是模型token throughput。

### 13.3 本轮已执行验证

| 项目 | 记录 |
| --- | --- |
| UT/lint | 188 passed / passed；[UT](evidence/2026-10-06/validated-unit-tests.log)、[lint](evidence/2026-10-06/validated-lint.log) |
| combined | 35 cases、3图形状变化输入，整数exact |
| EP | 456 cases、8种index/topk/map dtype，sentinel/invalid/hole；不包含duplicate支持 |
| activation reference | 78cases，floating/INT8/FP32 scale三输出rawbit exact，24变化图cases |
| activation stress | 2×1200=2400变化replays，三输出bit差0 |
| compiled activation | D两种×limit0/10，R48，4cases；变化ND exact、fresh/cached blocked拒绝 |
| compiled EP | R48/R384，变化ND exact，ND已编译callable换blocked拒绝 |
| MTP helper | D两种×T1/T8×flag0/1，8组；head/tail/pointer/输出变化验证 |
| 全网 | 本轮未执行，权重不可用 |

证据：[activation](evidence/2026-10-06/installed-activation-reference.json)、[stress1](evidence/2026-10-06/activation-native/stress-result.json)、[stress2](evidence/2026-10-06/activation-native/stress-core24-result.json)、[compiled activation](evidence/2026-10-06/activation-native/compiled-guard-result.json)、[compiled EP](evidence/2026-10-06/ep-routing-native/storage-guard-compile-result.json)、[installed EP format](evidence/2026-10-06/installed-ep-format-smoke.json)。

历史文档还记录large recover85 NPU ST、grouped W2 123 NPU precision/graph＋149 host-Meta＋36 CPUdispatch；属于先前CANN9.0.1/torch_npu2.10.post2阶段，不是本轮重新执行。其目标grouped W2更慢。当前repo其他测试文件存在，不等于当前正式安装版本全部case已执行。

早期activation scalar/vector舍入失败、负收益、缺load_library的fixture保留，最终不通过放宽容差掩盖差异。Meta/fake device不等于真机数值/格式验证。

### 13.4 历史整网与profile边界

9月30日TP8EP8 AllGather LoRA、43层、rank0旧graph profile：device span约49.018ms，LoRA四类kernel精确和6.681ms、index/check3.488ms、routed activation/quant0.594ms、MTP整块copy1.690ms。跨stream可能重叠，不能把sum当串行critical path；[归因](evidence/2026-10-06/profiles/rank0_decode_hotspots.md)。

先前会话记录的8k＋4k/C8/batch16384为eager约29.46 output tok/s、Decode graph＋Prefill Cube约129.40 output tok/s；**两组均TP8EP8、无spec/DSpark、历史零值adapter仍走完整LoRA计算**。graph和Cube共同变化，不能归因单项Cube；不能作为非零adapter质量或非EP吞吐证明。

原result路径当前未挂载、未独立复查或复测。仅列历史记录，不用于宣称本轮新融合或当前DSpark整网收益。容器存量profiles在{original_profiles}/lora-8k4k-c8-{eager-r1,graph-cube-r1}，详细来源见 [专项notes](notes/design-notes-cube-mtp.txt)。

权重恢复后需固定模型/adapter hash、完整参数、spec配置、真实输入/输出数、成功/失败、TTFT/TPOT、aggregate output tok/s、preemption和DSpark acceptance，再做相同场景对比。

## 14. 回退/报错统一表

| 触发 | 行为 |
| --- | --- |
| AllGather EP并行/placement/EPLB非法 | 启动报错，不自动换backend |
| EP fully-sharded/未注册quant/MC2/offset等 | 明确不支持，不普通非量化fallback |
| 新optional flag0、schema缺失、合法guard miss | 保留原链 |
| direct native已检查的dtype/shape/device/contiguity非法 | 相应binding报错；旧binding未完整验证，不能泛化 |
| 新EP routing/activation真实storage不是0/2 | TORCH_CHECK；旧ops无统一physical guard，依caller合约 |
| 新EP routing/activation compiled ND后换blocked | runtime拒绝，不动态改图 |
| 新EP routing/activation eager blocked | 旧路径；旧CANN自身可能unsupported |
| recover UB/core/scratch预算miss | native内部framework |
| 非EP排列/EP unique destination/grouped permutation违反 | 超数据合约，未做完整data校验 |
| 旧core INT8 schema缺失 | lookup/error，需匹配部署 |
| 真实Prefill/graph descriptor不符 | eager/dispatcher正常选择 |
| Cube合法guard miss | BGMV；成功prepare后的invariant变化可ValueError |
| MTP flag0/conversion | 原copy，buffer仍刷新 |

## 15. 复现、部署与后续设计

### 15.1 单层复现

以下在容器执行，物理15映射npu0；新输出文件，benchmark拒绝覆盖。合成单层不代替8卡整网：

~~~bash
# 从已构建/安装且环境匹配的仓库根目录执行
export ASCEND_RT_VISIBLE_DEVICES=15
export VLLM_ASCEND_MOE_LORA_FUSED_ROUTING=1
export VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION=1
python benchmarks/benchmark_moe_lora_decode.py \
  --device 0 \
  --library ./vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so \
  --baseline-dir ./docs/source/developer_guide/moe_lora_optimization/evidence/2026-10-06/baseline \
  --tokens 1 2 4 8 64 --limit 0 --iterations 50 \
  --output /tmp/moe-lora-decode-new-run.json
~~~

baseline-dir包含本目录随附的quant_moe.baseline.py/routing.baseline.py源码快照。上述library文件名对应已记录的Python3.12/aarch64构建；其他ABI需使用实际构建产物。正式binding与standalone probe勿在同进程重复注册schema。原activation-native/ep-routing-native/profiles专项脚本、build与raw profiles只保留在本地，未随此文档发布；公开复现入口为仓库benchmark及相关测试源码。

### 15.2 ABI、构建与发布

recover设备查询通过Torch-free POD C bridge隔离SDK std::string ABI；platform object单独_GLIBCXX_USE_CXX11_ABI=0，Torch binding保持自身ABI，不无差别加到所有C++文件。

本轮full kernel使用干净build目录。某AscendC merge生成规则将输入.o替换成linked image，重复增量link可unknown file type；使用独立目录重建，不能据此判断运行算子错误。

发布必须同时匹配Python、binding、kernel；kernel安装位置为vllm_ascend/libvllm_ascend_kernels.so及vllm_ascend/lib/libvllm_ascend_kernels.so。部署/回滚依据 [manifest](evidence/2026-10-06/runtime-source-publication.json) 和runtime-backup，不只换Python就启用新schema。

### 15.3 未完成候选

| 候选 | 目的与前置 |
| --- | --- |
| 普通EP BF16 W13 paired A | 共读输入、少launch；需稳定view/loader、inactive、rank/reduction、graph/hotload验证 |
| shared clipped activation＋quant | 精确alpha/bias/limit，保留floating LoRA消费者 |
| DSpark普通MTP skip | 先v1/v2 consumer/TP一致性，再allocation/copy/gather |
| R>48 activation | 覆盖大投机验证；native能执行不表示稳定性能获益 |
| device-only Cube planner或gather/BMM/scatter融合 | 省CPU counts同步；当前未实现，需多adapter/stream/lifecycle验证 |
| 小Decode Cube | 把gather/padding/packing/launch纳入，与paired Vector比较 |
| 全网重测 | 恢复权重后8192＋4096/C8/batch16384/DSpark、设备8–15，吞吐与准确性 |

## 16. 审计与源码索引

| 主题 | 文件 |
| --- | --- |
| EP/INT8/grouped/float/graph完整schema与限制 | [核心审计](notes/design-notes-core.txt) |
| recover/combined/EP/activation完整UB、数据、错误合约 | [原生审计](notes/design-notes-native.txt) |
| Cube/cache/内存/MTP/DSpark/历史指标来源 | [Cube与MTP审计](notes/design-notes-cube-mtp.txt) |
| 当前实现与已执行验证 | [results.json](evidence/2026-10-06/results.json) |
| 部署源/二进制/rollback | [runtime-source-publication.json](evidence/2026-10-06/runtime-source-publication.json) |
| 唯一环境变量注册处 | [envs.py](../../../../vllm_ascend/envs.py) |

后续扩大shape、实现DSpark skip或接入新kernel时，应同步更新guard/Meta、graph/cache、使用矩阵、部署清单与验证证据；本设计候选不提前视为完成。
