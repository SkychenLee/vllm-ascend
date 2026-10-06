# Paired W13 shrink：独立数学、原生存储与内存审阅

审阅日期：2026-10-07。审阅对象为 host `{repo_root}` 当前候选源码；只读生产代码，不启动 NPU，不安装运行时。本文与 `pair-integration-review.md` 独立，重点覆盖原生数学和边界。下列已执行验证来自现有 artifact，并非本次重新执行。

## 结论

未发现阻止组合部署的 paired shrink 代码问题。H=4096、rank=16、0<M<=384 的候选沿用旧 FP32 Cast/Mul/ReduceSum/acc/scale 顺序，将 gate/up 两次 shrink 的 X 读取和转换合并到一次 launch；不改 B expand、top-k 权重或后续激活。

初稿时待完成的 installed Torch NPU 入口、task-queue=1 和 npugraph_ex 验证现已通过：正式binding在device12、BF16/M48与288、fullgraph=True/dynamic=False、FX＋outerNPUGraph中3次变化输入/全部weights/indices，含allinactive→恢复均rawbits0；Meta、negativepreserve、canary、alias、blocked和cachedND换blocked拒绝通过。[生产结果](production-binding-validation.json)。这是额外正式入口证据，与standalone24case范围区分，仍不等于整网吞吐。内存方面，Cube 的新增常驻存储可能超过 baseline 的 max-model-len 余量，启动验收必须以候选的实际 KV capacity 为准。

## 1. 数学顺序与适用区域

来源：`csrc/kernels/bgmv_shrink.cpp:29,255,389`。

旧 shrink 的 TILE_LENGTH=11776；本候选 H=4096 小于单 tile。对每个有效 routed row、每个 rank，旧算子在余数路径完成：将 X/W 转 FP32，FP32 Mul，长度 4096 的 ReduceSum，`float acc=0.0f; acc+=sum`，最后在 rank 向量上 Muls(scale)。新版保留同样操作、长度和 scalar 加零顺序，且 scale 同样在 host 从 double 转 float。X FP32 buffer 在 gate/up 两个 plane 共用；W FP32 buffer 每个 rank 重新写入。不改变 reducer 的长度，不把 FP32 ReduceSum 替换成 Cube 或低精度积累。

单核 UB 分配约 49,280 B（48.125 KiB），所有定长缓冲大小满足对齐要求。新 kernel 只有 FP16/BF16 两种输入，FP32 输出。新硬件路径的 BF16 编译条件沿用 `__CCE_AICORE__ >= 220`。公开 pair 入口对其他合法形状继续调用两次旧 shrink；Python 生产快速分支则只允许 H4096/r16/M<=384。

生产调用位于 `vllm_ascend/lora/punica_npu.py:539`：Cube 路由优先，其次仅在 `input_scale is None`、非 fully-sharded、非 fused SwiGLU 时考虑 vector pair。A 的实际堆叠形状必须是两个相同的 `[adapter_slots, local_experts, 16, 4096]` BF16/FP16 张量。展平前两个轴后传给原生入口；两 plane 各自复用原有 B expand。W2 的 shrink 输入宽度 2048，不命中新分支；INT8 A、fully-sharded 的 TP reduction/fused 分支亦保留原路径。

## 2. row 调度、负 index 和输出边界

每个 block 使用 `row=blockIdx; row<M; row+=blockNum`，blockDim 为 min(M, queried_vector_cores)。因此即使 M=384、vector cores=64，每个 row 仍有唯一执行者且全部覆盖。这里支持的是行的跨核步进，不代表支持任意 strided tensor；公开入口和 Python 均要求逻辑 contiguous。

所有负 int64 index（包括 INT64_MIN）在转 uint64 或读取权重之前直接跳过。负行保留调用者原输出；生产分支先 zero-init 一个 `[2,M,16]` FP32 buffer，所以 inactive 行为零。每个 plane 的 offset 是 `(plane*M+row)*16` float，每次 DataCopyPad 恰好写 64 B，不会触及相邻 plane/row 或 protected padding。使用 uint64 进行权重/输出地址运算避免常见 32 位 offset 溢出。native binding 检查输出不与输入/两组权重/indices 重叠；输入彼此重叠只产生读操作。

仍沿用原有 producer 契约：所有非负 combined index 必须落在展平后的 A 槽位范围内。metadata checker 只检查张量形状、类型、设备和连续性，不扫描 NPU indices 的最大值；直接手工调用公开 API 传正的越界 index 不安全。这是旧 BGMV 同有的边界，生产路由保证有效 index；不应将本接口描述为能接收任意 int64 index。

## 3. 物理格式、FakeTensor 与 fallback

来源：`csrc/bgmv_shrink_pair_validation.h:10`、`csrc/torch_binding.cpp:320`、`csrc/torch_binding_meta.cpp:53`、`vllm_ascend/lora/punica_npu.py:25`。

Python gate 检查 NPU、dtype、shape、contiguity、indices int64 同设备、硬件非 310P、flag=1 和 native symbol 存在。eager 时 X、两组 A、indices 必须物理 format 0/2；不满足则进入既有 Python 路径。逻辑 contiguous 不足以判断 blocked storage，所以此检查不可删除。

编译时 Python 跳过 get_npu_format，因为 FakeTensor 无法查询运行时 NPU descriptor。Meta 只做元数据验证，不查询 descriptor、不访问 data_ptr；实际 native 特化路径对 X、A0、A1、indices、y_pair 全部检查 physical format 0/2 后才 launch。compiled blocked 输入因此会明确报错，不能声称编译后也自动 fallback。原有普通路径对 blocked 格式本身可能受 CANN 限制；Python fallback 表示未选 pair，不保证任意 blocked 输入的旧算子都支持。

flag 默认 0；此版本已有 pair schema 不变：`bgmv_shrink_pair(Tensor x, Tensor weight0, Tensor weight1, Tensor indices, Tensor(a!) y_pair, float scale) -> ()`。旧 `.so` 可能存在同名 symbol，却只实现两次旧 launch，故 has-symbol 仅是兼容门槛，不能证明载入的是新融合版本。部署 provenance 和实测 operator trace 必须确认。

## 4. 现有验证证据及范围

[standalone raw-bit 结果](shrink-pair-result-slots96.json)：device 12，64 vector cores，96 个有效 weight slots，BF16/FP16，M=48/144/288/384，prefix/random/full 路由，共 24 case。eager 与两个旧 launch 的 raw-bit differences 均为 0；预填输出保留测试均为 0；changed-input/replay 差异为 0；所有 canary 值保持 `[123,123]`。负 index 覆盖 -1 和 INT64_MIN；输出保护测试使用 backing[1:-1] 的非零 storage offset，并测试 scale=-0.5 与 inactive -0.0。

该 artifact 标明 Torch-free POD adapter、task_queue_enable=0；这是对原生算术和内存写边界的证据；正式Torch binding/npugraph_ex另由production-binding-validation.json验证，均不是整网性能证据。CPU UT 另覆盖 helper gates、0/384/385、H/rank/dtype/device/contiguity、physical 29、FakeTensor、INT8/fully-sharded exclusion、zero-init/plane view、Cube priority。actor 报告本轮 98 项 CPU UT 通过。

## 5. Cube/MTP/KV 内存预算

首次双buffer候选 CubeScratch 的固定张量合计 **294.893930 MiB/rank**；每层 packed A `[32,4096,32]` BF16 为 8 MiB，当前 43 个 MoE 层合计 **344 MiB/rank**。总计 **638.893930 MiB/rank**，尚不计 allocator rounding、BMM workspace 和首次打包临时张量。

当前 max-num-batched-tokens=16384、HC multiplier=4、H4096、BF16 的普通 MTP buffer 是 **512 MiB/rank**。明确 V1 DSpark 且 MTP optimization flag=1 时不分配普通 buffer，且其专用 stash copy/all-gather 消失。对原 baseline flag=0 比较，常驻存储单看这两项净增 **126.893930 MiB/rank**；新 flag=1 普通 MTP 额外 row-indices 的 0.125 MiB 不应计成 baseline 已有节省。Pair 的最大 `[2,384,16]` FP32 工作区仅 48 KiB，旧两个 shrink 输出总大小也为 48 KiB，不是主要常驻增量。

baseline 日志实际 KV cache 为 **128,438 tokens**，max-model-len 为 **128,000**，余量仅 438 tokens。按日志 10.33 GiB KV / 128,438 的平均值线性估计，此余量约 **36.1 MiB**；实际异构/压缩 block 对齐会使换算不同。仅上述常驻净增就超过此估计余量，故不能只凭删除 512 MiB 普通 buffer 判定更省内存。

baseline profile 还出现普通 MTP stash 带来的整缓冲 compiler copy，skip 分支会删去这些操作；但不能把两次 512 MiB copy 直接记成必然可返还 KV 的 1 GiB，因为峰值、graph pool 和 allocator 生命周期需要实际测量。Cube 的 profile/warmup context guard 会排除某些 profile forward；BMM workspace 和首次 pack 峰值也需在真实长 Prefill 中观察。

部署验收：以候选启动日志检查 KV capacity 是否仍达到 128000、graph/activation/non-Torch/model 内存项；再检查第一轮真实长 Prefill 是否出现 OOM、preemption 或容量不足。此项为具体已观测的边缘容量风险，未发现 pair 本身的额外显著内存风险。现阶段正式 baseline 吞吐为 259.4327 tok/s，optimized 整网结果尚待 parent 补充，本文不声称吞吐提升。

## 6. 后续已观测容量失败与arena修复

首次双buffer Cube1正式启动KV126909<原maxlen128000（10.21GiB，baseline10.33GiB），controller未bench即停止。此结果验证了第5节常驻内存风险。随后host源码将tiles完全alias gathered，逐元素where out=gathered，省112MiB。新sharedscratch182.893930MiB、43layer total526.893930MiB，抵消MTP512后净+14.893930MiB。原参数新启动容量仍待验收，不能只以此预算标记通过。

[arena复用真机结果](cube-scratch-reuse-validation.json)：device12/taskqueue1、5cases，old/new同BMM rawbits0，涵盖240tiles/two waves、两layer共享scratch、counts/slot/reload变化、NaNpadding/negativecapacity/canary；完整wave where Torch allocated峰值增量0、aten::clone0。相对legacyBGMV最大abs1.64e-7，CPUFP64每case16行样本最大1.12e-7，属于独立容差对照。51CPUUT与Ruff PASS，未由本subtask修改installed文件；安装来源[manifest](cube-scratch-reuse-source-manifest.json)。

root随后已同步arena复用installed同SHA8a9e...，更新manifest保留intermediatebackup；focused installed59UT PASS。新engine原maxlen128000/util.93容量验收和正式整网性能仍pending。
