# Cube W13 Prefill 单arena内存修复

2026-10-07。仅修改host `vllm_ascend/lora/cube_moe.py` 的CubeScratch分配和新增 `tests/ut/lora/test_cube_scratch_reuse.py`；不修改native kernel/ABI、static/runtime guards、packing、routing、BMM、outputs或模型参数。本subtask没有同步installed源。

第一次组合Cube1启动KV126909，低于原max_model_len128000，因此controller安全停止且没有bench。原gathered/tiles是两份[224,64,4096]BF16，每份117440512B=112MiB；新tiles完全alias gathered。index_select完成后同stream执行逐元素where（full input/output alias），每元素只读取自身位置并写回自身位置，BMM随后消费同arena。下一wave沿原同stream提交顺序在前waveBMM/scatter之后覆盖arena，没有跨wave读写竞争。mask保留where语义：无效padding即使NaN也归零，不能改成乘零。

本轮51CPU/fake UT通过，包括4个新meaningful真方法测试和47个原DSpark guards：多wave、NaNpadding、跨层sharedscratch/独立packedweights、negativecapacity tail零、protectedcanary、counts变化、invalidate/reload/slot切换与packcount。Ruff format/check和git diff --check通过。

物理12已获parent释放设备授权后执行focused NPU，TASK_QUEUE_ENABLE=1。旧独立buffer模块与host新alias模块保持同一BMM，5case所有两输出rawbit differences=0；包含240tiles/two waves、两layer串行共arena、changedcounts、大→小、slot1invalidate/reload。首轮harness漏显式load正式C.so，在legacy参考lookup处停止，已补load_library后完整PASS；这不是alias错误。

完整full-wave where[224,64,4096]测得outputpointer保持、paddingzero/valid值保持，CPUprofile含aten::where、aten::clone0，TorchNPU可见allocated峰值增量0。没有112MiB Torch hidden clone；该测量不宣称覆盖所有CANN内部workspace/allocator。旧scratch309218698B、新191778186B，省117440512B。新scratch182.893930MiB＋43layer pack344MiB=526.893930MiB，减普通MTP512后常驻净+14.893930MiB；不是先前误用十进制117.5作MiB所算的9.4MiB。

Cube和旧Vector BGMV的FP32求和次序不同，单列容差参考：5case最大abs1.6391e-7；每case跨prefix16行独立CPUFP64参考最大abs1.1176e-7；使用rtol1e-4/atol1e-5，不能称Cube与BGMV bitexact。oldCube/newCube同算术则确为bitexact。完整数据[JSON](cube-scratch-reuse-validation.json)、[脚本](cube_scratch_reuse_validation.py.txt)、[日志](cube-scratch-reuse-validation.log)。

[sourcehash/UT/内存manifest](cube-scratch-reuse-source-manifest.json)、[patch](cube-scratch-reuse.patch)、[旧snapshot](cube_moe.before-scratch-reuse.py.txt)。root随后已原子同步纯Python源同SHA8a9e...并更新[installedsource manifest](../runtime-build/installed-source-binaries-manifest.json)，保留失败中间版本备份；[focused installed59UT](installed-cube-reuse-unit-tests.log) PASS。native库不变；新engine以原maxlen128000、util.93、batch16384及全部原参数启动验收pending，再正式8k＋4k/C8/DSpark5对照。预算小于baseline约36.1MiB余量不代替实际启动验收。
