# 2026-10-07 设计更新摘要

完整限制、回退和验证见[设计文档](../../../design.md)。本轮保留AllGather、非EP fully-sharded INT8、grouped和旧float族的合约，新增或收窄以下行为：

- 普通EP W13双A新增paired Vector特化：BF16/FP16、H4096/r16/1≤M≤384、base格式0/2，默认0。共享X及Cast、跨核row步进，保留FP32reduction与两个B expand；W2、INT8、fully-sharded不命中。fresh binding已验证BF16M48/288/taskqueue1/npugraph_ex及变化输入rawbits0。
- 普通EP SiLU＋dynamic INT8量化生产guard扩大：limit0 R≤288、positiveclip R≤384、I2048，返回float＋INT8＋FP32scale；实际DSpark5 targetT48/R288/routedlimit0命中，不能套用sharedclip10收益。
- 同一COPY flag=1，仅explicitV1＋normalizedDSpark跳过普通pre-HC MTPbufferallocation、stashcopy和专用TPgather；aux、draft、mtp.*权重保留。V2/runnerNone/ordinaryMTP/nonSpec保留buffer及row writer，flag0完全旧行为。
- Prefill Cube窄扩到explicitV1 DSpark5、FULL_DECODE_ONLY且captures1..48；runtime必须target、NONE、非profile、attention非空，大R≥6144/单已加载adapter等旧条件不变。draft、Decode、graph和缺ctx先fallback，不回读counts。
- inner/outer cache加入pair等四flag、MTPderivedskip和COPY1 normalizedmethod；grouped有效import-time值保留，runner已有官方factor；flag0恢复原spec hash，重复import不双wrap。切换需新engine/capture。
- 初次Cube1启动KV126909<原128000，未bench即安全停止。合并gathered/tiles省112MiB/rank，新scratch182.893930MiB＋43layer pack344MiB=526.893930MiB；抵消普通MTP512后常驻净+14.893930MiB。
- arena复用5个真机case（含240tiles/two waves、跨层、reload、NaNpadding/canary）old/new同BMM rawbits全0；fullwavewhere Torch峰值增量0/clone0。51host-source UT和59installedfocused UT通过，已原子安装并保留rollback；新启动实际KV与首个长Prefill验收仍需确认。

正式baseline为8k输入＋4k输出/C8/batch16384/TP8EP8/DSpark5：8/8 PASS，259.432696 output tok/s，TTFT mean17.550260s，TPOT mean23.084438ms，acceptance85.866581%，无cachehit/preemption。optimized正式结果待补，所有micro收益均非整网收益；随机token和adapter抽样为0不证明质量。
