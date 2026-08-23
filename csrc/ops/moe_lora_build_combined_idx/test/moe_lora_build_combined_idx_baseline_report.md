# AscendC 标量基线性能报告

- 标杆为 NPU 小算子拼接：`abs/argsort/gather/div/clamp/index/where/contiguous`。
- Trace：`/opt/wqy2/temp/moe_lora_profiler_trace_baseline`
- 指标：`op_statistic.csv` 全部算子的 `Total Time(us)` 求和 / active(5)。
- 固定 schedule：warmup=5、active=5、repeat=1。

| Case | Shape | DType | 单核标量 AscendC (us) | AiCPU 小算子链 (us) | 加速比 |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | [1, 6] | bool | 2.376 | 81.753 | 34.408x |
| 1 | [8, 6] | int8 | 3.368 | 82.230 | 24.415x |
| 2 | [32, 1] | bool | 2.616 | 75.137 | 28.722x |
| 3 | [128, 6] | int8 | 21.120 | 176.611 | 8.362x |
| 4 | [512, 6] | bool | 76.062 | 458.893 | 6.033x |
| 5 | [2048, 6] | int8 | 468.637 | 2618.820 | 5.588x |
| 6 | [4096, 6] | bool | 1110.762 | 6832.509 | 6.151x |
| 7 | [512, 8] | int8 | 98.142 | 602.952 | 6.144x |

平均加速比为 14.978x。大 shape 受单核标量 GM gather/scatter 限制。
