# MoE LoRA 优化设计与公开证据

本目录保存2026-10-06的设计快照、三份只读源码审计和小型验证结果。主文档见 [design.md](design.md)。当前代码同时包含此前优化与本轮接入项，文档中的“已实现”不等于所有模型、硬件和投机场景均已验证。

证据使用合成算子输入、非零合成LoRA/基座权重或旧profile分析。没有模型权重、真实用户prompt、生成正文、二进制或完整raw profiler数据。8K输入＋4K输出、8并发、max_num_batched_tokens16384整网尚未重测；历史整网数字明确标为未独立复核的会话记录。

## 目录与复现

- `notes/`：core、native、Cube/MTP三份完整审计。
- `evidence/2026-10-06/`：主文档链接的JSON、日志、profile分析，另附notes提及的格式guard、早期失败、tiling和MTP小型结果。
- `evidence/2026-10-06/baseline/`：公开单层benchmark必需的两份旧源码快照，不是当前生产实现。
- [evidence-manifest.json](evidence-manifest.json)：原文本SHA256、发布副本SHA256、相对路径与大小。内容脱敏后两个文件hash可能不同；记录中的生产源码/原生库hash和所有测量数字保持原样。

在已匹配Python3.12/aarch64/Torch2.10/torch_npu2.10.post4环境并完成原生构建/安装的仓库根目录，按主文档第15节执行 `benchmarks/benchmark_moe_lora_decode.py`。命令采用仓库相对baseline和library路径；运行会使用明确选定的NPU，公开证据打包本身没有运行NPU。

专项standalone probe/CMake脚本、build目录、`.so`/`.o`、compiled cache、raw profiler/SQLite/trace，以及eightrow-v1/onerow-v2早期目录仅保留在原本地实验目录，未发布。notes中提及的未附脚本及历史失效路径是来源说明，不代表公开下载入口。没有用保存JSON代替代码执行或声称所有专项实验可从本目录重跑。

## 路径占位符

| 占位符 | 原路径用途 |
| --- | --- |
| `{repo_root}` | 当时宿主仓库根目录；未公开个人目录名 |
| `{installed_repo}` | 容器中editable安装仓库根目录 |
| `{original_workspace}` | 当前未挂载的历史工作目录 |
| `{upstream_repo}` | 容器中的上游vLLM源码根目录 |
| `{original_profiles}` | 本地保存的历史profile目录 |
| `{compile_cache}` | 原机器的vLLM编译缓存目录 |
| `{capture_host}` | 原始profiler目录中的宿主名称 |

manifest里的backup/previous_intermediate、profile_directory以及旧capture来源属于原机器本地路径；本提交不包含这些目录，不能将其当作可下载backup或raw profile。容器名称、真实选定设备12–15、工具版本、数值、源码/库hash保留作为实验来源信息。日志与JSON扫描没有发现凭据字段；`tokens`、`topk`、`adapter`等词在本目录是合成测试维度或配置，不是access token。

## 范围提示

共享model runner对真实Prefill的强制eager检查没有LoRA条件，也作用于无LoRA配置/base-only请求和mixed batch；它不是仅LoRA局部策略。Native physical-format拒绝仅属于文档明确检查descriptor的新EP routing和activation接口；不能推广到recover、combined或旧LoRA算子。

## 发布检查

发布前已执行仓库要求的 `bash format.sh ci`，全部 hooks 通过；最终发布工作树的8个相关 CPU/Meta/Mock 测试文件共188项通过。它们不代表完整仓库测试套件，也没有进行新的有权重整网测试。

证据里的生产源码和原生库 hash 对应当时安装并验证的快照。发布准备仅整理了 Python 格式和 EP kernel 模板参数名称；原始验证 hash 保留，不宣称发布源码与安装快照全部逐字节一致。公开 manifest 的 published_sha256 记录发布文本本身的 hash。
