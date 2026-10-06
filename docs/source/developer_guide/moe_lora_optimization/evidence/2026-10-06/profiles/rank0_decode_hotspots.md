# Rank 0 Decode profile analysis (2026-10-06)

Source is the September 30 optimized TP8/EP8 AllGather LoRA profile; analysis does not run a model or touch NPUs.

T8, topK6, R48, H4096, I2048, E32 local, r16; 43 layers. SQLite model ID 49 has 12 graph replays with exactly 8308 tasks each. Device span mean 49.018 ms (48.442–49.552), starts 57.385 ms apart; outside-graph gap mean 8.415 ms. These are profile timings, not a new benchmark.

| Kernel | Calls / Decode step | Summed device ms / step |
|---|---:|---:|
| aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul | 86 | 4.290 |
| AivKernel | 87 | 3.568 |
| aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3 | 236 | 3.048 |
| bgmv_shrink_bfloat16_t_2 | 129 | 2.930 |
| Compressor | 62 | 2.882 |
| HcPre | 86 | 2.850 |
| aclnnTransposeBatchMatMul_TransposeBatchMatMul_TransposeBatchMatMul | 43 | 2.318 |
| aclnnIndex_IndexCheck_IndexCheck | 215 | 1.776 |
| aclnnIndex_IndexAiCore_Index | 215 | 1.712 |
| aclnnInplaceCopy_TensorMoveAiCore_TensorMove | 2 | 1.690 |
| sgmv_shrink_bfloat16_t_35 | 129 | 1.639 |
| aclnnMatmul_MatMulCommon_MatMulV2 | 149 | 1.606 |
| aclnnDynamicQuantV2_DynamicQuant_DynamicQuant | 300 | 1.481 |
| aclnnInplaceMaskedFillScalar_MaskedFillAiCore_MaskedFill | 130 | 1.468 |
| SparseAttnSharedkv | 43 | 1.308 |
| bgmv_expand_bfloat16_t_0 | 129 | 1.307 |
| aclnnInplaceCopy_CastAiCore_Cast | 369 | 0.915 |
| sgmv_expand_bfloat16_t_33 | 129 | 0.807 |
| aclnnScatterNdUpdateV2_ScatterNdUpdateV2AiCore_ScatterNdUpdateV2 | 126 | 0.789 |
| RmsNorm | 130 | 0.716 |
| InplacePartialRotaryMul | 150 | 0.708 |
| HcPost | 86 | 0.670 |
| VllmQuantLightningIndexer | 21 | 0.646 |
| aclnnMoeInitRoutingV3_MoeInitRoutingV3_MoeInitRoutingV3 | 43 | 0.530 |
| aclnnClamp_ClipByValueV2AiCore_ClipByValueV2 | 129 | 0.442 |
| aclnnGeScalar_GreaterEqual_GreaterEqual | 173 | 0.434 |
| SwiGlu | 43 | 0.372 |
| aclnnArange_ArangeAiCore_Range | 43 | 0.347 |
| triton_rms_kernel | 43 | 0.342 |
| aclnnSWhere_SelectV2AiCore_SelectV2 | 132 | 0.337 |

Durations are summed kernel durations; concurrent streams overlap. AivKernel is the AllReduce implementation (87 calls / step, 3.568 ms). kernel_details.csv also lists matching hcom_allReduce entries, so adding them would double-count communication. No AllGather kernels were observed in this DP1/TP8 captured graph.

Custom LoRA/Triton entries in kernel_details.csv lose model ID and appear as 4294967295. Canonical selection uses SQLite TASK.modelId=49 joined to COMPUTE_TASK_INFO; its task IDs repeat exactly 12 times. Filtering CSV Model ID 49 alone would omit the important LoRA kernels.

## Activation evidence

operator_details.csv contains npu::npu_swiglu / aclnnSwiGlu (344 real Prefill API entries each) and no npu_clipped_swiglu. Captured SwiGlu input is 48×4096 with attr_0:-1. No external 48×2048 or 48×4096 Clamp appears. The profile demonstrates ordinary SwiGLU, and does not establish a positive routed clamp limit. The 8×256 clamps belong to the shared expert activation. Routed SwiGlu + DynamicQuant(48×2048) costs 0.372 + 0.222 = 0.594 ms / step.

## Actionable candidates

1. EP metadata recovery and adapter lookup: IndexCheck + Index sum 3.488 ms / step, before masks and arithmetic. Native fused recovery can eliminate repeated gathers/checks, but benchmark the exact real shapes and mapping contract.
2. Unconditional MTP residual stash: two BF16 TensorMove operations of 268435456 and 16384×16384 elements repeat every graph step, totaling 1.690 ms. Model allocates _mtp_hidden_buffer at [max_num_batched_tokens, hc_mult*hidden_size] = [16384,16384], and forward copies only [:num_tokens] into it. Both shapes match the full 512 MiB buffer, near the final HcPost and final Norm. Actual npugraph_ex synthetic reproduction now confirms partial copy functionalization expands to two full-buffer copies (details below). For absent speculative/MTP use, skipping the unused stash could also avoid allocation, subject to consumer inspection.
3. BF16 W13 paired shrink: routed BGMV shrink is 2.930 ms / step (129 = 43×3 calls); 2 of the 3 calls per layer read the same H4096 input for gate/up. Pairing can reuse input loads and remove one kernel launch, preserving FP32 accumulation and all inactive/adapter rules.
4. Shared expert clipped SwiGLU plus quant: 43 repeated small T8/I256 chains (two clamp slices, two clamps, multiplies, sigmoid, bias add, dynamic quant). Fusing this chain is distinct from routed SwiGLU, and needs its actual alpha/bias semantics.

No full-model improvement or output-quality conclusion is inferred from these sums. All figures are rank 0 only.

## MTP partial-copy investigation and synthetic validation

The original surviving rank0 compilation cache `{compile_cache}/fb20f96bb8/rank_0_0/backbone/computation_graph.py` was written September 30 at 01:32, shortly before the old profile. It annotates `l_self_mtp_hidden_buffer: bf16[16384,16384]` and captures the same model line1160 slice copy. The small exact excerpt is saved as `old_mtp_capture_excerpt.txt`.

`get_mtp_target_hidden_states()` is consumed in the Ascend model runner's speculative draft branch; only method `mtp` uses the returned residual there. For speculative_config=None this branch does not execute. Upstream consumers are likewise in speculative branches. A future no-spec guard could avoid the stash and 512 MiB allocation, but no production guard is applied by this analysis.

Physical device12 / visible0 was used for synthetic checks only. The buffer is BF16[16384,16384] and the source BF16[8,16384]. Every accepted case updates the source three times (+7, -3, +19), verifies all 8 updated rows exactly, and reduces equality over the **entire remaining 16376-row tail**, which must remain the initial constant5.

| Method | Graph median µs | Interpretation |
|---|---:|---|
| Plain NPUGraph partial slice copy | 9.32 | Correct, but does not reproduce compiled functionalization |
| Plain NPUGraph scatter_nd_update_ | 16.64 | Correct, but not faster than the plain copy |
| torch.func.functionalize(slice copy) | 2582.87 | Three full-buffer TensorMove tasks per replay |
| torch.func.functionalize(scatter_nd_update_) | 1736.78 | Two full-buffer TensorMove tasks per replay |
| Actual npugraph_ex compiled slice copy | **1726.29** | **Two** full-buffer TensorMove tasks per replay |
| Actual npugraph_ex compiled scatter_nd_update_ | **22.59** | One small ScatterNdUpdate task; no full-buffer TensorMove |

The last two use the same backend options as vLLM: force_eager=True, inplace_pass=False, clone_input=False, clone_output=False, then ordinary NPUGraph capture/replay. Both return `source+1` to exercise compiled outputs; that output also matches exactly after all three source changes. Their graph micro-path speedup is 76.43×. This measures only the isolated residual stash, **not full-model Decode or throughput**.

The isolated compiled profiler collected 3 replays per method: legacy has 6 TensorMove tasks (815.3–858.7 µs each), replacement has 3 ScatterNdUpdate tasks (9.14–10.70 µs each); each method has one small Adds task per replay for the verification output. Profiling reported an ACL-to-NPU-flow export warning, so no host-to-device correlation claim is made; device task counts, kernel durations, event timing, and complete buffer correctness are retained.

The real old profile's two TensorMove tasks total 1690.46 µs per Decode step, consistent with the actual compiled reproduction. Root cause is the compiled partial-view mutation, not the direct eager/native slice-copy itself. Generic torch.func functionalization alone does not predict the actual backend's treatment of opaque scatter, so the compiled comparison is authoritative here.

Reviewable next changes: (1) a no-speculative-config guard can avoid unused allocation and copy after confirming supported runner consumers; (2) for retained MTP stash, use `torch_npu.npu_scatter_nd_update_(buffer, rows[:,None], hidden_states.flatten(1))` with a persistent device integer row-index buffer sliced to num_tokens, because this exact backend keeps a true in-place update. Handle flashcomm allgather/padding semantics before the update. Full-model confirmation remains deferred because weights are absent. Production source was not edited.

All device12 test processes exited; `npu-smi info -t proc-mem -i 6 -c 0` reports `No process in device.` No devices0–7 were accessed.

## Production helper integration (2026-10-06)

`vllm_ascend/models/deepseek_v4.py` now has the reviewed candidate behind `VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY` (strict 0/1, default0; the root agent owns its central env registration and publication). When enabled, initialization retains persistent int64 row indices and `_copy_mtp_hidden_states()` performs a full-buffer in-place ScatterNdUpdate on only the active prefix. This adds 128 KiB at batch capacity16384. It preserves the 512 MiB residual buffer, address and unmodified tail; MTP functionality remains available independently of speculative mode.

Both SP and non-SP forward branches invoke this same helper. SP still all-gathers residual rows and removes trailing padding first. The legacy partial copy remains when disabled, or when dtype/device conversions fall outside the tested matched FP16/BF16 path. Empty inputs return without a native call; invalid rank, width or over-capacity input fails before mutation.

22 CPU unit tests pass, including actual initializer, helper and forward methods extracted from source to avoid model/weight loading. They cover both flags and dtypes, changed data, alias and tail, empty/capacity/rank/width, mixed-dtype fallback, both SP/non-SP branches and padding. ruff check and format checks pass for both source and tests.

The actual production helper also passes 8 NPU12 compiled graph cases using the real npugraph_ex options:

| dtype / active tokens | flag0 µs | flag1 µs |
|---|---:|---:|
| BF16 / 1 | 1721.97 | 13.55 |
| BF16 / 8 | 1725.08 | 21.78 |
| FP16 / 1 | 1721.39 | 14.07 |
| FP16 / 8 | 1728.42 | 22.35 |

Each case uses the real helper with a synthetic BF16/FP16[16384,16384] buffer, updates input three times and verifies the active head, full tail, pointer and compiled output exactly. Its profile confirms 2 full TensorMove tasks per legacy replay versus one small ScatterNdUpdate per enabled replay, with no full TensorMove in the enabled case. These are isolated helper figures, not a new full-model or throughput result. Root controls copying the reviewed source into the installed container repository.

Evidence: `verify_integrated_mtp_copy.py`, `mtp_copy_integrated_result.json`, `mtp_copy_integrated_kernel_summary.json`, and `mtp_copy_integrated_release.json`. NPU12 was released after testing.
