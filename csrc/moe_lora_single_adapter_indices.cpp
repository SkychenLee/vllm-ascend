// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS

#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <algorithm>

#include "kernels/moe_lora_single_adapter_indices.h"

extern "C" uint64_t vllm_ascend_recover_ub_bytes();
extern "C" uint64_t vllm_ascend_recover_vector_cores();

namespace vllm_ascend {
namespace {
bool integer(const at::Tensor& tensor) {
    return tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong;
}

void check_single_adapter_indices(const at::Tensor& expanded, const at::Tensor& topk,
    const at::Tensor& slots, const at::Tensor& enabled, int64_t topK, int64_t numExperts) {
    TORCH_CHECK(expanded.dim() == 1 && topk.dim() == 2 && slots.dim() == 1 && enabled.dim() == 1,
                "moe_lora_single_adapter_indices: expected expanded[R], topk[T,K], slots[L], enabled[A]");
    TORCH_CHECK(topK > 0 && topk.size(1) == topK && expanded.numel() == topk.numel(),
                "moe_lora_single_adapter_indices: routing dimensions and top_k must agree");
    TORCH_CHECK(integer(expanded) && integer(topk) && slots.scalar_type() == at::kLong &&
                enabled.scalar_type() == at::kInt,
                "moe_lora_single_adapter_indices: int32/int64 routing, int64 slots and int32 enabled required");
    TORCH_CHECK(expanded.device() == topk.device() && expanded.device() == slots.device() &&
                expanded.device() == enabled.device(),
                "moe_lora_single_adapter_indices: metadata must share a device");
    TORCH_CHECK(expanded.is_contiguous() && topk.is_contiguous() && slots.is_contiguous() && enabled.is_contiguous(),
                "moe_lora_single_adapter_indices: contiguous metadata required");
    TORCH_CHECK(numExperts > 0 && enabled.numel() > 0 &&
                enabled.numel() <= MOE_LORA_SINGLE_ADAPTER_MAX_ADAPTERS,
                "moe_lora_single_adapter_indices: positive num_experts and 1..64 enabled entries required");
    TORCH_CHECK(expanded.numel() == 0 || slots.numel() > 0,
                "moe_lora_single_adapter_indices: nonempty routing requires nonempty slots");
    const uint64_t rows = expanded.numel();
    const uint64_t tokens = topk.size(0);
    TORCH_CHECK(rows + tokens <= MOE_LORA_SINGLE_ADAPTER_SMALL_WORK_BUDGET ||
                (rows >= MOE_LORA_SINGLE_ADAPTER_LARGE_MIN_ROWS &&
                 rows <= MOE_LORA_SINGLE_ADAPTER_LARGE_MAX_ROWS),
                "moe_lora_single_adapter_indices: unsupported shape; use the established routing fallback");
    // Values are deliberately not copied to the host. The fully-sharded,
    // non-EP AllGather producer supplies unique abs(expanded) in [0,R).
    // Invalid destinations or duplicate indices are outside that contract.
}
}  // namespace

at::Tensor moe_lora_single_adapter_indices_meta(const at::Tensor& expanded, const at::Tensor& topk,
    const at::Tensor& slots, const at::Tensor& enabled, int64_t topK, int64_t numExperts) {
    check_single_adapter_indices(expanded, topk, slots, enabled, topK, numExperts);
    return at::empty(expanded.sizes(), expanded.options().dtype(at::kLong));
}

at::Tensor moe_lora_single_adapter_indices(const at::Tensor& expanded, const at::Tensor& topk,
    const at::Tensor& slots, const at::Tensor& enabled, int64_t topK, int64_t numExperts) {
    check_single_adapter_indices(expanded, topk, slots, enabled, topK, numExperts);
    TORCH_CHECK(expanded.device().type() == c10::DeviceType::PrivateUse1,
                "moe_lora_single_adapter_indices: NPU metadata required");
    const c10_npu::NPUGuard guard(expanded.device());
    // Fake/Meta validation never queries runtime NPU descriptors. Logical
    // contiguity alone cannot make blocked integer storage safe to read raw.
    for (const auto* tensor : {&expanded, &topk, &slots, &enabled}) {
        const auto format = at_npu::native::get_npu_format(*tensor);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "moe_lora_single_adapter_indices: ND/base storage (format 0 or 2) required");
    }
    auto output = at::empty(expanded.sizes(), expanded.options().dtype(at::kLong));
    const uint64_t rows = expanded.numel();
    if (rows == 0) return output;
    const uint64_t slotCount = slots.numel();
    const uint32_t indexBytes = expanded.element_size(), expertBytes = topk.element_size();
    const uint64_t ubBytes = vllm_ascend_recover_ub_bytes();
    const bool small = moe_lora_single_adapter_indices_small_supported(
        rows, topK, slotCount, indexBytes, expertBytes, ubBytes);
    const bool large = !small && moe_lora_single_adapter_indices_large_supported(rows, ubBytes);
    TORCH_CHECK(small || large,
                "moe_lora_single_adapter_indices: device UB budget cannot support this routing shape");
    const uint32_t adapters = enabled.numel();
    void* expandedPtr = expanded.data_ptr();
    void* topkPtr = topk.data_ptr();
    void* slotsPtr = slots.data_ptr();
    void* enabledPtr = enabled.data_ptr();
    void* outputPtr = output.data_ptr();
    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    at::Tensor scratch;
    uint32_t cores = 1;
    if (large) {
        const uint64_t available = vllm_ascend_recover_vector_cores();
        TORCH_CHECK(available > 0, "moe_lora_single_adapter_indices: cannot query vector cores");
        const uint64_t tiles = (rows + MOE_LORA_SINGLE_ADAPTER_LARGE_TILE - 1) /
                               MOE_LORA_SINGLE_ADAPTER_LARGE_TILE;
        cores = static_cast<uint32_t>(std::min(available, tiles));
        // A permutation gives exactly one writer per padded record. Padding
        // protects neighboring destinations from sub-cache-line DMA writes.
        scratch = at::empty({static_cast<int64_t>(rows), 8}, output.options());
    }
    void* scratchPtr = large ? scratch.data_ptr() : nullptr;
    at_npu::native::OpCommand command;
    command.Name("moe_lora_single_adapter_indices");
    command.SetCustomHandler([small, stream, expandedPtr, topkPtr, slotsPtr, enabledPtr,
                             scratchPtr, outputPtr, rows, topK, slotCount, adapters,
                             numExperts, indexBytes, expertBytes, cores]() -> int {
        // Keep this handler POD-only, like moe_lora_recover and bgmv. TorchNPU
        // retains COMPILE_AND_EXECUTE handlers until the execution-ring slot
        // is reused; releasing captured Tensor owners under its enqueue lock
        // can acquire the Python GIL and deadlock another event producer.
        // Locals remain alive through Run()'s enqueue. Same-stream allocator
        // reuse and FIFO submission order the launches before later writes;
        // the large scratch is used only by the two launches on this stream.
        if (small) {
            moe_lora_single_adapter_indices_small_impl(stream, expandedPtr, topkPtr, slotsPtr,
                enabledPtr, outputPtr, static_cast<uint32_t>(rows), topK, slotCount,
                adapters, numExperts, indexBytes, expertBytes);
        } else {
            moe_lora_single_adapter_indices_large_impl(stream, expandedPtr, topkPtr, slotsPtr,
                enabledPtr, scratchPtr, outputPtr, static_cast<uint32_t>(rows), topK, slotCount,
                adapters, numExperts, indexBytes, expertBytes, cores);
        }
        return 0;
    });
    command.Run();
    return output;
}
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("moe_lora_single_adapter_indices(Tensor expanded, Tensor topk, Tensor token_slots, Tensor enabled, int top_k, int num_experts) -> Tensor");
    ops.impl("moe_lora_single_adapter_indices", c10::DispatchKey::PrivateUse1,
             &vllm_ascend::moe_lora_single_adapter_indices);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
    ops.impl("moe_lora_single_adapter_indices", &vllm_ascend::moe_lora_single_adapter_indices_meta);
}
#endif
