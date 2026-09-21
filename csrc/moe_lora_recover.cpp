// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <tuple>

#include "moe_lora_recover_validation.h"
#include "kernels/moe_lora_recover_sort.h"

// POD-only boundary to the SDK ABI bridge, queried under NPUGuard.
extern "C" uint64_t vllm_ascend_recover_ub_bytes();

namespace vllm_ascend {
namespace {

std::tuple<at::Tensor, at::Tensor> recover_original(const at::Tensor& expanded,
                                                   const at::Tensor& topk,
                                                   const at::Tensor& slots, int64_t top_k)
{
    // Preserve the original float32 sorting keys, order, integer floor divide
    // and upper clamp. This includes large shapes outside the native budget.
    const auto inverse = at::argsort(at::abs(expanded).to(at::kFloat));
    // Both gathers have one nonnegative, in-range, one-dimensional index.
    // index_select preserves every integer bit and avoids general advanced
    // indexing and its separate index-check device kernels.
    const auto experts = at::index_select(topk.reshape({-1}), 0, inverse).to(at::kLong);
    auto token = at::floor_divide(inverse, top_k);
    token.clamp_max_(slots.numel() - 1);
    return {experts, at::index_select(slots, 0, token)};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> moe_lora_recover(const at::Tensor& expanded,
                                                  const at::Tensor& topk,
                                                  const at::Tensor& slots, int64_t top_k)
{
    check_moe_lora_recover_metadata(expanded, topk, slots, top_k);
    TORCH_CHECK(expanded.device().type() == c10::DeviceType::PrivateUse1,
                "moe_lora_recover: expected NPU tensors");
    const auto options = expanded.options().dtype(at::kLong);
    const auto count = expanded.numel();
    if (count == 0) {
        return {at::empty({0}, options), at::empty({0}, options)};
    }
    const c10_npu::NPUGuard device_guard(expanded.device());
    const auto rows = static_cast<uint64_t>(count);
    const auto slot_count = static_cast<uint64_t>(slots.numel());
    const auto k = static_cast<uint64_t>(top_k);
    const auto index_bytes = static_cast<uint32_t>(expanded.element_size());
    const auto expert_bytes = static_cast<uint32_t>(topk.element_size());
    constexpr uint64_t max_exact_sort_rows = uint64_t{1} << 24;
    // Small inputs avoid sorting entirely. Larger complete permutations use
    // vector Sort/Gather when both the API and actual UB capacity allow it.
    // Inputs beyond those limits retain the original framework implementation.
    constexpr uint64_t scalar_work_budget = 1024;
    const bool scalar_work = rows + static_cast<uint64_t>(topk.size(0)) <= scalar_work_budget;
    if (rows > max_exact_sort_rows || (!scalar_work && rows > MOE_LORA_RECOVER_SORT_MAX_ROWS)) {
        return recover_original(expanded, topk, slots, top_k);
    }
    const auto ub_bytes = vllm_ascend_recover_ub_bytes();
    const bool use_small = scalar_work &&
        moe_lora_recover_small_supported(rows, k, slot_count, index_bytes, expert_bytes, ub_bytes);
    const bool use_sort = !use_small &&
        moe_lora_recover_sort_supported(rows, k, slot_count, index_bytes, expert_bytes, ub_bytes);
    if (!use_small && !use_sort) {
        return recover_original(expanded, topk, slots, top_k);
    }
    const auto launch = use_small ? moe_lora_recover_small_impl : moe_lora_recover_sort_impl;
    auto expert_out = at::empty({count}, options);
    auto slot_out = at::empty({count}, options);
    void* stream = c10_npu::getCurrentNPUStream().stream();
    void* expanded_ptr = expanded.data_ptr();
    void* topk_ptr = topk.data_ptr();
    void* slots_ptr = slots.data_ptr();
    void* expert_out_ptr = expert_out.data_ptr();
    void* slot_out_ptr = slot_out.data_ptr();
    at_npu::native::OpCommand command;
    command.Name("moe_lora_recover");
    command.SetCustomHandler([=]() -> int {
        launch(stream, expanded_ptr, topk_ptr, slots_ptr, expert_out_ptr, slot_out_ptr,
               rows, k, slot_count, index_bytes, expert_bytes, ub_bytes);
        return 0;
    });
    command.Run();
    return {expert_out, slot_out};
}

std::tuple<at::Tensor, at::Tensor> moe_lora_recover_meta(const at::Tensor& expanded,
                                                       const at::Tensor& topk,
                                                       const at::Tensor& slots, int64_t top_k)
{
    check_moe_lora_recover_metadata(expanded, topk, slots, top_k);
    const auto options = expanded.options().dtype(at::kLong);
    return {at::empty({expanded.size(0)}, options), at::empty({expanded.size(0)}, options)};
}

}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def("moe_lora_recover(Tensor expanded_row_idx, Tensor topk_ids, Tensor token_lora_indices, int top_k) -> (Tensor, Tensor)");
    ops.impl("moe_lora_recover", c10::DispatchKey::PrivateUse1, &vllm_ascend::moe_lora_recover);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)
{
    ops.impl("moe_lora_recover", &vllm_ascend::moe_lora_recover_meta);
}

#endif  // VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
