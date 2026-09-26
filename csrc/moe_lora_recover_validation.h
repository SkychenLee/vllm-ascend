// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <ATen/ATen.h>
#include <cstdint>

namespace vllm_ascend {

// Resource predicate and kernel supplied by the independently built native library.
bool moe_lora_recover_small_supported(uint64_t rows, uint64_t topK, uint64_t slotCount,
                                      uint32_t expandedElementBytes, uint32_t expertElementBytes,
                                      uint64_t ubBytes);
void moe_lora_recover_small_impl(void* stream, void* expanded, void* topk, void* slots,
                                 void* expertOut, void* slotOut, uint64_t rows, uint64_t topK,
                                 uint64_t slotCount, uint32_t expandedElementBytes,
                                 uint32_t expertElementBytes, uint64_t ubBytes);

inline void check_moe_lora_recover_metadata(const at::Tensor& expanded, const at::Tensor& topk,
                                          const at::Tensor& slots, int64_t top_k)
{
    TORCH_CHECK(expanded.dim() == 1 && topk.dim() == 2 && slots.dim() == 1,
                "moe_lora_recover: expected expanded[M], topk[T,K], slots[L]");
    TORCH_CHECK(top_k > 0 && topk.size(1) == top_k && topk.numel() == expanded.numel(),
                "moe_lora_recover: top_k and routing dimensions must agree");
    TORCH_CHECK((expanded.scalar_type() == at::kInt || expanded.scalar_type() == at::kLong) &&
                (topk.scalar_type() == at::kInt || topk.scalar_type() == at::kLong) &&
                slots.scalar_type() == at::kLong,
                "moe_lora_recover: expanded/topk must be int32 or int64, slots int64");
    TORCH_CHECK(expanded.device() == topk.device() && expanded.device() == slots.device(),
                "moe_lora_recover: tensors must be on the same device");
    TORCH_CHECK(expanded.is_contiguous() && topk.is_contiguous() && slots.is_contiguous(),
                "moe_lora_recover: tensors must be contiguous");
    TORCH_CHECK(expanded.numel() == 0 || slots.numel() > 0,
                "moe_lora_recover: nonempty routing requires nonempty slots");
    // Values are not read here. The ordinary caller provides a complete
    // permutation of [0,M); the EP caller may also provide repeated -1 values
    // for remote experts and padded rows, with unique valid destinations.
}

}  // namespace vllm_ascend
