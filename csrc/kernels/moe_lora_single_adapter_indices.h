// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#pragma once

#include <cstdint>

namespace vllm_ascend {

constexpr uint64_t MOE_LORA_SINGLE_ADAPTER_SMALL_WORK_BUDGET = 1024;
constexpr uint32_t MOE_LORA_SINGLE_ADAPTER_MAX_ADAPTERS = 64;
constexpr uint32_t MOE_LORA_SINGLE_ADAPTER_LARGE_TILE = 512;
constexpr uint64_t MOE_LORA_SINGLE_ADAPTER_LARGE_MIN_ROWS = 8161;
constexpr uint64_t MOE_LORA_SINGLE_ADAPTER_LARGE_MAX_ROWS = 98304;

// Metadata/resource predicates only. The producer must provide a complete,
// unique permutation in abs(expanded). Payloads remain signed integer bits.
bool moe_lora_single_adapter_indices_small_supported(
    uint64_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t indexBytes, uint32_t expertBytes, uint64_t ubBytes);
bool moe_lora_single_adapter_indices_large_supported(uint64_t rows, uint64_t ubBytes);

// Small routes inverse-scatter into one AIV's UB, then store int64[rows].
// The large variant needs disjoint int64[rows,8] scratch: each destination
// owns a 64-byte record, with its combined index in the first eight bytes.
// No input mutation, global scalar reads, or data-dependent host decisions.
void moe_lora_single_adapter_indices_small_impl(
    void* stream, void* expanded, void* topk, void* slots, void* enabled,
    void* output, uint32_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t adapters, uint64_t numExperts, uint32_t indexBytes, uint32_t expertBytes);
void moe_lora_single_adapter_indices_large_impl(
    void* stream, void* expanded, void* topk, void* slots, void* enabled,
    void* scratch, void* output, uint32_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t adapters, uint64_t numExperts, uint32_t indexBytes,
    uint32_t expertBytes, uint32_t cores);

}  // namespace vllm_ascend
