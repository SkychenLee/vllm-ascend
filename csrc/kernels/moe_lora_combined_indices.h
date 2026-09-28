// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

constexpr uint32_t MOE_LORA_COMBINED_TILE_ROWS = 128;
constexpr uint32_t MOE_LORA_COMBINED_MAX_ADAPTERS = 64;

void moe_lora_combined_indices_impl(void* stream, void* experts, void* slots,
    void* enabled, void* output, uint64_t rows, uint64_t num_experts,
    uint32_t adapters, uint32_t expert_bytes, uint32_t cores);
