// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#pragma once

#include <cstdint>

constexpr uint32_t MOE_LORA_RECOVER_LARGE_TILE = 512;
constexpr uint64_t MOE_LORA_RECOVER_LARGE_MIN_ROWS = 8161;
// Bound the per-forward padded workspace to 16 MiB. Larger routes retain
// the established framework path instead of allocating unbounded scratch.
constexpr uint64_t MOE_LORA_RECOVER_LARGE_MAX_ROWS = (uint64_t{1} << 24) / 64;

bool moe_lora_recover_large_supported(uint64_t rows, uint64_t ubBytes);
void moe_lora_recover_large_impl(void* stream, void* expanded, void* expert, void* slots, void* scratch,
                                 void* expertOut, void* slotOut, uint32_t rows, uint64_t topK, uint64_t slotCount,
                                 uint32_t indexBytes, uint32_t expertBytes, uint32_t cores);
