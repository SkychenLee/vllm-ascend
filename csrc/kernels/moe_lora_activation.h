// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "types.h"

namespace vllm_ascend {
// Decode specialization: every vector core processes independent whole rows.
// Scale stores use exact four-byte DataCopyPad transfers, as DynamicQuant's
// single-row kernel does. Empty inputs do not launch a kernel.
constexpr uint32_t MOE_LORA_ACTIVATION_WIDTH = 2048;
constexpr uint32_t MOE_LORA_ACTIVATION_MAX_ROWS = 384;
void moe_lora_activation_impl(AscendType type, void* stream, void* gate_up,
    void* activated, void* quantized, void* scales, uint32_t rows,
    uint32_t cores, float limit);
}  // namespace vllm_ascend
