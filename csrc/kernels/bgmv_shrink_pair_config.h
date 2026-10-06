// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vllm_ascend {
constexpr uint32_t MOE_LORA_SHRINK_PAIR_HIDDEN = 4096;
constexpr uint32_t MOE_LORA_SHRINK_PAIR_RANK = 16;
constexpr uint32_t MOE_LORA_SHRINK_PAIR_MAX_ROWS = 384;
}  // namespace vllm_ascend
