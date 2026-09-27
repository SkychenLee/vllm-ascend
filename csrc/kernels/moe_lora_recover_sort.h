// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_MOE_LORA_RECOVER_SORT_H
#define VLLM_ASCEND_MOE_LORA_RECOVER_SORT_H

#include <cstdint>

namespace vllm_ascend {
// The DAV2201 Sort API accepts at most 255 groups of 32 elements.
constexpr uint64_t MOE_LORA_RECOVER_SORT_MAX_ROWS = 32 * 255;
// Same complete-permutation and contiguous integer tensor contract as the
// small kernel. Resource limits are independent of model, hidden size and rank.
bool moe_lora_recover_sort_supported(uint64_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t indexBytes, uint32_t expertBytes, uint64_t ubBytes);
void moe_lora_recover_sort_impl(void* stream, void* expanded, void* topk, void* slots,
    void* expertOut, void* slotOut, uint64_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t indexBytes, uint32_t expertBytes, uint64_t ubBytes);
} // namespace vllm_ascend
#endif
