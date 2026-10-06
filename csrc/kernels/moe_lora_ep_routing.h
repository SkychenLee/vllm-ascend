/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cstdint>

namespace vllm_ascend {
constexpr uint32_t MOE_LORA_EP_ROUTING_MAX_PAIRS = 384;
constexpr uint32_t MOE_LORA_EP_ROUTING_MAX_EXPERTS = 512;

// The producer supplies unique nonnegative destinations. A single vector core
// performs the inverse scatter in UB, then copies the compact outputs to GM.
void moe_lora_ep_routing_impl(void* stream, void* expanded, void* topk,
    void* slots, void* expertMap, void* expertOut, void* slotOut,
    uint32_t pairs, uint32_t capacity, uint32_t topK, uint32_t experts,
    uint32_t expandedBytes, uint32_t topkBytes, uint32_t mapBytes);
}  // namespace vllm_ascend
