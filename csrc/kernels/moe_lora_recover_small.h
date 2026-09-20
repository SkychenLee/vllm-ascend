/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VLLM_ASCEND_MOE_LORA_RECOVER_SMALL_H
#define VLLM_ASCEND_MOE_LORA_RECOVER_SMALL_H

#include <cstdint>

namespace vllm_ascend {

// The caller supplies the UB capacity of the guarded input device. This checks
// metadata/resources only, never the data-dependent permutation precondition.
bool moe_lora_recover_small_supported(uint64_t rows, uint64_t topK, uint64_t slotCount,
                                      uint32_t expandedElementBytes, uint32_t expertElementBytes,
                                      uint64_t ubBytes);

// Preconditions: supported(...) is true; inputs are contiguous signed integer
// tensors of the given widths, slots are int64, outputs are disjoint contiguous
// int64[rows], and abs(expanded) is a permutation of [0, rows). No mutation of
// inputs, no GM scalar accesses, and no data-dependent host synchronization.
// rows==0 performs no device launch. Unsupported metadata performs no launch;
// the binding must choose the original framework fallback using the predicate.
void moe_lora_recover_small_impl(void* stream, void* expanded, void* topk, void* slots,
                                void* expertOut, void* slotOut, uint64_t rows, uint64_t topK,
                                uint64_t slotCount, uint32_t expandedElementBytes,
                                uint32_t expertElementBytes, uint64_t ubBytes);

} // namespace vllm_ascend
#endif
