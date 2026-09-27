// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#pragma once
#include <cstdint>
#include "types.h"

namespace vllm_ascend {
void bgmv_shrink_int8_impl(AscendType type, void* stream, void* x, void* weight, void* indices, void* scale, void* out,
                           uint32_t rows, uint32_t hidden, uint32_t rank, uint32_t groups, uint32_t cores);
void moe_lora_expand_swiglu_quant_impl(AscendType type, void* stream, void* base, void* gate, void* up, void* bg,
                                       void* bu, void* indices, void* topk, void* out, void* scale, uint32_t rows,
                                       uint32_t hidden, uint32_t rank, uint32_t groups, uint32_t cores, float limit);
void bgmv_shrink_int8_pair_impl(AscendType type, void* stream, void* x, void* weight, void* indices, void* scale,
                                void* out, uint32_t rows, uint32_t hidden, uint32_t rank, uint32_t groups,
                                uint32_t cores);
void moe_lora_expand_swiglu_quant_pair_impl(AscendType type, void* stream, void* base, void* a, void* bg, void* bu,
                                            void* indices, void* topk, void* out, void* scale, uint32_t rows,
                                            uint32_t hidden, uint32_t rank, uint32_t groups, uint32_t cores,
                                            float limit, uint32_t tp);
void bgmv_shrink_int8_pair_grouped_impl(AscendType type, void* stream, void* x, void* w, void* ids, void* order,
                                  void* scale, void* y, uint32_t rows, uint32_t hidden, uint32_t rank,
                                  uint32_t groups, uint32_t cores);
void moe_lora_expand_swiglu_quant_pair_grouped_impl(AscendType type, void* stream, void* base, void* a,
    void* bg, void* bu, void* ids, void* order, void* topk, void* y, void* scale, uint32_t rows,
    uint32_t hidden, uint32_t rank, uint32_t groups, uint32_t cores, float limit, uint32_t tp);
void bgmv_expand_grouped_impl(AscendType type, void* stream, void* x, void* w, void* ids, void* order, void* y,
    uint32_t rows, uint32_t rank, uint32_t width, uint32_t full_width, uint32_t offset, uint32_t groups, uint32_t cores);
}  // namespace vllm_ascend
