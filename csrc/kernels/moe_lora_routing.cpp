/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"

// AllGather decode previously inverted expanded_row_idx with torch.argsort.
// Integer ArgSort runs on AICPU on A2 and dominates the entire MoE LoRA path.
// Decode metadata is small, so one vector core can invert the permutation and
// fold (LoRA slot, expert) into the final BGMV index in a single launch.
template <typename index_t, typename enabled_t>
class MoeLoraRouting {
public:
    static constexpr uint32_t MAX_ROWS = 1024;
    static constexpr uint32_t MAX_ADAPTERS = 64;

    __aicore__ inline MoeLoraRouting(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void* expandedRowIdx, __gm__ void* topkIds,
                                __gm__ void* tokenLoraIndices, __gm__ void* adapterEnabled,
                                __gm__ void* combinedIndices, uint32_t numRows, uint32_t numTokens,
                                uint32_t numAdapters, uint32_t topK, uint32_t numExperts)
    {
        numRows_ = numRows;
        numTokens_ = numTokens;
        numAdapters_ = numAdapters;
        topK_ = topK;
        numExperts_ = numExperts;
        expandedGm_.SetGlobalBuffer((__gm__ index_t*)expandedRowIdx, numRows);
        topkGm_.SetGlobalBuffer((__gm__ index_t*)topkIds, numRows);
        tokenLoraGm_.SetGlobalBuffer((__gm__ int64_t*)tokenLoraIndices, numTokens);
        adapterEnabledGm_.SetGlobalBuffer((__gm__ enabled_t*)adapterEnabled, numAdapters);
        combinedGm_.SetGlobalBuffer((__gm__ int64_t*)combinedIndices, numRows);

        pipe_->InitBuffer(expandedQueue_, 1, MAX_ROWS * sizeof(index_t));
        pipe_->InitBuffer(topkQueue_, 1, MAX_ROWS * sizeof(index_t));
        pipe_->InitBuffer(tokenLoraQueue_, 1, MAX_ROWS * sizeof(int64_t));
        pipe_->InitBuffer(adapterEnabledQueue_, 1, MAX_ADAPTERS * sizeof(enabled_t));
        pipe_->InitBuffer(combinedQueue_, 1, MAX_ROWS * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<index_t> expanded = expandedQueue_.AllocTensor<index_t>();
        AscendC::LocalTensor<index_t> topk = topkQueue_.AllocTensor<index_t>();
        AscendC::LocalTensor<int64_t> tokenLora = tokenLoraQueue_.AllocTensor<int64_t>();
        AscendC::LocalTensor<enabled_t> adapterEnabled = adapterEnabledQueue_.AllocTensor<enabled_t>();
        DataCopyPad(expanded, expandedGm_,
                    {1, static_cast<uint16_t>(numRows_ * sizeof(index_t)), 0, 0}, {});
        DataCopyPad(topk, topkGm_,
                    {1, static_cast<uint16_t>(numRows_ * sizeof(index_t)), 0, 0}, {});
        DataCopyPad(tokenLora, tokenLoraGm_,
                    {1, static_cast<uint16_t>(numTokens_ * sizeof(int64_t)), 0, 0}, {});
        DataCopyPad(adapterEnabled, adapterEnabledGm_,
                    {1, static_cast<uint16_t>(numAdapters_ * sizeof(enabled_t)), 0, 0}, {});
        expandedQueue_.EnQue(expanded);
        topkQueue_.EnQue(topk);
        tokenLoraQueue_.EnQue(tokenLora);
        adapterEnabledQueue_.EnQue(adapterEnabled);

        expanded = expandedQueue_.DeQue<index_t>();
        topk = topkQueue_.DeQue<index_t>();
        tokenLora = tokenLoraQueue_.DeQue<int64_t>();
        adapterEnabled = adapterEnabledQueue_.DeQue<enabled_t>();
        AscendC::LocalTensor<int64_t> combined = combinedQueue_.AllocTensor<int64_t>();

        for (uint32_t originalRow = 0; originalRow < numRows_; ++originalRow) {
            int64_t destination = static_cast<int64_t>(expanded.GetValue(originalRow));
            destination = destination < 0 ? -destination : destination;
            const int64_t loraIdx = tokenLora.GetValue(originalRow / topK_);
            int64_t combinedIdx = -1;
            if (loraIdx >= 0 && loraIdx < numAdapters_ &&
                static_cast<int64_t>(adapterEnabled.GetValue(loraIdx)) != 0) {
                const int64_t expertIdx = static_cast<int64_t>(topk.GetValue(originalRow));
                combinedIdx = loraIdx * numExperts_ + expertIdx;
            }
            combined.SetValue(destination, combinedIdx);
        }

        combinedQueue_.EnQue(combined);
        combined = combinedQueue_.DeQue<int64_t>();
        DataCopyPad(combinedGm_, combined,
                    {1, static_cast<uint16_t>(numRows_ * sizeof(int64_t)), 0, 0});

        expandedQueue_.FreeTensor(expanded);
        topkQueue_.FreeTensor(topk);
        tokenLoraQueue_.FreeTensor(tokenLora);
        adapterEnabledQueue_.FreeTensor(adapterEnabled);
        combinedQueue_.FreeTensor(combined);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> expandedQueue_, topkQueue_, tokenLoraQueue_, adapterEnabledQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> combinedQueue_;
    AscendC::GlobalTensor<index_t> expandedGm_, topkGm_;
    AscendC::GlobalTensor<int64_t> tokenLoraGm_, combinedGm_;
    AscendC::GlobalTensor<enabled_t> adapterEnabledGm_;
    uint32_t numRows_;
    uint32_t numTokens_;
    uint32_t numAdapters_;
    uint32_t topK_;
    uint32_t numExperts_;
};

#define MOE_LORA_ROUTING_DECLARE(INDEX_TYPE, INDEX_NAME, ENABLED_TYPE, ENABLED_NAME)                                  \
    extern "C" __global__ __aicore__ void moe_lora_routing_##INDEX_NAME##_##ENABLED_NAME(                           \
        __gm__ void* expandedRowIdx, __gm__ void* topkIds, __gm__ void* tokenLoraIndices,                            \
        __gm__ void* adapterEnabled, __gm__ void* combinedIndices, uint32_t numRows, uint32_t numTokens,             \
        uint32_t numAdapters, uint32_t topK, uint32_t numExperts)                                                     \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                            \
        MoeLoraRouting<INDEX_TYPE, ENABLED_TYPE> op(&pipe);                                                            \
        op.Init(expandedRowIdx, topkIds, tokenLoraIndices, adapterEnabled, combinedIndices, numRows, numTokens,       \
                numAdapters, topK, numExperts);                                                                         \
        op.Process();                                                                                                   \
    }

MOE_LORA_ROUTING_DECLARE(int32_t, int32, bool, bool)
MOE_LORA_ROUTING_DECLARE(int32_t, int32, int32_t, int32)
MOE_LORA_ROUTING_DECLARE(int32_t, int32, int64_t, int64)
MOE_LORA_ROUTING_DECLARE(int64_t, int64, bool, bool)
MOE_LORA_ROUTING_DECLARE(int64_t, int64, int32_t, int32)
MOE_LORA_ROUTING_DECLARE(int64_t, int64, int64_t, int64)

namespace vllm_ascend {
extern void moe_lora_routing_impl(void* stream, void* expandedRowIdx, void* topkIds,
                                  void* tokenLoraIndices, void* adapterEnabled, void* combinedIndices,
                                  uint32_t numRows, uint32_t numTokens, uint32_t numAdapters,
                                  uint32_t topK, uint32_t numExperts, bool index64, uint32_t enabledType)
{
    if (!index64 && enabledType == 0) {
        moe_lora_routing_int32_bool<<<1, nullptr, stream>>>(expandedRowIdx, topkIds, tokenLoraIndices,
                                                           adapterEnabled, combinedIndices, numRows, numTokens,
                                                           numAdapters, topK, numExperts);
    } else if (!index64 && enabledType == 1) {
        moe_lora_routing_int32_int32<<<1, nullptr, stream>>>(expandedRowIdx, topkIds, tokenLoraIndices,
                                                            adapterEnabled, combinedIndices, numRows, numTokens,
                                                            numAdapters, topK, numExperts);
    } else if (!index64) {
        moe_lora_routing_int32_int64<<<1, nullptr, stream>>>(expandedRowIdx, topkIds, tokenLoraIndices,
                                                            adapterEnabled, combinedIndices, numRows, numTokens,
                                                            numAdapters, topK, numExperts);
    } else if (enabledType == 0) {
        moe_lora_routing_int64_bool<<<1, nullptr, stream>>>(expandedRowIdx, topkIds, tokenLoraIndices,
                                                           adapterEnabled, combinedIndices, numRows, numTokens,
                                                           numAdapters, topK, numExperts);
    } else if (enabledType == 1) {
        moe_lora_routing_int64_int32<<<1, nullptr, stream>>>(expandedRowIdx, topkIds, tokenLoraIndices,
                                                            adapterEnabled, combinedIndices, numRows, numTokens,
                                                            numAdapters, topK, numExperts);
    } else {
        moe_lora_routing_int64_int64<<<1, nullptr, stream>>>(expandedRowIdx, topkIds, tokenLoraIndices,
                                                            adapterEnabled, combinedIndices, numRows, numTokens,
                                                            numAdapters, topK, numExperts);
    }
}
}  // namespace vllm_ascend
