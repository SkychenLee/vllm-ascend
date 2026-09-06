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

class MoeLoraPrepareAllGatherBgmvIndices {
public:
    __aicore__ inline MoeLoraPrepareAllGatherBgmvIndices(AscendC::TPipe* pipe)
        : pipe_(pipe)
    {
    }

    __aicore__ inline void Init(
        __gm__ void* expandedRowIdx, __gm__ void* topkIds,
        __gm__ void* tokenLoraIndices, __gm__ void* expertMap,
        __gm__ void* adapterEnabled, __gm__ void* output,
        uint32_t numPairs, uint32_t numTokens, uint32_t topK,
        uint32_t numGlobalExperts, uint32_t numLocalExperts,
        uint32_t numLoras)
    {
        numPairs_ = numPairs;
        numTokens_ = numTokens;
        topK_ = topK;
        numGlobalExperts_ = numGlobalExperts;
        numLocalExperts_ = numLocalExperts;
        numLoras_ = numLoras;

        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, numPairs);
        topkIdsGm_.SetGlobalBuffer((__gm__ int32_t*)topkIds, numPairs);
        tokenLoraIndicesGm_.SetGlobalBuffer(
            (__gm__ int64_t*)tokenLoraIndices, numTokens);
        expertMapGm_.SetGlobalBuffer((__gm__ int32_t*)expertMap, numGlobalExperts);
        adapterEnabledGm_.SetGlobalBuffer((__gm__ int32_t*)adapterEnabled, numLoras);
        outputGm_.SetGlobalBuffer((__gm__ int64_t*)output, numPairs);

        pipe_->InitBuffer(expandedRowIdxBuf_, AlignBytes(numPairs * sizeof(int32_t)));
        pipe_->InitBuffer(topkIdsBuf_, AlignBytes(numPairs * sizeof(int32_t)));
        pipe_->InitBuffer(
            tokenLoraIndicesBuf_, AlignBytes(numTokens * sizeof(int64_t)));
        pipe_->InitBuffer(expertMapBuf_, AlignBytes(numGlobalExperts * sizeof(int32_t)));
        pipe_->InitBuffer(adapterEnabledBuf_, AlignBytes(numLoras * sizeof(int32_t)));
        pipe_->InitBuffer(outputBuf_, AlignBytes(numPairs * sizeof(int64_t)));
    }

    __aicore__ inline void Process()
    {
        CopyIn();

        AscendC::LocalTensor<int32_t> expandedRowIdx =
            expandedRowIdxBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> topkIds = topkIdsBuf_.Get<int32_t>();
        AscendC::LocalTensor<int64_t> tokenLoraIndices =
            tokenLoraIndicesBuf_.Get<int64_t>();
        AscendC::LocalTensor<int32_t> expertMap = expertMapBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> adapterEnabled =
            adapterEnabledBuf_.Get<int32_t>();
        AscendC::LocalTensor<int64_t> output = outputBuf_.Get<int64_t>();

        // Preserve the validated AllGather recover contract. expanded_row_idx
        // maps each original (token, top-k) pair to its expert-major
        // destination. Non-local pairs are ignored before their undefined
        // destination is inspected. Local destinations are unique, so this
        // one-core scatter needs neither atomics nor intermediate tensors.
        for (uint32_t row = 0; row < numPairs_; ++row) {
            output.SetValue(row, static_cast<int64_t>(-1));
        }
        for (uint32_t pair = 0; pair < numPairs_; ++pair) {
            int32_t globalExpert = topkIds.GetValue(pair);
            if (globalExpert < 0 ||
                static_cast<uint32_t>(globalExpert) >= numGlobalExperts_) {
                continue;
            }
            int32_t localExpert = expertMap.GetValue(globalExpert);
            if (localExpert < 0 ||
                static_cast<uint32_t>(localExpert) >= numLocalExperts_) {
                continue;
            }

            int64_t loraSlot = tokenLoraIndices.GetValue(pair / topK_);
            if (loraSlot < 0 || static_cast<uint64_t>(loraSlot) >= numLoras_ ||
                adapterEnabled.GetValue(loraSlot) == 0) {
                continue;
            }

            int32_t destination = expandedRowIdx.GetValue(pair);
            if (destination < 0) {
                destination = -destination;
            }
            if (static_cast<uint32_t>(destination) >= numPairs_) {
                destination = static_cast<int32_t>(numPairs_ - 1);
            }
            int64_t combined =
                loraSlot * static_cast<int64_t>(numLocalExperts_) + localExpert;
            output.SetValue(destination, combined);
        }

        event_t eventIdSToMte3 =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::DataCopyExtParams outputCopyParams{
            1, static_cast<uint32_t>(numPairs_ * sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm_, output, outputCopyParams);
    }

private:
    __aicore__ inline static uint32_t AlignBytes(uint32_t bytes)
    {
        constexpr uint32_t alignment = 32;
        return (bytes + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline void CopyIn()
    {
        AscendC::DataCopyPad(
            expandedRowIdxBuf_.Get<int32_t>(), expandedRowIdxGm_,
            {1, static_cast<uint32_t>(numPairs_ * sizeof(int32_t)), 0, 0, 0},
            {true, 0, 0, 0});
        AscendC::DataCopyPad(
            topkIdsBuf_.Get<int32_t>(), topkIdsGm_,
            {1, static_cast<uint32_t>(numPairs_ * sizeof(int32_t)), 0, 0, 0},
            {true, 0, 0, 0});
        AscendC::DataCopyPad(
            tokenLoraIndicesBuf_.Get<int64_t>(), tokenLoraIndicesGm_,
            {1, static_cast<uint32_t>(numTokens_ * sizeof(int64_t)), 0, 0, 0},
            {true, 0, 0, 0});
        AscendC::DataCopyPad(
            expertMapBuf_.Get<int32_t>(), expertMapGm_,
            {1, static_cast<uint32_t>(numGlobalExperts_ * sizeof(int32_t)), 0, 0, 0},
            {true, 0, 0, 0});
        AscendC::DataCopyPad(
            adapterEnabledBuf_.Get<int32_t>(), adapterEnabledGm_,
            {1, static_cast<uint32_t>(numLoras_ * sizeof(int32_t)), 0, 0, 0},
            {true, 0, 0, 0});
        event_t eventIdMte2ToS =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::GlobalTensor<int32_t> expandedRowIdxGm_;
    AscendC::GlobalTensor<int32_t> topkIdsGm_;
    AscendC::GlobalTensor<int64_t> tokenLoraIndicesGm_;
    AscendC::GlobalTensor<int32_t> expertMapGm_;
    AscendC::GlobalTensor<int32_t> adapterEnabledGm_;
    AscendC::GlobalTensor<int64_t> outputGm_;
    AscendC::TBuf<AscendC::TPosition::VECIN> expandedRowIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> topkIdsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> tokenLoraIndicesBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> expertMapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> adapterEnabledBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outputBuf_;
    uint32_t numPairs_;
    uint32_t numTokens_;
    uint32_t topK_;
    uint32_t numGlobalExperts_;
    uint32_t numLocalExperts_;
    uint32_t numLoras_;
};

extern "C" __global__ __aicore__ void moe_lora_prepare_allgather_bgmv_indices(
    __gm__ void* expandedRowIdx, __gm__ void* topkIds,
    __gm__ void* tokenLoraIndices, __gm__ void* expertMap,
    __gm__ void* adapterEnabled, __gm__ void* output,
    uint32_t numPairs, uint32_t numTokens, uint32_t topK,
    uint32_t numGlobalExperts, uint32_t numLocalExperts,
    uint32_t numLoras)
{
    AscendC::TPipe pipe;
    MoeLoraPrepareAllGatherBgmvIndices op(&pipe);
    op.Init(expandedRowIdx, topkIds, tokenLoraIndices, expertMap,
            adapterEnabled, output, numPairs, numTokens, topK,
            numGlobalExperts, numLocalExperts, numLoras);
    op.Process();
}

namespace vllm_ascend {
extern void moe_lora_prepare_allgather_bgmv_indices_impl(
    void* stream, void* expandedRowIdx, void* topkIds,
    void* tokenLoraIndices, void* expertMap, void* adapterEnabled,
    void* output, uint32_t numPairs, uint32_t numTokens, uint32_t topK,
    uint32_t numGlobalExperts, uint32_t numLocalExperts,
    uint32_t numLoras)
{
    constexpr uint32_t blockDim = 1;
    moe_lora_prepare_allgather_bgmv_indices<<<blockDim, nullptr, stream>>>(
        expandedRowIdx, topkIds, tokenLoraIndices, expertMap, adapterEnabled,
        output, numPairs, numTokens, topK, numGlobalExperts,
        numLocalExperts, numLoras);
}
}  // namespace vllm_ascend
