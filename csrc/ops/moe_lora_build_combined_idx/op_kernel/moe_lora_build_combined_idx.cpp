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

class MoeLoraBuildCombinedIdx {
public:
    __aicore__ inline void Init(
        GM_ADDR expandedRowIdx,
        GM_ADDR topkIds,
        GM_ADDR tokenLoraIndices,
        GM_ADDR adapterEnabled,
        GM_ADDR combinedIdx,
        uint32_t numPairs,
        uint32_t topK,
        uint32_t numExperts,
        uint32_t tileLength,
        uint32_t outputPairsPerCore,
        AscendC::TPipe* pipe)
    {
        expandedRowIdxGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(expandedRowIdx), numPairs);
        topkIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(topkIds), numPairs);
        tokenLoraIndicesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t*>(tokenLoraIndices));
        adapterEnabledGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(adapterEnabled));
        combinedIdxGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(combinedIdx), numPairs);

        numPairs_ = numPairs;
        topK_ = topK;
        numExperts_ = numExperts;
        tileLength_ = tileLength;
        outputPairsPerCore_ = outputPairsPerCore;

        if (tileLength_ != 0) {
            tileAlignedLength_ = AlignInt32(tileLength_);
            outputBegin_ = AscendC::GetBlockIdx() * outputPairsPerCore_;
            outputLength_ = numPairs_ - outputBegin_;
            if (outputLength_ > outputPairsPerCore_) {
                outputLength_ = outputPairsPerCore_;
            }
            pipe->InitBuffer(
                inputBuf_, 2 * tileAlignedLength_ * sizeof(int32_t));
            pipe->InitBuffer(
                outputBuf_, AlignInt32(outputLength_) * sizeof(int32_t));
        }
    }

    __aicore__ inline void Process()
    {
        if (tileLength_ == 0) {
            ProcessScalarFallback();
        } else {
            ProcessUb();
        }
    }

private:
    __aicore__ inline static uint32_t AlignInt32(uint32_t length)
    {
        constexpr uint32_t elementsPerBlock = 32 / sizeof(int32_t);
        return (length + elementsPerBlock - 1) /
            elementsPerBlock * elementsPerBlock;
    }

    template <AscendC::HardEvent event>
    __aicore__ inline static void SyncPipes()
    {
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(eventId);
        AscendC::WaitFlag<event>(eventId);
    }

    __aicore__ inline void ProcessUb()
    {
        AscendC::LocalTensor<int32_t> inputLocal = inputBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> expandedLocal = inputLocal;
        AscendC::LocalTensor<int32_t> expertLocal =
            inputLocal[tileAlignedLength_];
        AscendC::LocalTensor<int32_t> outputLocal = outputBuf_.Get<int32_t>();

        for (uint32_t begin = 0; begin < numPairs_; begin += tileLength_) {
            uint32_t count = numPairs_ - begin;
            if (count > tileLength_) {
                count = tileLength_;
            }

            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> padParams{
                false, 0, 0, 0};
            AscendC::DataCopyPad(
                expandedLocal, expandedRowIdxGm_[begin], copyParams, padParams);
            AscendC::DataCopyPad(
                expertLocal, topkIdsGm_[begin], copyParams, padParams);
            SyncPipes<AscendC::HardEvent::MTE2_S>();

            for (uint32_t localPair = 0; localPair < count; ++localPair) {
                int32_t sortedRow = expandedLocal.GetValue(localPair);
                if (sortedRow < 0) {
                    sortedRow = -sortedRow;
                }
                if (sortedRow < static_cast<int32_t>(outputBegin_) ||
                    sortedRow >= static_cast<int32_t>(
                        outputBegin_ + outputLength_)) {
                    continue;
                }

                const uint32_t pair = begin + localPair;
                const int64_t lora =
                    tokenLoraIndicesGm_.GetValue(pair / topK_);
                int32_t combined = -1;
                if (lora >= 0 &&
                    adapterEnabledGm_.GetValue(lora) != 0) {
                    combined = static_cast<int32_t>(
                        lora * static_cast<int64_t>(numExperts_) +
                        expertLocal.GetValue(localPair));
                }
                outputLocal.SetValue(sortedRow - outputBegin_, combined);
            }

            if (begin + count < numPairs_) {
                SyncPipes<AscendC::HardEvent::S_MTE2>();
            }
        }

        SyncPipes<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(outputLength_ * sizeof(int32_t)),
            0,
            0,
            0};
        AscendC::DataCopyPad(
            combinedIdxGm_[outputBegin_], outputLocal, copyParams);
    }

    __aicore__ inline void ProcessScalarFallback()
    {
        for (uint32_t pair = 0; pair < numPairs_; ++pair) {
            int32_t sortedRow = expandedRowIdxGm_.GetValue(pair);
            if (sortedRow < 0) {
                sortedRow = -sortedRow;
            }

            const int64_t lora = tokenLoraIndicesGm_.GetValue(pair / topK_);
            int32_t combined = -1;
            if (lora >= 0 && adapterEnabledGm_.GetValue(lora) != 0) {
                const int32_t expert = topkIdsGm_.GetValue(pair);
                combined = static_cast<int32_t>(
                    lora * static_cast<int64_t>(numExperts_) + expert);
            }
            combinedIdxGm_.SetValue(sortedRow, combined);
        }
    }

    AscendC::GlobalTensor<int32_t> expandedRowIdxGm_;
    AscendC::GlobalTensor<int32_t> topkIdsGm_;
    AscendC::GlobalTensor<int64_t> tokenLoraIndicesGm_;
    AscendC::GlobalTensor<int8_t> adapterEnabledGm_;
    AscendC::GlobalTensor<int32_t> combinedIdxGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf_;
    uint32_t numPairs_;
    uint32_t topK_;
    uint32_t numExperts_;
    uint32_t tileLength_;
    uint32_t tileAlignedLength_;
    uint32_t outputPairsPerCore_;
    uint32_t outputBegin_;
    uint32_t outputLength_;
};

extern "C" __global__ __aicore__ void moe_lora_build_combined_idx(
    GM_ADDR expandedRowIdx,
    GM_ADDR topkIds,
    GM_ADDR tokenLoraIndices,
    GM_ADDR adapterEnabled,
    GM_ADDR combinedIdx,
    uint32_t numPairs,
    uint32_t topK,
    uint32_t numExperts,
    uint32_t tileLength,
    uint32_t outputPairsPerCore)
{
    AscendC::TPipe pipe;
    MoeLoraBuildCombinedIdx op;
    op.Init(
        expandedRowIdx,
        topkIds,
        tokenLoraIndices,
        adapterEnabled,
        combinedIdx,
        numPairs,
        topK,
        numExperts,
        tileLength,
        outputPairsPerCore,
        &pipe);
    op.Process();
}

namespace vllm_ascend {

extern void moe_lora_build_combined_idx_impl(
    void* stream,
    void* expandedRowIdx,
    void* topkIds,
    void* tokenLoraIndices,
    void* adapterEnabled,
    void* combinedIdx,
    uint32_t numPairs,
    uint32_t topK,
    uint32_t numExperts,
    uint32_t tileLength,
    uint32_t outputPairsPerCore,
    uint32_t blockDim)
{
    moe_lora_build_combined_idx<<<blockDim, nullptr, stream>>>(
        expandedRowIdx,
        topkIds,
        tokenLoraIndices,
        adapterEnabled,
        combinedIdx,
        numPairs,
        topK,
        numExperts,
        tileLength,
        outputPairsPerCore);
}

}  // namespace vllm_ascend
