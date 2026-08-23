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
#include "../../../kernels/types.h"

template <typename scalar_t, typename index_t, bool reuse_fp32_weight>
class MoeLoraBgmvFused {
public:
    using DataT = scalar_t;

    static constexpr uint32_t kRank = 16;
    static constexpr uint32_t kGroupRows = 8;
    static constexpr uint32_t kFallbackGroupRows = 4;
    static constexpr uint32_t kOutputTileElements = 512;
    static constexpr uint32_t kWeightTileElements =
        kOutputTileElements * kRank;
    static constexpr uint32_t kVectorBytes = 256;
    static constexpr uint32_t kFloatElementsPerRepeat =
        kVectorBytes / sizeof(float);
    static constexpr uint32_t kFloatElementsPerBlock =
        32 / sizeof(float);
    static constexpr uint32_t kBlocksPerRepeat = 8;
    static constexpr uint32_t kReduceTmpBytes = 256;

    __aicore__ inline explicit MoeLoraBgmvFused(AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR loraA,
        GM_ADDR loraB,
        GM_ADDR indices,
        GM_ADDR y,
        uint32_t numRows,
        uint32_t inputHiddenDim,
        uint32_t outputHiddenDim,
        uint32_t outputFullDim,
        uint32_t sliceOffset,
        uint32_t rowsPerCore,
        float scale)
    {
        numRows_ = numRows;
        inputHiddenDim_ = inputHiddenDim;
        outputHiddenDim_ = outputHiddenDim;
        outputFullDim_ = outputFullDim;
        sliceOffset_ = sliceOffset;
        rowsPerCore_ = rowsPerCore;
        scale_ = scale;
        singleAWeightElements_ =
            static_cast<uint64_t>(kRank) * inputHiddenDim_;
        singleBWeightElements_ =
            static_cast<uint64_t>(outputHiddenDim_) * kRank;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(x));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(loraA));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(loraB));
        indicesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ index_t*>(indices), numRows_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(y));

        uint32_t weightBufferElements = inputHiddenDim_;
        if (weightBufferElements < kWeightTileElements) {
            weightBufferElements = kWeightTileElements;
        }

        pipe_->InitBuffer(
            indicesQueue_, 1, kGroupRows * sizeof(index_t));
        pipe_->InitBuffer(
            xQueue_, 1,
            kGroupRows * inputHiddenDim_ * sizeof(DataT));
        pipe_->InitBuffer(
            weightQueue_, 1,
            weightBufferElements * sizeof(DataT));
        pipe_->InitBuffer(
            yInputQueue_, 1,
            kOutputTileElements * sizeof(DataT));
        pipe_->InitBuffer(
            yOutputQueue_, 1,
            kOutputTileElements * sizeof(DataT));

        pipe_->InitBuffer(
            xFp32Buffer_,
            kGroupRows * inputHiddenDim_ * sizeof(float));
        pipe_->InitBuffer(
            weightFp32Buffer_,
            weightBufferElements * sizeof(float));
        if constexpr (reuse_fp32_weight) {
            pipe_->InitBuffer(
                productFp32Buffer_,
                weightBufferElements * sizeof(float));
        }
        pipe_->InitBuffer(
            rankBuffer_, kGroupRows * kRank * sizeof(float));
        pipe_->InitBuffer(reduceTmpBuffer_, kReduceTmpBytes);
        pipe_->InitBuffer(
            yInputFp32Buffer_,
            kOutputTileElements * sizeof(float));
        pipe_->InitBuffer(
            yAccumFp32Buffer_,
            kOutputTileElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t row = blockIdx * rowsPerCore_;
        uint32_t endRow = row + rowsPerCore_;
        if (endRow > numRows_) {
            endRow = numRows_;
        }

        while (row < endRow) {
            const uint32_t remainingRows = endRow - row;
            const uint32_t currentRows =
                remainingRows >= kGroupRows ? kGroupRows : remainingRows;
            AscendC::LocalTensor<index_t> indicesLocal =
                CopyInIndices(row, currentRows);

            if (currentRows >= kFallbackGroupRows &&
                inputHiddenDim_ % kFloatElementsPerBlock == 0) {
                const int64_t index0 =
                    static_cast<int64_t>(indicesLocal.GetValue(0));
                const int64_t index1 =
                    static_cast<int64_t>(indicesLocal.GetValue(1));
                const int64_t index2 =
                    static_cast<int64_t>(indicesLocal.GetValue(2));
                const int64_t index3 =
                    static_cast<int64_t>(indicesLocal.GetValue(3));
                if (currentRows == kGroupRows) {
                    const int64_t index4 =
                        static_cast<int64_t>(indicesLocal.GetValue(4));
                    const int64_t index5 =
                        static_cast<int64_t>(indicesLocal.GetValue(5));
                    const int64_t index6 =
                        static_cast<int64_t>(indicesLocal.GetValue(6));
                    const int64_t index7 =
                        static_cast<int64_t>(indicesLocal.GetValue(7));
                    if (index0 == index1 && index0 == index2 &&
                        index0 == index3 && index0 == index4 &&
                        index0 == index5 && index0 == index6 &&
                        index0 == index7) {
                        indicesQueue_.FreeTensor(indicesLocal);
                        if (index0 >= 0) {
                            ProcessGroup<kGroupRows>(
                                row, static_cast<uint64_t>(index0));
                        }
                        row += kGroupRows;
                        continue;
                    }
                }
                if (index0 == index1 && index0 == index2 &&
                    index0 == index3) {
                    indicesQueue_.FreeTensor(indicesLocal);
                    if (index0 >= 0) {
                        ProcessGroup<kFallbackGroupRows>(
                            row, static_cast<uint64_t>(index0));
                    }
                    row += kFallbackGroupRows;
                    continue;
                }

                indicesQueue_.FreeTensor(indicesLocal);
                if (index0 >= 0) {
                    ProcessGroup<1>(row, static_cast<uint64_t>(index0));
                }
                row += 1;
                continue;
            }

            for (uint32_t localRow = 0; localRow < currentRows; ++localRow) {
                const int64_t weightIndex = static_cast<int64_t>(
                    indicesLocal.GetValue(localRow));
                if (weightIndex >= 0) {
                    ProcessGroup<1>(
                        row + localRow,
                        static_cast<uint64_t>(weightIndex));
                }
            }
            indicesQueue_.FreeTensor(indicesLocal);
            row += currentRows;
        }
    }

private:
    __aicore__ inline AscendC::LocalTensor<index_t> CopyInIndices(
        uint32_t row,
        uint32_t rows)
    {
        AscendC::LocalTensor<index_t> indicesLocal =
            indicesQueue_.AllocTensor<index_t>();
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(rows * sizeof(index_t)),
            0,
            0,
            0};
        AscendC::DataCopyPadExtParams<index_t> padParams{
            false, 0, 0, static_cast<index_t>(0)};
        AscendC::DataCopyPad(
            indicesLocal, indicesGm_[row], copyParams, padParams);
        indicesQueue_.EnQue(indicesLocal);
        return indicesQueue_.DeQue<index_t>();
    }

    template <uint32_t Rows>
    __aicore__ inline void ProcessGroup(
        uint32_t startRow,
        uint64_t weightIndex)
    {
        CopyInX<Rows>(startRow);
        ComputeShrink<Rows>(weightIndex);
        ComputeExpand<Rows>(startRow, weightIndex);
    }

    template <uint32_t Rows>
    __aicore__ inline void CopyInX(uint32_t startRow)
    {
        constexpr uint32_t groupRows = Rows;
        const uint32_t inputElements = groupRows * inputHiddenDim_;
        AscendC::LocalTensor<DataT> xLocal =
            xQueue_.AllocTensor<DataT>();
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(inputElements * sizeof(DataT)),
            0,
            0,
            0};
        AscendC::DataCopyPadExtParams<DataT> padParams{
            false, 0, 0, static_cast<DataT>(0)};
        AscendC::DataCopyPad(
            xLocal,
            xGm_[static_cast<uint64_t>(startRow) * inputHiddenDim_],
            copyParams,
            padParams);
        xQueue_.EnQue(xLocal);

        xLocal = xQueue_.DeQue<DataT>();
        AscendC::LocalTensor<float> xFp32 = xFp32Buffer_.Get<float>();
        AscendC::Cast(
            xFp32,
            xLocal,
            AscendC::RoundMode::CAST_NONE,
            inputElements);
        AscendC::PipeBarrier<PIPE_V>();
        xQueue_.FreeTensor(xLocal);
    }

    template <uint32_t Rows>
    __aicore__ inline void ComputeShrink(uint64_t weightIndex)
    {
        AscendC::LocalTensor<float> xFp32 = xFp32Buffer_.Get<float>();
        AscendC::LocalTensor<float> weightFp32 =
            weightFp32Buffer_.Get<float>();
        AscendC::LocalTensor<float> productFp32 = weightFp32;
        if constexpr (reuse_fp32_weight) {
            productFp32 = productFp32Buffer_.Get<float>();
        }
        AscendC::LocalTensor<float> rankLocal = rankBuffer_.Get<float>();
        AscendC::LocalTensor<float> reduceTmp =
            reduceTmpBuffer_.Get<float>();
        const uint64_t weightBase = weightIndex * singleAWeightElements_;

        for (uint32_t rank = 0; rank < kRank; ++rank) {
            AscendC::LocalTensor<DataT> weightLocal =
                weightQueue_.AllocTensor<DataT>();
            AscendC::DataCopyExtParams copyParams{
                1,
                static_cast<uint32_t>(inputHiddenDim_ * sizeof(DataT)),
                0,
                0,
                0};
            AscendC::DataCopyPadExtParams<DataT> padParams{
                false, 0, 0, static_cast<DataT>(0)};
            AscendC::DataCopyPad(
                weightLocal,
                aGm_[weightBase +
                    static_cast<uint64_t>(rank) * inputHiddenDim_],
                copyParams,
                padParams);
            weightQueue_.EnQue(weightLocal);
            weightLocal = weightQueue_.DeQue<DataT>();

            if constexpr (reuse_fp32_weight) {
                AscendC::Cast(
                    weightFp32,
                    weightLocal,
                    AscendC::RoundMode::CAST_NONE,
                    inputHiddenDim_);
                AscendC::PipeBarrier<PIPE_V>();
            }

            for (uint32_t localRow = 0; localRow < Rows; ++localRow) {
                if constexpr (!reuse_fp32_weight) {
                    AscendC::Cast(
                        weightFp32,
                        weightLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputHiddenDim_);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Mul(
                    productFp32,
                    xFp32[localRow * inputHiddenDim_],
                    weightFp32,
                    inputHiddenDim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceSum<float>(
                    rankLocal[localRow * kRank + rank],
                    productFp32,
                    reduceTmp,
                    inputHiddenDim_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            weightQueue_.FreeTensor(weightLocal);
        }

        AscendC::Muls(
            rankLocal,
            rankLocal,
            scale_,
            Rows * kRank);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <uint32_t Rows>
    __aicore__ inline void ComputeExpand(
        uint32_t startRow,
        uint64_t weightIndex)
    {
        const uint64_t weightBase = weightIndex * singleBWeightElements_;
        AscendC::LocalTensor<float> rankLocal = rankBuffer_.Get<float>();
        AscendC::LocalTensor<float> rankDup = xFp32Buffer_.Get<float>();
        AscendC::LocalTensor<float> weightFp32 =
            weightFp32Buffer_.Get<float>();
        AscendC::LocalTensor<float> productFp32 = weightFp32;
        if constexpr (reuse_fp32_weight) {
            productFp32 = productFp32Buffer_.Get<float>();
        }
        AscendC::LocalTensor<float> yInputFp32 =
            yInputFp32Buffer_.Get<float>();
        AscendC::LocalTensor<float> yAccumFp32 =
            yAccumFp32Buffer_.Get<float>();

        for (uint32_t outputBegin = 0;
             outputBegin < outputHiddenDim_;
             outputBegin += kOutputTileElements) {
            uint32_t outputElements = outputHiddenDim_ - outputBegin;
            if (outputElements > kOutputTileElements) {
                outputElements = kOutputTileElements;
            }
            const uint32_t weightElements = outputElements * kRank;
            AscendC::LocalTensor<DataT> weightLocal =
                weightQueue_.AllocTensor<DataT>();
            AscendC::DataCopyExtParams weightCopyParams{
                1,
                static_cast<uint32_t>(weightElements * sizeof(DataT)),
                0,
                0,
                0};
            AscendC::DataCopyPadExtParams<DataT> weightPadParams{
                false, 0, 0, static_cast<DataT>(0)};
            AscendC::DataCopyPad(
                weightLocal,
                bGm_[weightBase +
                    static_cast<uint64_t>(outputBegin) * kRank],
                weightCopyParams,
                weightPadParams);
            weightQueue_.EnQue(weightLocal);
            weightLocal = weightQueue_.DeQue<DataT>();
            if constexpr (reuse_fp32_weight) {
                CastWeight(weightFp32, weightLocal, weightElements);
            }

            for (uint32_t localRow = 0; localRow < Rows; ++localRow) {
                if constexpr (!reuse_fp32_weight) {
                    CastWeight(weightFp32, weightLocal, weightElements);
                }
                DuplicateRank(
                    rankDup,
                    rankLocal[localRow * kRank]);
                ExpandDot(
                    yAccumFp32,
                    productFp32,
                    rankDup,
                    weightFp32,
                    weightElements);
                AddAndCopyOut(
                    startRow + localRow,
                    outputBegin,
                    outputElements,
                    yAccumFp32,
                    yInputFp32);
            }
            weightQueue_.FreeTensor(weightLocal);
        }
    }

    __aicore__ inline void DuplicateRank(
        AscendC::LocalTensor<float> rankDup,
        AscendC::LocalTensor<float> rankSource)
    {
        constexpr uint8_t repeatTime =
            kFloatElementsPerRepeat / kRank;
        constexpr uint16_t dstRepeatStride =
            kRank / kFloatElementsPerBlock;
        AscendC::Copy(
            rankDup,
            rankSource,
            kRank,
            repeatTime,
            {1, 1, dstRepeatStride, 0});
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CastWeight(
        AscendC::LocalTensor<float> weightFp32,
        AscendC::LocalTensor<DataT> weightLocal,
        uint32_t weightElements)
    {
        const uint32_t fullRepeats =
            weightElements / kFloatElementsPerRepeat;
        const uint32_t tailElements =
            weightElements % kFloatElementsPerRepeat;
        if (fullRepeats != 0) {
            AscendC::Cast(
                weightFp32,
                weightLocal,
                AscendC::RoundMode::CAST_NONE,
                kFloatElementsPerRepeat,
                static_cast<uint8_t>(fullRepeats),
                castParams_);
        }
        if (tailElements != 0) {
            const uint32_t tailOffset =
                fullRepeats * kFloatElementsPerRepeat;
            AscendC::Cast(
                weightFp32[tailOffset],
                weightLocal[tailOffset],
                AscendC::RoundMode::CAST_NONE,
                tailElements);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExpandDot(
        AscendC::LocalTensor<float> output,
        AscendC::LocalTensor<float> productFp32,
        AscendC::LocalTensor<float> rankDup,
        AscendC::LocalTensor<float> weightFp32,
        uint32_t weightElements)
    {
        const uint32_t fullRepeats =
            weightElements / kFloatElementsPerRepeat;
        const uint32_t tailElements =
            weightElements % kFloatElementsPerRepeat;
        if (fullRepeats != 0) {
            AscendC::Mul(
                productFp32,
                rankDup,
                weightFp32,
                kFloatElementsPerRepeat,
                static_cast<uint8_t>(fullRepeats),
                dotProductParams_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (tailElements != 0) {
            const uint32_t tailOffset =
                fullRepeats * kFloatElementsPerRepeat;
            AscendC::Mul(
                productFp32[tailOffset],
                rankDup,
                weightFp32[tailOffset],
                tailElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        const uint32_t blockReduceRepeats =
            (weightElements + kFloatElementsPerRepeat - 1) /
            kFloatElementsPerRepeat;
        const uint32_t paddedWeightElements =
            blockReduceRepeats * kFloatElementsPerRepeat;
        if (paddedWeightElements > weightElements) {
            AscendC::Duplicate(
                productFp32[weightElements],
                0.0f,
                paddedWeightElements - weightElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::BlockReduceSum(
            productFp32,
            productFp32,
            static_cast<uint8_t>(blockReduceRepeats),
            kFloatElementsPerRepeat,
            reduceSumParams_.dstRepStride,
            reduceSumParams_.srcBlkStride,
            reduceSumParams_.srcRepStride);
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t blockOutputs =
            blockReduceRepeats * kBlocksPerRepeat;
        const uint32_t pairReduceRepeats =
            (blockOutputs + kFloatElementsPerRepeat - 1) /
            kFloatElementsPerRepeat;
        const uint32_t paddedBlockOutputs =
            pairReduceRepeats * kFloatElementsPerRepeat;
        if (paddedBlockOutputs > blockOutputs) {
            AscendC::Duplicate(
                productFp32[blockOutputs],
                0.0f,
                paddedBlockOutputs - blockOutputs);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::PairReduceSum(
            output,
            productFp32,
            static_cast<uint8_t>(pairReduceRepeats),
            kFloatElementsPerRepeat,
            reduceSumParams_.dstRepStride,
            reduceSumParams_.srcBlkStride,
            reduceSumParams_.srcRepStride);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddAndCopyOut(
        uint32_t row,
        uint32_t outputBegin,
        uint32_t outputElements,
        AscendC::LocalTensor<float> yAccumFp32,
        AscendC::LocalTensor<float> yInputFp32)
    {
        const uint64_t yOffset =
            static_cast<uint64_t>(row) * outputFullDim_ +
            sliceOffset_ + outputBegin;
        AscendC::LocalTensor<DataT> yInputLocal =
            yInputQueue_.AllocTensor<DataT>();
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(outputElements * sizeof(DataT)),
            0,
            0,
            0};
        AscendC::DataCopyPadExtParams<DataT> padParams{
            false, 0, 0, static_cast<DataT>(0)};
        AscendC::DataCopyPad(
            yInputLocal, yGm_[yOffset], copyParams, padParams);
        yInputQueue_.EnQue(yInputLocal);
        yInputLocal = yInputQueue_.DeQue<DataT>();

        AscendC::Cast(
            yInputFp32,
            yInputLocal,
            AscendC::RoundMode::CAST_NONE,
            outputElements);
        AscendC::PipeBarrier<PIPE_V>();
        yInputQueue_.FreeTensor(yInputLocal);
        AscendC::Add(
            yAccumFp32,
            yAccumFp32,
            yInputFp32,
            outputElements);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<DataT> yOutputLocal =
            yOutputQueue_.AllocTensor<DataT>();
        AscendC::Cast(
            yOutputLocal,
            yAccumFp32,
            AscendC::RoundMode::CAST_RINT,
            outputElements);
        AscendC::PipeBarrier<PIPE_V>();
        yOutputQueue_.EnQue(yOutputLocal);
        yOutputLocal = yOutputQueue_.DeQue<DataT>();
        AscendC::DataCopyPad(
            yGm_[yOffset], yOutputLocal, copyParams);
        yOutputQueue_.FreeTensor(yOutputLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> indicesQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> weightQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> yInputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> yOutputQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xFp32Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> weightFp32Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> productFp32Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rankBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceTmpBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> yInputFp32Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> yAccumFp32Buffer_;

    AscendC::GlobalTensor<DataT> xGm_;
    AscendC::GlobalTensor<DataT> aGm_;
    AscendC::GlobalTensor<DataT> bGm_;
    AscendC::GlobalTensor<index_t> indicesGm_;
    AscendC::GlobalTensor<DataT> yGm_;

    uint32_t numRows_;
    uint32_t inputHiddenDim_;
    uint32_t outputHiddenDim_;
    uint32_t outputFullDim_;
    uint32_t sliceOffset_;
    uint32_t rowsPerCore_;
    float scale_;
    uint64_t singleAWeightElements_;
    uint64_t singleBWeightElements_;

    AscendC::UnaryRepeatParams castParams_ = {1, 1, 8, 4};
    AscendC::UnaryRepeatParams reduceSumParams_ = {1, 1, 1, 8};
    AscendC::BinaryRepeatParams dotProductParams_ =
        {1, 1, 1, 8, 0, 8};
};

#define MOE_LORA_BGMV_FUSED_DECLARE(                                         \
    TYPE, INDEX_TYPE, INDEX_SUFFIX, REUSE, REUSE_SUFFIX)                     \
    extern "C" __global__ __aicore__ void                                   \
        moe_lora_bgmv_fused_##TYPE##_##INDEX_SUFFIX##_##REUSE_SUFFIX(        \
            GM_ADDR x, GM_ADDR loraA, GM_ADDR loraB, GM_ADDR indices,        \
            GM_ADDR y, uint32_t numRows, uint32_t inputHiddenDim,            \
            uint32_t outputHiddenDim, uint32_t outputFullDim,                \
            uint32_t sliceOffset, uint32_t rowsPerCore, float scale)         \
    {                                                                         \
        AscendC::TPipe pipe;                                                  \
        MoeLoraBgmvFused<TYPE, INDEX_TYPE, REUSE> op(&pipe);                 \
        op.Init(x, loraA, loraB, indices, y, numRows, inputHiddenDim,        \
                outputHiddenDim, outputFullDim, sliceOffset, rowsPerCore,    \
                scale);                                                       \
        op.Process();                                                         \
    }

MOE_LORA_BGMV_FUSED_DECLARE(half, int32_t, int32, false, inplace)
MOE_LORA_BGMV_FUSED_DECLARE(half, int32_t, int32, true, reuse)
MOE_LORA_BGMV_FUSED_DECLARE(half, int64_t, int64, false, inplace)
MOE_LORA_BGMV_FUSED_DECLARE(half, int64_t, int64, true, reuse)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
MOE_LORA_BGMV_FUSED_DECLARE(bfloat16_t, int32_t, int32, false, inplace)
MOE_LORA_BGMV_FUSED_DECLARE(bfloat16_t, int32_t, int32, true, reuse)
MOE_LORA_BGMV_FUSED_DECLARE(bfloat16_t, int64_t, int64, false, inplace)
MOE_LORA_BGMV_FUSED_DECLARE(bfloat16_t, int64_t, int64, true, reuse)
#endif

namespace vllm_ascend {

namespace {

constexpr uint32_t kFp32ReuseMinRows = 2048;
constexpr uint32_t kInplaceHiddenDim = 2048;

}  // namespace

extern void moe_lora_bgmv_fused_impl(
    AscendType type,
    void* stream,
    void* x,
    void* loraA,
    void* loraB,
    void* indices,
    void* y,
    uint32_t numRows,
    uint32_t inputHiddenDim,
    uint32_t outputHiddenDim,
    uint32_t outputFullDim,
    uint32_t sliceOffset,
    uint32_t rowsPerCore,
    float scale,
    bool indicesIsInt32)
{
    const uint32_t blockDim =
        (numRows + rowsPerCore - 1) / rowsPerCore;
    const bool reuseFp32Weight =
        numRows >= kFp32ReuseMinRows ||
        inputHiddenDim < kInplaceHiddenDim;
    if (type == AscendType::FP16) {
        if (indicesIsInt32) {
            if (reuseFp32Weight) {
                moe_lora_bgmv_fused_half_int32_reuse
                    <<<blockDim, nullptr, stream>>>(
                        x, loraA, loraB, indices, y, numRows, inputHiddenDim,
                        outputHiddenDim, outputFullDim, sliceOffset,
                        rowsPerCore, scale);
            } else {
                moe_lora_bgmv_fused_half_int32_inplace
                    <<<blockDim, nullptr, stream>>>(
                        x, loraA, loraB, indices, y, numRows, inputHiddenDim,
                        outputHiddenDim, outputFullDim, sliceOffset,
                        rowsPerCore, scale);
            }
        } else {
            if (reuseFp32Weight) {
                moe_lora_bgmv_fused_half_int64_reuse
                    <<<blockDim, nullptr, stream>>>(
                        x, loraA, loraB, indices, y, numRows, inputHiddenDim,
                        outputHiddenDim, outputFullDim, sliceOffset,
                        rowsPerCore, scale);
            } else {
                moe_lora_bgmv_fused_half_int64_inplace
                    <<<blockDim, nullptr, stream>>>(
                        x, loraA, loraB, indices, y, numRows, inputHiddenDim,
                        outputHiddenDim, outputFullDim, sliceOffset,
                        rowsPerCore, scale);
            }
        }
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        if (indicesIsInt32) {
            if (reuseFp32Weight) {
                moe_lora_bgmv_fused_bfloat16_t_int32_reuse
                    <<<blockDim, nullptr, stream>>>(
                        x, loraA, loraB, indices, y, numRows, inputHiddenDim,
                        outputHiddenDim, outputFullDim, sliceOffset,
                        rowsPerCore, scale);
            } else {
                moe_lora_bgmv_fused_bfloat16_t_int32_inplace
                    <<<blockDim, nullptr, stream>>>(
                        x, loraA, loraB, indices, y, numRows, inputHiddenDim,
                        outputHiddenDim, outputFullDim, sliceOffset,
                        rowsPerCore, scale);
            }
        } else {
            if (reuseFp32Weight) {
                moe_lora_bgmv_fused_bfloat16_t_int64_reuse
                    <<<blockDim, nullptr, stream>>>(
                        x, loraA, loraB, indices, y, numRows, inputHiddenDim,
                        outputHiddenDim, outputFullDim, sliceOffset,
                        rowsPerCore, scale);
            } else {
                moe_lora_bgmv_fused_bfloat16_t_int64_inplace
                    <<<blockDim, nullptr, stream>>>(
                        x, loraA, loraB, indices, y, numRows, inputHiddenDim,
                        outputHiddenDim, outputFullDim, sliceOffset,
                        rowsPerCore, scale);
            }
        }
#endif
    }
}

}  // namespace vllm_ascend
