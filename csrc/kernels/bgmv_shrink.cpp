/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "types.h"
#include <cstdint>

namespace {
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t VECTOR_BYTES = 256;
constexpr uint32_t MAX_VECTOR_REPEAT = 255;
constexpr uint32_t MAX_VECTOR_REPEAT_STRIDE = 255;
constexpr uint32_t BATCHED_UB_BYTES = 128 * 1024;
// Three FP32 output buffers can each require up to one block of padding.
constexpr uint32_t OUTPUT_ALIGNMENT_RESERVE = 3 * BLOCK_BYTES;
}

template <typename scalar_t, bool BATCHED_RANK = false>
class BGMVShrink {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = float;

    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t TILE_LENGTH = 11776;
    static constexpr uint32_t INPUT_BLOCK_ELEMENTS = BLOCK_BYTES / sizeof(X_T);
    static constexpr uint32_t OUTPUT_BLOCK_ELEMENTS = BLOCK_BYTES / sizeof(Y_T);
    static constexpr uint32_t VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(float);

public:
    __aicore__ inline BGMVShrink(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(__gm__ void *x, __gm__ void *weight, __gm__ void *indices, uint32_t indicesSize, __gm__ void *y,
                                uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim,
                                uint32_t maxLoRARank, float scale)
    {
        batchSize_ =  batchSize;
        numTokensPerCore_ = numTokensPerCore;
        inputHiddenDim_ = inputHiddenDim;
        maxLoRARank_ = maxLoRARank;
        scale_ = scale;
        singleLoRAWeightLen_ = static_cast<uint64_t>(inputHiddenDim_) * maxLoRARank_;
        incremental_ = inputHiddenDim_ > TILE_LENGTH;

        xGm_.SetGlobalBuffer((__gm__ X_T *)x);
        yOutGm_.SetGlobalBuffer((__gm__ Y_T *)y);
        wGm_.SetGlobalBuffer((__gm__ W_T *)weight);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, indicesSize);

        if constexpr (BATCHED_RANK) {
            paddedInputDim_ = (inputHiddenDim_ + INPUT_BLOCK_ELEMENTS - 1) / INPUT_BLOCK_ELEMENTS * INPUT_BLOCK_ELEMENTS;
            const uint32_t rowBytes = paddedInputDim_ * (sizeof(W_T) + sizeof(float));
            rankTile_ = (BATCHED_UB_BYTES - rowBytes - OUTPUT_ALIGNMENT_RESERVE) / (rowBytes + 3 * sizeof(float));
            rankTile_ = rankTile_ < MAX_VECTOR_REPEAT ? rankTile_ : MAX_VECTOR_REPEAT;
            rankTile_ = rankTile_ < maxLoRARank_ ? rankTile_ : maxLoRARank_;
            if (rankTile_ >= OUTPUT_BLOCK_ELEMENTS) {
                rankTile_ = rankTile_ / OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
            }
            const uint32_t outputBytes = (rankTile_ + OUTPUT_BLOCK_ELEMENTS - 1) / OUTPUT_BLOCK_ELEMENTS * BLOCK_BYTES;
            pipe_->InitBuffer(inQueueX_, BUFFER_NUM, paddedInputDim_ * sizeof(X_T));
            pipe_->InitBuffer(inQueueW_, BUFFER_NUM, rankTile_ * paddedInputDim_ * sizeof(W_T));
            pipe_->InitBuffer(tmpBufferX_, paddedInputDim_ * sizeof(float));
            pipe_->InitBuffer(tmpBufferW_, rankTile_ * paddedInputDim_ * sizeof(float));
            pipe_->InitBuffer(outQueueY_, 1, outputBytes);
            pipe_->InitBuffer(outBufferY_, outputBytes);
            pipe_->InitBuffer(tailBufferY_, outputBytes);
        } else {
            pipe_->InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(X_T));
            pipe_->InitBuffer(inQueueW_, BUFFER_NUM, TILE_LENGTH * sizeof(W_T));
            pipe_->InitBuffer(tmpBufferX_, TILE_LENGTH * sizeof(float));
            pipe_->InitBuffer(tmpBufferW_, TILE_LENGTH * sizeof(float));
            pipe_->InitBuffer(outQueueY_, 1, maxLoRARank_ * sizeof(Y_T));
            pipe_->InitBuffer(outBufferY_, maxLoRARank_ * sizeof(float));
        }
    }

    __aicore__ inline void Process(const uint32_t localBlockIdx)
    {
        int64_t startIdx = static_cast<int64_t>(localBlockIdx) * numTokensPerCore_;
        int64_t endIdx = startIdx + numTokensPerCore_;
        if (endIdx > batchSize_) {
            endIdx = batchSize_;
        }
        for (int64_t idx = startIdx; idx < endIdx; idx++) {
            // set up LoRA index
            CopyInIndex(idx);
            if (reqLoRAIndex_ < 0) {
                continue;
            }
            reqLoRAWeightOffset_ = reqLoRAIndex_ * singleLoRAWeightLen_;

            if constexpr (BATCHED_RANK) {
                ProcessBatched(idx);
            } else {
                if (incremental_) {
                    ProcessImpl<true>(idx);
                } else {
                    ProcessImpl<false>(idx);
                }
                ScaleOutput();
                CopyOut(idx);
            }
        }
    }

private:
    __aicore__ inline void ProcessBatched(const int64_t idx)
    {
        const uint8_t rowStride = paddedInputDim_ * sizeof(float) / BLOCK_BYTES;
        const AscendC::BinaryRepeatParams mulParams{1, 1, 1, rowStride, 0, rowStride};
        const uint8_t rightPadding = paddedInputDim_ - inputHiddenDim_;
        auto xLocal = inQueueX_.AllocTensor<X_T>();
        const AscendC::DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(inputHiddenDim_ * sizeof(X_T)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<X_T> padParams{true, 0, rightPadding, 0};
        if (paddedInputDim_ == inputHiddenDim_) {
            DataCopy(xLocal, xGm_[inputHiddenDim_ * idx], inputHiddenDim_);
        } else {
            DataCopyPad(xLocal, xGm_[inputHiddenDim_ * idx], xCopyParams, padParams);
        }
        inQueueX_.EnQue(xLocal);
        xLocal = inQueueX_.DeQue<X_T>();
        auto xFp32 = tmpBufferX_.Get<float>();
        auto wFp32 = tmpBufferW_.Get<float>();
        auto yLocal = outBufferY_.Get<float>();
        auto tailLocal = tailBufferY_.Get<float>();
        Cast(xFp32, xLocal, AscendC::RoundMode::CAST_NONE, paddedInputDim_);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueX_.FreeTensor(xLocal);

        for (uint32_t rankStart = 0; rankStart < maxLoRARank_; rankStart += rankTile_) {
            const uint32_t ranksRemaining = maxLoRARank_ - rankStart;
            const uint8_t rankCount = ranksRemaining < rankTile_ ? ranksRemaining : rankTile_;
            auto wLocal = inQueueW_.AllocTensor<W_T>();
            // GM rows are tightly packed; DataCopyPad aligns each UB row to a
            // block, so arbitrary input widths have the same vector layout.
            const AscendC::DataCopyExtParams wCopyParams{rankCount, static_cast<uint32_t>(inputHiddenDim_ * sizeof(W_T)), 0, 0, 0};
            const uint64_t weightOffset = reqLoRAWeightOffset_ + static_cast<uint64_t>(rankStart) * inputHiddenDim_;
            if (paddedInputDim_ == inputHiddenDim_) {
                DataCopy(wLocal, wGm_[weightOffset], rankCount * inputHiddenDim_);
            } else {
                DataCopyPad(wLocal, wGm_[weightOffset], wCopyParams, padParams);
            }
            inQueueW_.EnQue(wLocal);
            wLocal = inQueueW_.DeQue<W_T>();
            Cast(wFp32, wLocal, AscendC::RoundMode::CAST_NONE, rankCount * paddedInputDim_);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueW_.FreeTensor(wLocal);

            for (uint32_t col = 0; col < inputHiddenDim_; col += VECTOR_ELEMENTS) {
                const uint32_t remaining = inputHiddenDim_ - col;
                const uint64_t mask = remaining < VECTOR_ELEMENTS ? remaining : VECTOR_ELEMENTS;
                Mul(wFp32[col], xFp32[col], wFp32[col], mask, rankCount, mulParams);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < inputHiddenDim_; col += VECTOR_ELEMENTS) {
                const uint32_t remaining = inputHiddenDim_ - col;
                const uint64_t mask = remaining < VECTOR_ELEMENTS ? remaining : VECTOR_ELEMENTS;
                // Destination repeat stride is in elements; source strides
                // are in blocks. All rank results stay in vector registers/UB.
                if (col == 0) {
                    WholeReduceSum(yLocal, wFp32[col], mask, rankCount, 1, 1, rowStride);
                } else {
                    WholeReduceSum(tailLocal, wFp32[col], mask, rankCount, 1, 1, rowStride);
                    AscendC::PipeBarrier<PIPE_V>();
                    Add(yLocal, yLocal, tailLocal, rankCount);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            auto yOutLocal = outQueueY_.AllocTensor<Y_T>();
            Muls(yOutLocal, yLocal, scale_, rankCount);
            AscendC::PipeBarrier<PIPE_V>();
            outQueueY_.EnQue(yOutLocal);
            yOutLocal = outQueueY_.DeQue<Y_T>();
            const AscendC::DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(rankCount * sizeof(Y_T)), 0, 0, 0};
            DataCopyPad(yOutGm_[maxLoRARank_ * idx + rankStart], yOutLocal, yCopyParams);
            outQueueY_.FreeTensor(yOutLocal);
        }
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ProcessImpl(const int64_t idx)
    {
        AscendC::LocalTensor<float> yOutLocal = outBufferY_.Get<float>();
        if constexpr (!INCREMENTAL_MODE) {
            CopyInX(idx, 0, inputHiddenDim_);
            AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, inputHiddenDim_);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueX_.FreeTensor(xLocal);
        }

        for (int i = 0; i < maxLoRARank_; i++) {
            float acc(0);
            for (int32_t j = 0; j < inputHiddenDim_ / TILE_LENGTH; j++) {
                if constexpr (INCREMENTAL_MODE) {
                    CopyInX(idx, j);
                }
                CopyInW(i, j);
                Compute<INCREMENTAL_MODE>(acc);
            }
            CopyAndComputeLastIteration<INCREMENTAL_MODE>(idx, i, acc);
            yOutLocal.SetValue(i, acc);
        }
    }

    __aicore__ inline void CopyInIndex(const int64_t idx)
    {
        // look up the LoRA index
        reqLoRAIndex_ = indicesGm_.GetValue(idx);
    }

    __aicore__ inline void CopyInX(const int64_t idx, int32_t colIdx, int32_t numElements = TILE_LENGTH)
    {
        AscendC::LocalTensor<X_T> xLocal = inQueueX_.AllocTensor<X_T>();
        const uint64_t offset = inputHiddenDim_ * idx + colIdx * TILE_LENGTH;
        if (offset % INPUT_BLOCK_ELEMENTS == 0 && numElements % INPUT_BLOCK_ELEMENTS == 0) {
            DataCopy(xLocal, xGm_[offset], numElements);
        } else {
            const AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(numElements * sizeof(X_T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<X_T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInW(int32_t rowIdx, int32_t colIdx, int32_t numElements = TILE_LENGTH)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        const uint64_t offset = reqLoRAWeightOffset_ + static_cast<uint64_t>(rowIdx) * inputHiddenDim_ + colIdx * TILE_LENGTH;
        if (offset % INPUT_BLOCK_ELEMENTS == 0 && numElements % INPUT_BLOCK_ELEMENTS == 0) {
            DataCopy(wLocal, wGm_[offset], numElements);
        } else {
            const AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(numElements * sizeof(W_T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<W_T> padParams{false, 0, 0, 0};
            DataCopyPad(wLocal, wGm_[offset], copyParams, padParams);
        }
        inQueueW_.EnQue(wLocal);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void Compute(float &acc, int32_t numElements = TILE_LENGTH)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
        AscendC::LocalTensor<float> wTmpTensor = tmpBufferW_.Get<float>();

        if constexpr (INCREMENTAL_MODE) {
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, numElements);
            Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueX_.FreeTensor(xLocal);
            inQueueW_.FreeTensor(wLocal);
        } else {
            Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueW_.FreeTensor(wLocal);
        }
        // dot product of the one tile of X and W 
        Mul(wTmpTensor, xTmpTensor, wTmpTensor, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        // reduce sum generate one number, which is the summation of all the dot product
        ReduceSum<float>(wTmpTensor, wTmpTensor, wTmpTensor, numElements);
        AscendC::PipeBarrier<PIPE_V>();

        acc += wTmpTensor.GetValue(0);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void CopyAndComputeLastIteration(const int64_t idx, int32_t rowIdx, float &acc)
    {
        int32_t colIdx = inputHiddenDim_ / TILE_LENGTH;
        int32_t remaining = inputHiddenDim_ % TILE_LENGTH;
        if (remaining == 0) {
            return;
        }
        if constexpr (INCREMENTAL_MODE) {
            CopyInX(idx, colIdx, remaining);
        }
        CopyInW(rowIdx, colIdx, remaining);
        Compute<INCREMENTAL_MODE>(acc, remaining);
    }

    __aicore__ inline void ScaleOutput()
    {
        AscendC::LocalTensor<float> yLocal = outBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.AllocTensor<Y_T>();

        Muls(yOutLocal, yLocal, scale_, maxLoRARank_);
        AscendC::PipeBarrier<PIPE_V>();

        outQueueY_.EnQue<Y_T>(yOutLocal);
    }

    __aicore__ inline void CopyOut(const int64_t idx)
    {
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.DeQue<Y_T>();
        const AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(maxLoRARank_ * sizeof(Y_T)), 0, 0, 0};
        DataCopyPad(yOutGm_[maxLoRARank_ * idx], yOutLocal, copyParams);
        outQueueY_.FreeTensor(yOutLocal);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_, inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferX_, tmpBufferW_, outBufferY_, tailBufferY_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> wGm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    float scale_;
    uint64_t singleLoRAWeightLen_;
    int64_t reqLoRAIndex_;
    uint64_t reqLoRAWeightOffset_;
    bool incremental_;
    uint32_t paddedInputDim_;
    uint32_t rankTile_;
};

#define BGMV_SHRINK_TYPE_DECLARE(TYPE, NAME, BATCHED_RANK)                                                              \
    extern "C" __global__ __aicore__ void NAME(__gm__ void* x, __gm__ void* weight, __gm__ void* indices,              \
                                                             uint32_t indicesSize, __gm__ void* y, uint32_t batchSize, \
                                                             uint32_t numTokensPerCore, uint32_t inputHiddenDim,       \
                                                             uint32_t maxLoRARank, float scale)                        \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        BGMVShrink<TYPE, BATCHED_RANK> op(&pipe);                                                                       \
        op.Init(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);  \
        op.Process(AscendC::GetBlockIdx());                                                                             \
    }

#define BGMV_SHRINK_PAIR_TYPE_DECLARE(TYPE, NAME, BATCHED_RANK)                                                        \
    extern "C" __global__ __aicore__ void NAME(__gm__ void* x, __gm__ void* weight0, __gm__ void* weight1,             \
        __gm__ void* indices, uint32_t indicesSize, __gm__ void* yPair, uint32_t batchSize,                             \
        uint32_t numTokensPerCore, uint32_t blocksPerProjection, uint32_t inputHiddenDim,                               \
        uint32_t maxLoRARank, float scale)                                                                              \
    {                                                                                                                  \
        const uint32_t blockIdx = AscendC::GetBlockIdx();                                                               \
        const uint32_t projection = blockIdx / blocksPerProjection;                                                     \
        const uint32_t localBlockIdx = blockIdx % blocksPerProjection;                                                  \
        const uint64_t planeElements = static_cast<uint64_t>(batchSize) * maxLoRARank;                                   \
        __gm__ void* weight = projection == 0 ? weight0 : weight1;                                                      \
        __gm__ void* y = reinterpret_cast<__gm__ float*>(yPair) + projection * planeElements;                            \
        AscendC::TPipe pipe;                                                                                            \
        BGMVShrink<TYPE, BATCHED_RANK> op(&pipe);                                                                         \
        op.Init(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);      \
        op.Process(localBlockIdx);                                                                                     \
    }

// declare all dtype kernel
BGMV_SHRINK_TYPE_DECLARE(half, bgmv_shrink_half, false)
BGMV_SHRINK_TYPE_DECLARE(half, bgmv_shrink_batched_half, true)
BGMV_SHRINK_PAIR_TYPE_DECLARE(half, bgmv_shrink_pair_half, false)
BGMV_SHRINK_PAIR_TYPE_DECLARE(half, bgmv_shrink_pair_batched_half, true)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    BGMV_SHRINK_TYPE_DECLARE(bfloat16_t, bgmv_shrink_bfloat16_t, false)
    BGMV_SHRINK_TYPE_DECLARE(bfloat16_t, bgmv_shrink_batched_bfloat16_t, true)
    BGMV_SHRINK_PAIR_TYPE_DECLARE(bfloat16_t, bgmv_shrink_pair_bfloat16_t, false)
    BGMV_SHRINK_PAIR_TYPE_DECLARE(bfloat16_t, bgmv_shrink_pair_batched_bfloat16_t, true)
#endif

namespace {
inline bool BatchRanksForShrink(uint32_t inputHiddenDim, uint32_t maxLoRARank)
{
    // Both supported input types are 16-bit. Dispatch depends on vector stride,
    // repeat count, UB capacity and estimated instruction count, never model dimensions.
    constexpr uint32_t INPUT_BLOCK_ELEMENTS = BLOCK_BYTES / sizeof(half);
    constexpr uint32_t OUTPUT_BLOCK_ELEMENTS = BLOCK_BYTES / sizeof(float);
    constexpr uint32_t VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(float);
    const uint64_t paddedInput = (static_cast<uint64_t>(inputHiddenDim) + INPUT_BLOCK_ELEMENTS - 1) /
        INPUT_BLOCK_ELEMENTS * INPUT_BLOCK_ELEMENTS;
    const uint64_t rowBytes = paddedInput * (sizeof(half) + sizeof(float));
    const bool fitsResources = inputHiddenDim > 0 && maxLoRARank > 0 &&
        paddedInput / OUTPUT_BLOCK_ELEMENTS <= MAX_VECTOR_REPEAT_STRIDE &&
        2 * rowBytes + OUTPUT_ALIGNMENT_RESERVE + 3 * sizeof(float) <= BATCHED_UB_BYTES;
    uint32_t rankTile = 0;
    if (fitsResources) {
        rankTile = (BATCHED_UB_BYTES - rowBytes - OUTPUT_ALIGNMENT_RESERVE) / (rowBytes + 3 * sizeof(float));
        rankTile = rankTile < MAX_VECTOR_REPEAT ? rankTile : MAX_VECTOR_REPEAT;
        rankTile = rankTile < maxLoRARank ? rankTile : maxLoRARank;
        if (rankTile >= OUTPUT_BLOCK_ELEMENTS) {
            rankTile = rankTile / OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
        }
    }
    const uint64_t columnRepeats = (static_cast<uint64_t>(inputHiddenDim) + VECTOR_ELEMENTS - 1) / VECTOR_ELEMENTS;
    return fitsResources && columnRepeats <= rankTile;
}
} // namespace

namespace vllm_ascend {
extern void bgmv_shrink_impl(AscendType type, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize,
                             void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim,
                             uint32_t maxLoRARank, float scale)
{
    uint32_t blockDim = (batchSize + numTokensPerCore - 1) / numTokensPerCore;
    const bool batchRanks = BatchRanksForShrink(inputHiddenDim, maxLoRARank);
    if (type == AscendType::FP16) {
        if (batchRanks) {
            bgmv_shrink_batched_half<<<blockDim, nullptr, stream>>>(x, weight, indices, indicesSize, y, batchSize,
                numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
        } else {
            bgmv_shrink_half<<<blockDim, nullptr, stream>>>(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore,
                                                        inputHiddenDim, maxLoRARank, scale);
        }
    } else if (type == AscendType::BF16) {
        #if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        if (batchRanks) {
            bgmv_shrink_batched_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weight, indices, indicesSize, y, batchSize,
                numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
        } else {
            bgmv_shrink_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore,
                                                                  inputHiddenDim, maxLoRARank, scale);
        }
        #endif
    } else {
        return;
    }
}

// The binding validates tensor metadata and obtains aivNum from the guarded
// input device. Only this host entry selects paired scheduling or two singles.
extern void bgmv_shrink_pair_impl(AscendType type, void* stream, void* x, void* weight0, void* weight1,
                                  void* indices, uint32_t indicesSize, void* yPair, uint32_t batchSize,
                                  uint32_t aivNum, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    if (batchSize == 0) {
        return;
    }
    const uint64_t planeElements = static_cast<uint64_t>(batchSize) * maxLoRARank;
    const uint64_t planeBytes = planeElements * sizeof(float);
    const bool alignedPlanes = reinterpret_cast<uintptr_t>(yPair) % BLOCK_BYTES == 0 &&
        planeBytes % BLOCK_BYTES == 0;
    if (aivNum < 2 || !alignedPlanes) {
        const uint32_t cores = aivNum > 0 ? aivNum : 1;
        const uint32_t numTokensPerCore = (static_cast<uint64_t>(batchSize) + cores - 1) / cores;
        void* ySecond = static_cast<float*>(yPair) + planeElements;
        bgmv_shrink_impl(type, stream, x, weight0, indices, indicesSize, yPair, batchSize, numTokensPerCore,
                         inputHiddenDim, maxLoRARank, scale);
        bgmv_shrink_impl(type, stream, x, weight1, indices, indicesSize, ySecond, batchSize, numTokensPerCore,
                         inputHiddenDim, maxLoRARank, scale);
        return;
    }
    const uint32_t halfCores = aivNum / 2;
    const uint32_t coresPerProjection = batchSize < halfCores ? batchSize : halfCores;
    const uint32_t numTokensPerCore =
        (static_cast<uint64_t>(batchSize) + coresPerProjection - 1) / coresPerProjection;
    const uint32_t blocksPerProjection =
        (static_cast<uint64_t>(batchSize) + numTokensPerCore - 1) / numTokensPerCore;
    const uint32_t blockDim = 2 * blocksPerProjection;
    const bool batchRanks = BatchRanksForShrink(inputHiddenDim, maxLoRARank);
    if (type == AscendType::FP16) {
        if (batchRanks) {
            bgmv_shrink_pair_batched_half<<<blockDim, nullptr, stream>>>(x, weight0, weight1, indices, indicesSize,
                yPair, batchSize, numTokensPerCore, blocksPerProjection, inputHiddenDim, maxLoRARank, scale);
        } else {
            bgmv_shrink_pair_half<<<blockDim, nullptr, stream>>>(x, weight0, weight1, indices, indicesSize,
                yPair, batchSize, numTokensPerCore, blocksPerProjection, inputHiddenDim, maxLoRARank, scale);
        }
    } else if (type == AscendType::BF16) {
        #if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        if (batchRanks) {
            bgmv_shrink_pair_batched_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weight0, weight1, indices,
                indicesSize, yPair, batchSize, numTokensPerCore, blocksPerProjection, inputHiddenDim, maxLoRARank, scale);
        } else {
            bgmv_shrink_pair_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weight0, weight1, indices, indicesSize,
                yPair, batchSize, numTokensPerCore, blocksPerProjection, inputHiddenDim, maxLoRARank, scale);
        }
        #endif
    }
}

} // namespace vllm_ascend
