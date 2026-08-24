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

// Rank-16 is the dominant MoE LoRA decode shape.  The generic kernel assigns
// one token to one vector core and computes all rank rows serially.  Decode
// usually has only a handful of routed rows, so most vector cores stay idle.
// Split the rank into two 32-byte-aligned groups.  Each task owns one output
// group, which avoids atomics and keeps GM writes naturally aligned.
template <typename scalar_t>
class BGMVShrinkRank16 {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = float;

    static constexpr uint32_t MAX_HIDDEN_DIM = 4096;
    static constexpr uint32_t RANK = 16;
    static constexpr uint32_t RANKS_PER_TASK = 8;
    static constexpr uint32_t TASKS_PER_TOKEN = RANK / RANKS_PER_TASK;

    __aicore__ inline BGMVShrinkRank16(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* weight, __gm__ void* indices,
                                uint32_t indicesSize, __gm__ void* y, uint32_t batchSize,
                                uint32_t blockDim, uint32_t inputHiddenDim, float scale)
    {
        batchSize_ = batchSize;
        blockDim_ = blockDim;
        inputHiddenDim_ = inputHiddenDim;
        scale_ = scale;
        singleLoRAWeightLen_ = static_cast<uint64_t>(inputHiddenDim_) * RANK;

        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        wGm_.SetGlobalBuffer((__gm__ W_T*)weight);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t*)indices, indicesSize);
        yOutGm_.SetGlobalBuffer((__gm__ Y_T*)y);

        pipe_->InitBuffer(indexQueue_, 1, 32);
        pipe_->InitBuffer(xQueue_, 1, MAX_HIDDEN_DIM * sizeof(X_T));
        pipe_->InitBuffer(wQueue_, 1, MAX_HIDDEN_DIM * sizeof(W_T));
        pipe_->InitBuffer(xFp32Buffer_, MAX_HIDDEN_DIM * sizeof(float));
        pipe_->InitBuffer(wFp32Buffer_, MAX_HIDDEN_DIM * sizeof(float));
        pipe_->InitBuffer(outQueue_, 1, RANKS_PER_TASK * sizeof(Y_T));
    }

    __aicore__ inline void Process()
    {
        const uint32_t totalTasks = batchSize_ * TASKS_PER_TOKEN;
        for (uint32_t task = AscendC::GetBlockIdx(); task < totalTasks; task += blockDim_) {
            const uint32_t tokenIdx = task / TASKS_PER_TOKEN;
            const uint32_t rankStart = (task % TASKS_PER_TOKEN) * RANKS_PER_TASK;
            const int64_t loraIdx = CopyInIndex(tokenIdx);
            if (loraIdx < 0) {
                continue;
            }
            CopyInX(tokenIdx);
            ComputeAndCopyOut(tokenIdx, rankStart, loraIdx);
        }
    }

private:
    __aicore__ inline int64_t CopyInIndex(uint32_t tokenIdx)
    {
        AscendC::LocalTensor<int64_t> indexLocal = indexQueue_.AllocTensor<int64_t>();
        DataCopyPad(indexLocal, indicesGm_[tokenIdx], {1, static_cast<uint16_t>(sizeof(int64_t)), 0, 0}, {});
        indexQueue_.EnQue(indexLocal);
        indexLocal = indexQueue_.DeQue<int64_t>();
        const int64_t loraIdx = indexLocal.GetValue(0);
        indexQueue_.FreeTensor(indexLocal);
        return loraIdx;
    }

    __aicore__ inline void CopyInX(uint32_t tokenIdx)
    {
        AscendC::LocalTensor<X_T> xLocal = xQueue_.AllocTensor<X_T>();
        DataCopy(xLocal, xGm_[static_cast<uint64_t>(tokenIdx) * inputHiddenDim_], inputHiddenDim_);
        xQueue_.EnQue(xLocal);
        xLocal = xQueue_.DeQue<X_T>();
        AscendC::LocalTensor<float> xFp32 = xFp32Buffer_.Get<float>();
        Cast(xFp32, xLocal, AscendC::RoundMode::CAST_NONE, inputHiddenDim_);
        AscendC::PipeBarrier<PIPE_V>();
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeAndCopyOut(uint32_t tokenIdx, uint32_t rankStart, int64_t loraIdx)
    {
        AscendC::LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
        AscendC::LocalTensor<float> xFp32 = xFp32Buffer_.Get<float>();
        AscendC::LocalTensor<float> wFp32 = wFp32Buffer_.Get<float>();
        const uint64_t loraOffset = static_cast<uint64_t>(loraIdx) * singleLoRAWeightLen_;

        for (uint32_t rankOffset = 0; rankOffset < RANKS_PER_TASK; ++rankOffset) {
            const uint32_t rankIdx = rankStart + rankOffset;
            AscendC::LocalTensor<W_T> wLocal = wQueue_.AllocTensor<W_T>();
            DataCopy(wLocal, wGm_[loraOffset + static_cast<uint64_t>(rankIdx) * inputHiddenDim_], inputHiddenDim_);
            wQueue_.EnQue(wLocal);
            wLocal = wQueue_.DeQue<W_T>();
            Cast(wFp32, wLocal, AscendC::RoundMode::CAST_NONE, inputHiddenDim_);
            AscendC::PipeBarrier<PIPE_V>();
            wQueue_.FreeTensor(wLocal);

            Mul(wFp32, xFp32, wFp32, inputHiddenDim_);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSum<float>(wFp32, wFp32, wFp32, inputHiddenDim_);
            AscendC::PipeBarrier<PIPE_V>();
            outLocal.SetValue(rankOffset, wFp32.GetValue(0) * scale_);
        }

        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<float>();
        const uint64_t outputOffset = static_cast<uint64_t>(tokenIdx) * RANK + rankStart;
        DataCopy(yOutGm_[outputOffset], outLocal, RANKS_PER_TASK);
        outQueue_.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> indexQueue_, xQueue_, wQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xFp32Buffer_, wFp32Buffer_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> wGm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;
    uint32_t batchSize_;
    uint32_t blockDim_;
    uint32_t inputHiddenDim_;
    float scale_;
    uint64_t singleLoRAWeightLen_;
};

template <typename scalar_t>
class BGMVShrink {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = float;

    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t TILE_LENGTH = 11776;  // optimal performance tile length

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
        singleLoRAWeightLen_ = inputHiddenDim_ * maxLoRARank_;
        incremental_ = inputHiddenDim_ > TILE_LENGTH;

        xGm_.SetGlobalBuffer((__gm__ X_T *)x);
        yOutGm_.SetGlobalBuffer((__gm__ Y_T *)y);
        wGm_.SetGlobalBuffer((__gm__ W_T *)weight);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, indicesSize);

        pipe_->InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(X_T));
        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, TILE_LENGTH * sizeof(W_T));
        pipe_->InitBuffer(tmpBufferX_, TILE_LENGTH * sizeof(float));
        pipe_->InitBuffer(tmpBufferW_, TILE_LENGTH * sizeof(float));
        
        pipe_->InitBuffer(outQueueY_, 1, maxLoRARank_ * sizeof(Y_T));
        pipe_->InitBuffer(outBufferY_, maxLoRARank_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t startIdx = blockIdx * numTokensPerCore_;
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

            if (incremental_) {
                ProcessImpl<true>(idx);
            } else {
                ProcessImpl<false>(idx);
            }

            ScaleOutput();
            CopyOut(idx);
        }
    }

private:
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
        DataCopy(xLocal, xGm_[inputHiddenDim_ * idx + colIdx * TILE_LENGTH], numElements);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInW(int32_t rowIdx, int32_t colIdx, int32_t numElements = TILE_LENGTH)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        DataCopy(wLocal, wGm_[reqLoRAWeightOffset_ + rowIdx * inputHiddenDim_ + colIdx * TILE_LENGTH], numElements);
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
        DataCopy(yOutGm_[maxLoRARank_ * idx], yOutLocal, maxLoRARank_);
        outQueueY_.FreeTensor(yOutLocal);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_, inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferX_, tmpBufferW_, outBufferY_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> wGm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    float scale_;
    uint32_t singleLoRAWeightLen_;
    int64_t reqLoRAIndex_;
    uint64_t reqLoRAWeightOffset_;
    bool incremental_;
};

#define BGMV_SHRINK_TYPE_DECLARE(TYPE)                                                                                 \
    extern "C" __global__ __aicore__ void bgmv_shrink_##TYPE(__gm__ void* x, __gm__ void* weight, __gm__ void* indices,\
                                                             uint32_t indicesSize, __gm__ void* y, uint32_t batchSize, \
                                                             uint32_t numTokensPerCore, uint32_t inputHiddenDim,       \
                                                             uint32_t maxLoRARank, float scale)                        \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        BGMVShrink<TYPE> op(&pipe);                                                                                    \
        op.Init(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);  \
        op.Process();                                                                                                  \
    }

// declare all dtype kernel
BGMV_SHRINK_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    BGMV_SHRINK_TYPE_DECLARE(bfloat16_t)
#endif

#define BGMV_SHRINK_RANK16_TYPE_DECLARE(TYPE)                                                                          \
    extern "C" __global__ __aicore__ void bgmv_shrink_rank16_##TYPE(                                                  \
        __gm__ void* x, __gm__ void* weight, __gm__ void* indices, uint32_t indicesSize, __gm__ void* y,              \
        uint32_t batchSize, uint32_t blockDim, uint32_t inputHiddenDim, float scale)                                  \
    {                                                                                                                   \
        AscendC::TPipe pipe;                                                                                            \
        BGMVShrinkRank16<TYPE> op(&pipe);                                                                               \
        op.Init(x, weight, indices, indicesSize, y, batchSize, blockDim, inputHiddenDim, scale);                       \
        op.Process();                                                                                                   \
    }

BGMV_SHRINK_RANK16_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    BGMV_SHRINK_RANK16_TYPE_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {
extern void bgmv_shrink_impl(AscendType type, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize,
                             void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim,
                             uint32_t maxLoRARank, float scale, uint32_t aivNum)
{
    if (maxLoRARank == 16 && inputHiddenDim <= BGMVShrinkRank16<half>::MAX_HIDDEN_DIM &&
        inputHiddenDim % 16 == 0) {
        const uint32_t totalTasks = batchSize * BGMVShrinkRank16<half>::TASKS_PER_TOKEN;
        const uint32_t blockDim = totalTasks < aivNum ? totalTasks : aivNum;
        if (type == AscendType::FP16) {
            bgmv_shrink_rank16_half<<<blockDim, nullptr, stream>>>(x, weight, indices, indicesSize, y, batchSize,
                                                                  blockDim, inputHiddenDim, scale);
        } else if (type == AscendType::BF16) {
            #if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
            bgmv_shrink_rank16_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weight, indices, indicesSize, y, batchSize,
                                                                        blockDim, inputHiddenDim, scale);
            #endif
        }
        return;
    }

    uint32_t blockDim = (batchSize + numTokensPerCore - 1) / numTokensPerCore;
    if (type == AscendType::FP16) {
        bgmv_shrink_half<<<blockDim, nullptr, stream>>>(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, 
                                                        inputHiddenDim, maxLoRARank, scale);
    } else if (type == AscendType::BF16) {
        #if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        bgmv_shrink_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, 
                                                                  inputHiddenDim, maxLoRARank, scale);
        #endif
    } else {
        return;
    }
}

} // namespace vllm_ascend
