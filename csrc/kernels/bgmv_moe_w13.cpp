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
#include "types.h"

// DeepSeek-style W13 has two independent rank-16 LoRA projections over the
// same routed rows.  Running the normal BGMV path per slice costs four kernel
// launches (two shrink + two expand), while decode has too little arithmetic
// to hide launch latency.  These kernels execute both slices in one launch.

template <typename scalar_t>
class BGMVMoeW13Shrink {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = float;

    static constexpr uint32_t MAX_HIDDEN_DIM = 4096;
    static constexpr uint32_t RANK = 16;
    static constexpr uint32_t NUM_SLICES = 2;
    static constexpr uint32_t RANKS_PER_TASK = 8;
    static constexpr uint32_t TASKS_PER_TOKEN = NUM_SLICES * RANK / RANKS_PER_TASK;

    __aicore__ inline BGMVMoeW13Shrink(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* weight0, __gm__ void* weight1,
                                __gm__ void* indices, uint32_t indicesSize, __gm__ void* workspace,
                                uint32_t batchSize, uint32_t blockDim, uint32_t inputHiddenDim,
                                float scale)
    {
        batchSize_ = batchSize;
        blockDim_ = blockDim;
        inputHiddenDim_ = inputHiddenDim;
        scale_ = scale;
        singleLoRAWeightLen_ = static_cast<uint64_t>(inputHiddenDim_) * RANK;

        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        w0Gm_.SetGlobalBuffer((__gm__ W_T*)weight0);
        w1Gm_.SetGlobalBuffer((__gm__ W_T*)weight1);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t*)indices, indicesSize);
        workspaceGm_.SetGlobalBuffer((__gm__ Y_T*)workspace);

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
            const uint32_t sliceAndGroup = task % TASKS_PER_TOKEN;
            const uint32_t sliceIdx = sliceAndGroup / (RANK / RANKS_PER_TASK);
            const uint32_t rankStart = (sliceAndGroup % (RANK / RANKS_PER_TASK)) * RANKS_PER_TASK;
            const int64_t loraIdx = CopyInIndex(tokenIdx);
            if (loraIdx < 0) {
                CopyOutZero(sliceIdx, tokenIdx, rankStart);
                continue;
            }
            CopyInX(tokenIdx);
            ComputeAndCopyOut(sliceIdx, tokenIdx, rankStart, loraIdx);
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

    __aicore__ inline uint64_t WorkspaceOffset(uint32_t sliceIdx, uint32_t tokenIdx, uint32_t rankStart)
    {
        return (static_cast<uint64_t>(sliceIdx) * batchSize_ + tokenIdx) * RANK + rankStart;
    }

    __aicore__ inline void CopyOutZero(uint32_t sliceIdx, uint32_t tokenIdx, uint32_t rankStart)
    {
        AscendC::LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
        Duplicate(outLocal, 0.0f, RANKS_PER_TASK);
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<float>();
        DataCopy(workspaceGm_[WorkspaceOffset(sliceIdx, tokenIdx, rankStart)], outLocal, RANKS_PER_TASK);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ComputeAndCopyOut(uint32_t sliceIdx, uint32_t tokenIdx,
                                             uint32_t rankStart, int64_t loraIdx)
    {
        AscendC::LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
        AscendC::LocalTensor<float> xFp32 = xFp32Buffer_.Get<float>();
        AscendC::LocalTensor<float> wFp32 = wFp32Buffer_.Get<float>();
        const uint64_t loraOffset = static_cast<uint64_t>(loraIdx) * singleLoRAWeightLen_;

        for (uint32_t rankOffset = 0; rankOffset < RANKS_PER_TASK; ++rankOffset) {
            const uint32_t rankIdx = rankStart + rankOffset;
            AscendC::LocalTensor<W_T> wLocal = wQueue_.AllocTensor<W_T>();
            const uint64_t weightOffset = loraOffset + static_cast<uint64_t>(rankIdx) * inputHiddenDim_;
            if (sliceIdx == 0) {
                DataCopy(wLocal, w0Gm_[weightOffset], inputHiddenDim_);
            } else {
                DataCopy(wLocal, w1Gm_[weightOffset], inputHiddenDim_);
            }
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
        DataCopy(workspaceGm_[WorkspaceOffset(sliceIdx, tokenIdx, rankStart)], outLocal, RANKS_PER_TASK);
        outQueue_.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> indexQueue_, xQueue_, wQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xFp32Buffer_, wFp32Buffer_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> w0Gm_, w1Gm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<Y_T> workspaceGm_;
    uint32_t batchSize_;
    uint32_t blockDim_;
    uint32_t inputHiddenDim_;
    float scale_;
    uint64_t singleLoRAWeightLen_;
};

template <typename scalar_t>
class BGMVMoeW13Expand {
public:
    using X_T = float;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr uint32_t RANK = 16;
    static constexpr uint32_t NUM_SLICES = 2;
    static constexpr uint32_t OUTPUT_TILE = 512;
    static constexpr uint32_t NUM_ELEMENTS_PER_REPEAT = 64;
    static constexpr uint32_t NUM_BLOCKS_PER_REPEAT = 8;
    static constexpr uint32_t W_TILE = OUTPUT_TILE * RANK;
    static constexpr uint32_t BLOCK_REDUCE_REPEATS = W_TILE / NUM_ELEMENTS_PER_REPEAT;
    static constexpr uint32_t PAIR_REDUCE_REPEATS =
        (BLOCK_REDUCE_REPEATS * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) /
        NUM_ELEMENTS_PER_REPEAT;

    __aicore__ inline BGMVMoeW13Expand(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void* workspace, __gm__ void* weight0, __gm__ void* weight1,
                                __gm__ void* indices, uint32_t indicesSize, __gm__ void* y,
                                uint32_t batchSize, uint32_t blockDim, uint32_t outputSliceDim,
                                uint32_t sliceOffset, uint32_t outputFullDim)
    {
        batchSize_ = batchSize;
        blockDim_ = blockDim;
        outputSliceDim_ = outputSliceDim;
        sliceOffset_ = sliceOffset;
        outputFullDim_ = outputFullDim;
        tilesPerSlice_ = outputSliceDim_ / OUTPUT_TILE;
        singleLoRAWeightLen_ = static_cast<uint64_t>(RANK) * outputSliceDim_;

        workspaceGm_.SetGlobalBuffer((__gm__ X_T*)workspace);
        w0Gm_.SetGlobalBuffer((__gm__ W_T*)weight0);
        w1Gm_.SetGlobalBuffer((__gm__ W_T*)weight1);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t*)indices, indicesSize);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);

        pipe_->InitBuffer(indexQueue_, 1, 32);
        pipe_->InitBuffer(xQueue_, 1, RANK * sizeof(X_T));
        pipe_->InitBuffer(wQueue_, 1, W_TILE * sizeof(W_T));
        pipe_->InitBuffer(yQueue_, 1, OUTPUT_TILE * sizeof(Y_T));
        pipe_->InitBuffer(outQueue_, 1, OUTPUT_TILE * sizeof(Y_T));
        pipe_->InitBuffer(xDupBuffer_, NUM_ELEMENTS_PER_REPEAT * sizeof(float));
        pipe_->InitBuffer(wFp32Buffer_, W_TILE * sizeof(float));
        pipe_->InitBuffer(productBuffer_, OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(yFp32Buffer_, OUTPUT_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tasksPerToken = NUM_SLICES * tilesPerSlice_;
        const uint32_t totalTasks = batchSize_ * tasksPerToken;
        for (uint32_t task = AscendC::GetBlockIdx(); task < totalTasks; task += blockDim_) {
            const uint32_t tokenIdx = task / tasksPerToken;
            const uint32_t sliceAndTile = task % tasksPerToken;
            const uint32_t sliceIdx = sliceAndTile / tilesPerSlice_;
            const uint32_t tileIdx = sliceAndTile % tilesPerSlice_;
            const int64_t loraIdx = CopyInIndex(tokenIdx);
            if (loraIdx < 0) {
                continue;
            }
            CopyInX(sliceIdx, tokenIdx);
            ComputeTile(sliceIdx, tokenIdx, tileIdx, loraIdx);
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

    __aicore__ inline void CopyInX(uint32_t sliceIdx, uint32_t tokenIdx)
    {
        const uint64_t workspaceOffset = (static_cast<uint64_t>(sliceIdx) * batchSize_ + tokenIdx) * RANK;
        AscendC::LocalTensor<X_T> xLocal = xQueue_.AllocTensor<X_T>();
        DataCopy(xLocal, workspaceGm_[workspaceOffset], RANK);
        xQueue_.EnQue(xLocal);
        xLocal = xQueue_.DeQue<X_T>();
        AscendC::LocalTensor<float> xDup = xDupBuffer_.Get<float>();
        for (uint32_t offset = 0; offset < NUM_ELEMENTS_PER_REPEAT; offset += RANK) {
            for (uint32_t rankIdx = 0; rankIdx < RANK; ++rankIdx) {
                xDup.SetValue(offset + rankIdx, xLocal.GetValue(rankIdx));
            }
        }
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeTile(uint32_t sliceIdx, uint32_t tokenIdx,
                                       uint32_t tileIdx, int64_t loraIdx)
    {
        const uint64_t weightOffset = static_cast<uint64_t>(loraIdx) * singleLoRAWeightLen_ +
                                      static_cast<uint64_t>(tileIdx) * W_TILE;
        AscendC::LocalTensor<W_T> wLocal = wQueue_.AllocTensor<W_T>();
        if (sliceIdx == 0) {
            DataCopy(wLocal, w0Gm_[weightOffset], W_TILE);
        } else {
            DataCopy(wLocal, w1Gm_[weightOffset], W_TILE);
        }
        wQueue_.EnQue(wLocal);
        wLocal = wQueue_.DeQue<W_T>();

        AscendC::LocalTensor<float> wFp32 = wFp32Buffer_.Get<float>();
        AscendC::LocalTensor<float> xDup = xDupBuffer_.Get<float>();
        AscendC::LocalTensor<float> product = productBuffer_.Get<float>();
        Cast(wFp32, wLocal, AscendC::RoundMode::CAST_NONE, NUM_ELEMENTS_PER_REPEAT,
             BLOCK_REDUCE_REPEATS, castParams_);
        AscendC::PipeBarrier<PIPE_V>();
        wQueue_.FreeTensor(wLocal);
        Mul(wFp32, xDup, wFp32, NUM_ELEMENTS_PER_REPEAT, BLOCK_REDUCE_REPEATS, dotProductParams_);
        AscendC::PipeBarrier<PIPE_V>();
        BlockReduceSum(wFp32, wFp32, BLOCK_REDUCE_REPEATS, NUM_ELEMENTS_PER_REPEAT,
                       reduceParams_.dstRepStride, reduceParams_.srcBlkStride, reduceParams_.srcRepStride);
        AscendC::PipeBarrier<PIPE_V>();
        PairReduceSum(product, wFp32, PAIR_REDUCE_REPEATS, NUM_ELEMENTS_PER_REPEAT,
                      reduceParams_.dstRepStride, reduceParams_.srcBlkStride, reduceParams_.srcRepStride);
        AscendC::PipeBarrier<PIPE_V>();

        const uint64_t yOffset = static_cast<uint64_t>(tokenIdx) * outputFullDim_ + sliceOffset_ +
                                 static_cast<uint64_t>(sliceIdx) * outputSliceDim_ +
                                 static_cast<uint64_t>(tileIdx) * OUTPUT_TILE;
        AscendC::LocalTensor<Y_T> yLocal = yQueue_.AllocTensor<Y_T>();
        DataCopy(yLocal, yGm_[yOffset], OUTPUT_TILE);
        yQueue_.EnQue(yLocal);
        yLocal = yQueue_.DeQue<Y_T>();
        AscendC::LocalTensor<float> yFp32 = yFp32Buffer_.Get<float>();
        Cast(yFp32, yLocal, AscendC::RoundMode::CAST_NONE, OUTPUT_TILE);
        AscendC::PipeBarrier<PIPE_V>();
        yQueue_.FreeTensor(yLocal);
        Add(product, product, yFp32, OUTPUT_TILE);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<Y_T> outLocal = outQueue_.AllocTensor<Y_T>();
        Cast(outLocal, product, AscendC::RoundMode::CAST_RINT, OUTPUT_TILE);
        AscendC::PipeBarrier<PIPE_V>();
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<Y_T>();
        DataCopy(yGm_[yOffset], outLocal, OUTPUT_TILE);
        outQueue_.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> indexQueue_, xQueue_, wQueue_, yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xDupBuffer_, wFp32Buffer_, productBuffer_, yFp32Buffer_;
    AscendC::GlobalTensor<X_T> workspaceGm_;
    AscendC::GlobalTensor<W_T> w0Gm_, w1Gm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t batchSize_;
    uint32_t blockDim_;
    uint32_t outputSliceDim_;
    uint32_t sliceOffset_;
    uint32_t outputFullDim_;
    uint32_t tilesPerSlice_;
    uint64_t singleLoRAWeightLen_;

    AscendC::UnaryRepeatParams castParams_ = {1, 1, 8, 4};
    AscendC::UnaryRepeatParams reduceParams_ = {1, 1, 1, 8};
    AscendC::BinaryRepeatParams dotProductParams_ = {1, 1, 1, 8, 0, 8};
};

#define BGMV_MOE_W13_TYPE_DECLARE(TYPE)                                                                                \
    extern "C" __global__ __aicore__ void bgmv_moe_w13_shrink_##TYPE(                                                \
        __gm__ void* x, __gm__ void* weight0, __gm__ void* weight1, __gm__ void* indices, uint32_t indicesSize,       \
        __gm__ void* workspace, uint32_t batchSize, uint32_t blockDim, uint32_t inputHiddenDim, float scale)          \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        BGMVMoeW13Shrink<TYPE> op(&pipe);                                                                               \
        op.Init(x, weight0, weight1, indices, indicesSize, workspace, batchSize, blockDim, inputHiddenDim, scale);     \
        op.Process();                                                                                                  \
    }                                                                                                                  \
    extern "C" __global__ __aicore__ void bgmv_moe_w13_expand_##TYPE(                                                \
        __gm__ void* workspace, __gm__ void* weight0, __gm__ void* weight1, __gm__ void* indices,                    \
        uint32_t indicesSize, __gm__ void* y, uint32_t batchSize, uint32_t blockDim, uint32_t outputSliceDim,         \
        uint32_t sliceOffset, uint32_t outputFullDim)                                                                  \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        BGMVMoeW13Expand<TYPE> op(&pipe);                                                                               \
        op.Init(workspace, weight0, weight1, indices, indicesSize, y, batchSize, blockDim, outputSliceDim,            \
                sliceOffset, outputFullDim);                                                                            \
        op.Process();                                                                                                  \
    }

BGMV_MOE_W13_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
BGMV_MOE_W13_TYPE_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {
extern void bgmv_moe_w13_impl(AscendType type, void* stream, void* x, void* weightA0, void* weightA1,
                              void* weightB0, void* weightB1, void* indices, uint32_t indicesSize,
                              void* workspace, void* y, uint32_t batchSize, uint32_t inputHiddenDim,
                              uint32_t outputSliceDim, uint32_t sliceOffset, uint32_t outputFullDim,
                              float scale, uint32_t aivNum)
{
    const uint32_t shrinkTasks = batchSize * BGMVMoeW13Shrink<half>::TASKS_PER_TOKEN;
    const uint32_t shrinkBlockDim = shrinkTasks < aivNum ? shrinkTasks : aivNum;
    const uint32_t expandTasks = batchSize * BGMVMoeW13Expand<half>::NUM_SLICES *
                                 (outputSliceDim / BGMVMoeW13Expand<half>::OUTPUT_TILE);
    const uint32_t expandBlockDim = expandTasks < aivNum ? expandTasks : aivNum;

    if (type == AscendType::FP16) {
        bgmv_moe_w13_shrink_half<<<shrinkBlockDim, nullptr, stream>>>(
            x, weightA0, weightA1, indices, indicesSize, workspace, batchSize, shrinkBlockDim,
            inputHiddenDim, scale);
        bgmv_moe_w13_expand_half<<<expandBlockDim, nullptr, stream>>>(
            workspace, weightB0, weightB1, indices, indicesSize, y, batchSize, expandBlockDim,
            outputSliceDim, sliceOffset, outputFullDim);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        bgmv_moe_w13_shrink_bfloat16_t<<<shrinkBlockDim, nullptr, stream>>>(
            x, weightA0, weightA1, indices, indicesSize, workspace, batchSize, shrinkBlockDim,
            inputHiddenDim, scale);
        bgmv_moe_w13_expand_bfloat16_t<<<expandBlockDim, nullptr, stream>>>(
            workspace, weightB0, weightB1, indices, indicesSize, y, batchSize, expandBlockDim,
            outputSliceDim, sliceOffset, outputFullDim);
#endif
    }
}
}  // namespace vllm_ascend
