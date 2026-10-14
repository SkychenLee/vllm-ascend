/*
 * Copyright (c) 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstdlib>

#include "kernel_operator.h"
#include "lib/activation/swiglu.h"
#include "types.h"

namespace {
constexpr uint32_t TILE_ELEMENTS = 8192;
constexpr uint32_t MAX_TOKENS_PER_TILE = 16;
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t UB_BLOCK_FLOATS = 8;
constexpr uint32_t FP32_PER_REPEAT = 64; // vector ops are 256-byte aligned
constexpr float SWIGLU_BETA = 1.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr float FLOAT_MAX_VALUE = 3.402823466e+38f;
constexpr uint32_t MAX_WIDTH = FP32_PER_REPEAT * FP32_PER_REPEAT;

template <typename scalar_t>
class AddLoraSwigluQuant {
public:
    __aicore__ inline AddLoraSwigluQuant(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *gateUp, __gm__ void *delta, __gm__ void *act,
                                __gm__ void *y, __gm__ void *scale, uint32_t batchSize,
                                uint32_t numTokensPerCore, uint32_t width, uint32_t hasDelta,
                                float swigluLimit)
    {
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        width_ = width;
        hasDelta_ = hasDelta;
        swigluLimit_ = swigluLimit;

        numRowsPerTile_ = TILE_ELEMENTS / width_;
        if (numRowsPerTile_ == 0) {
            numRowsPerTile_ = 1;
        } else if (numRowsPerTile_ > MAX_TOKENS_PER_TILE) {
            numRowsPerTile_ = MAX_TOKENS_PER_TILE;
        }
        uint32_t tileElements = numRowsPerTile_ * width_;

        gateUpGm_.SetGlobalBuffer((__gm__ scalar_t *)gateUp);
        deltaGm_.SetGlobalBuffer((__gm__ scalar_t *)delta);
        actGm_.SetGlobalBuffer((__gm__ scalar_t *)act);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y);
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale, batchSize_);

        pipe_->InitBuffer(inQueueGateUp_, BUFFER_NUM, tileElements * 2 * sizeof(scalar_t));
        if (hasDelta_ != 0) {
            pipe_->InitBuffer(inQueueDelta_, BUFFER_NUM, tileElements * 2 * sizeof(scalar_t));
        }
        pipe_->InitBuffer(outQueueAct_, BUFFER_NUM, tileElements * sizeof(scalar_t));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, tileElements * sizeof(int8_t));
        pipe_->InitBuffer(outQueueScale_, BUFFER_NUM, MAX_TOKENS_PER_TILE * sizeof(float));
        pipe_->InitBuffer(gateBuffer_, tileElements * sizeof(float));
        pipe_->InitBuffer(upBuffer_, tileElements * sizeof(float));
        pipe_->InitBuffer(actBuffer_, tileElements * sizeof(float));
        reduceFull_ = width_ / FP32_PER_REPEAT;
        reduceTail_ = width_ % FP32_PER_REPEAT;
        reducePartials_ = reduceFull_ + (reduceTail_ != 0 ? 1 : 0);
        reducePitch_ = (reducePartials_ + UB_BLOCK_FLOATS - 1) / UB_BLOCK_FLOATS * UB_BLOCK_FLOATS;

        pipe_->InitBuffer(maxBuffer_, 2 * MAX_TOKENS_PER_TILE * UB_BLOCK_FLOATS * sizeof(float));
        pipe_->InitBuffer(rowMaxBuffer_, MAX_TOKENS_PER_TILE * sizeof(float));
        pipe_->InitBuffer(recipBuffer_, MAX_TOKENS_PER_TILE * sizeof(float));
        pipe_->InitBuffer(constBuffer_, MAX_TOKENS_PER_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t startIdx = AscendC::GetBlockIdx() * (int64_t)numTokensPerCore_;
        int64_t endIdx = startIdx + numTokensPerCore_;
        endIdx = endIdx > (int64_t)batchSize_ ? batchSize_ : endIdx;

        for (int64_t idx = startIdx; idx < endIdx; idx += numRowsPerTile_) {
            uint32_t numRows = (uint32_t)(endIdx - idx);
            numRows = numRows > numRowsPerTile_ ? numRowsPerTile_ : numRows;

            CopyIn(idx, numRows);
            Compute(numRows);
            CopyOut(idx, numRows);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t idx, uint32_t numRows)
    {
        int64_t offset = idx * (int64_t)width_ * 2;
        uint32_t numElements = numRows * width_ * 2;

        AscendC::LocalTensor<scalar_t> gateUpLocal = inQueueGateUp_.AllocTensor<scalar_t>();
        DataCopy(gateUpLocal, gateUpGm_[offset], numElements);
        inQueueGateUp_.EnQue(gateUpLocal);

        if (hasDelta_ != 0) {
            AscendC::LocalTensor<scalar_t> deltaLocal = inQueueDelta_.AllocTensor<scalar_t>();
            DataCopy(deltaLocal, deltaGm_[offset], numElements);
            inQueueDelta_.EnQue(deltaLocal);
        }
    }

    __aicore__ inline void Compute(uint32_t numRows)
    {
        uint32_t numElements = numRows * width_;
        AscendC::LocalTensor<float> gate = gateBuffer_.Get<float>();
        AscendC::LocalTensor<float> up = upBuffer_.Get<float>();
        AscendC::LocalTensor<float> activated = actBuffer_.Get<float>();
        AscendC::LocalTensor<float> maxs = maxBuffer_.Get<float>();
      
        AscendC::LocalTensor<scalar_t> gateUpLocal = inQueueGateUp_.DeQue<scalar_t>();

        // unstack (gate,up) into two distinct buffers, widen to fp32
        for (uint32_t i = 0; i < numRows; i++) {
            Cast(gate[i * width_], gateUpLocal[i * 2 * width_], AscendC::RoundMode::CAST_NONE, width_);
            Cast(up[i * width_], gateUpLocal[i * 2 * width_ + width_], AscendC::RoundMode::CAST_NONE, width_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        inQueueGateUp_.FreeTensor(gateUpLocal);

        if (hasDelta_ != 0) {
            // same for lora_delta, then add
            AscendC::LocalTensor<scalar_t> deltaLocal = inQueueDelta_.DeQue<scalar_t>();
            for (uint32_t i = 0; i < numRows; i++) {
                Cast(activated[i * width_], deltaLocal[i * 2 * width_], AscendC::RoundMode::CAST_NONE, width_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(gate, gate, activated, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < numRows; i++) {
                Cast(activated[i * width_], deltaLocal[i * 2 * width_ + width_], AscendC::RoundMode::CAST_NONE,
                     width_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Add(up, up, activated, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueDelta_.FreeTensor(deltaLocal);
        }

        if (swigluLimit_ > 0.0f) {
            Mins(gate, gate, swigluLimit_, (int32_t)numElements);
            Maxs(up, up, -swigluLimit_, (int32_t)numElements);
            Mins(up, up, swigluLimit_, (int32_t)numElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // swiglu
        AscendC::SwiGLU<float, false>(activated, up, gate, SWIGLU_BETA, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        
        // prepare bf_16 activations for W2 lora shink
        // TODO: check how quality degrades if we use quantized inputs instead
        AscendC::LocalTensor<scalar_t> actLocal = outQueueAct_.AllocTensor<scalar_t>();
        Cast(actLocal, activated, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueAct_.EnQue(actLocal);
        
        // reuse gate buffer as tmp for abs values
        Abs(gate, activated, numElements);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> rowMax = rowMaxBuffer_.Get<float>();
        // the problem with max-reduce is that we want to have scales as contiguous tensor, but we can only read 32B blocks from UB, and compute in repeats of 256B
        // pass 1: each row W -> ceil(W/64) partials, at a 32B-aligned pitch
        for (uint32_t i = 0; i < numRows; i++) {
            AscendC::WholeReduceMax<float>(
                maxs[i * reducePitch_],    // dst
                gate[i * width_],          // src
                (int32_t)FP32_PER_REPEAT,  // mask: all 64 lanes of the repeat
                (int32_t)reduceFull_,      // repeatTime: W / 64
                1,                         // dstRepStride: elements, we want contiguous output
                1,                         // srcBlkStride: blocks, stride between blocks in repeat
                (int32_t)UB_BLOCK_FLOATS,  // srcRepStride: elements, stride inside each block  
                AscendC::ReduceOrder::ORDER_ONLY_VALUE // don't return argmax
            );
            // mask is set per-repeat, so we need to operate the last w%64  in additional call with different mask
            if (reduceTail_ != 0) {
                    AscendC::WholeReduceMax<float>(
                        maxs[i * reducePitch_ + reduceFull_],
                        gate[i * width_ + reduceFull_ * FP32_PER_REPEAT],
                        (int32_t)reduceTail_,
                        1,
                        1,
                        1,
                        (int32_t)UB_BLOCK_FLOATS,
                        AscendC::ReduceOrder::ORDER_ONLY_VALUE
                    );
                }
            }
        AscendC::PipeBarrier<PIPE_V>();

        // pass 2: partials -> one contiguous maximum per row. 
        // TODO: for support W > 64 * 64, replace two passes with a cycle log64(W)
        AscendC::WholeReduceMax<float>(
            rowMax, maxs, (int32_t)reducePartials_,
            (int32_t)numRows,
            1,
            1,
            (int32_t)(reducePitch_ / UB_BLOCK_FLOATS),
            AscendC::ReduceOrder::ORDER_ONLY_VALUE
        );
        AscendC::PipeBarrier<PIPE_V>();
        
        // maxs -> scales
        AscendC::LocalTensor<float> scaleLocal = outQueueScale_.AllocTensor<float>();
        Muls(scaleLocal, rowMax, 1.0f / INT8_MAX_VALUE, numRows);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueScale_.EnQue(scaleLocal);
        
        // precumpute 127.0f / max per-row
        AscendC::LocalTensor<float> recip = recipBuffer_.Get<float>();
        AscendC::LocalTensor<float> ones = constBuffer_.Get<float>();
        Duplicate(ones, INT8_MAX_VALUE, (int32_t)numRows); 
        AscendC::PipeBarrier<PIPE_V>();
        Div(recip, ones, rowMax, numRows);
        AscendC::PipeBarrier<PIPE_V>();
        
        // repeats values into blocks (1, 2, 3) -> [1] * 8 + [2] * 8 + [3] * 8
        // needed to feed stride=0 matmul later, resues maxs buffer
        AscendC::Brcb(
            maxs, // dst
            recip, // src
            (uint8_t)((numRows + UB_BLOCK_FLOATS - 1) / UB_BLOCK_FLOATS),  // repeatTime: ceil(numRows, 8 floats per block)
            AscendC::BrcbRepeatParams(
                1,                // dstBlkStride: stride between blocks in repeat
                UB_BLOCK_FLOATS   // dstRepStride: stride inside each blocks
            )
        );
        AscendC::PipeBarrier<PIPE_V>();
        
        // scaling activations
        
        // set 0-strides to broadcast scales to row elements
        AscendC::BinaryRepeatParams bcast(
            1,                // dstBlkStride:  blocks contiguous within a repeat
            1,                // src0BlkStride: ditto, activations stream normally
            0,                // src1BlkStride: pin to one block, do not walk the scales
            UB_BLOCK_FLOATS,  // dstRepStride:  8 BLOCKS = the 64 fp32 per repeat
            UB_BLOCK_FLOATS,  // src0RepStride: 8 BLOCKS = the 64 fp32 per repeat
            0                 // src1RepStride: pin across repeats too, one scale per row
        );
        uint32_t fullRepeats = width_ / FP32_PER_REPEAT;
        uint32_t tailMask = width_ % FP32_PER_REPEAT;

        // the same mask is required, so processing the tail separately
        for (uint32_t i = 0; i < numRows; i++) {
            if (fullRepeats != 0) {
                Mul(activated[i * width_], activated[i * width_], maxs[i * UB_BLOCK_FLOATS],
                    (int32_t)FP32_PER_REPEAT, (uint8_t)fullRepeats, bcast);
            }
            if (tailMask != 0) {
                uint32_t off = i * width_ + fullRepeats * FP32_PER_REPEAT;
                Mul(activated[off], activated[off], maxs[i * UB_BLOCK_FLOATS],
                    (int32_t)tailMask, (uint8_t)1, bcast);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        // there is no fp32 -> int8 cast, so we do fp32->half->int8 in quant path
        AscendC::LocalTensor<half> halfLocal = up.ReinterpretCast<half>();
        Cast(halfLocal, activated, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY_.AllocTensor<int8_t>();
        Cast(yLocal, halfLocal, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t idx, uint32_t numRows)
    {
        int64_t offset = idx * (int64_t)width_;
        uint32_t numElements = numRows * width_;

        AscendC::LocalTensor<scalar_t> actLocal = outQueueAct_.DeQue<scalar_t>();
        DataCopy(actGm_[offset], actLocal, numElements);
        outQueueAct_.FreeTensor(actLocal);

        AscendC::LocalTensor<int8_t> yLocal = outQueueY_.DeQue<int8_t>();
        DataCopy(yGm_[offset], yLocal, numElements);
        outQueueY_.FreeTensor(yLocal);

        AscendC::LocalTensor<float> scaleLocal = outQueueScale_.DeQue<float>();
        AscendC::DataCopyExtParams scaleParams{1, numRows * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(scaleGm_[idx], scaleLocal, scaleParams);
        outQueueScale_.FreeTensor(scaleLocal);
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueGateUp_, inQueueDelta_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueAct_, outQueueY_, outQueueScale_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gateBuffer_, upBuffer_, actBuffer_, maxBuffer_,
        rowMaxBuffer_, recipBuffer_, constBuffer_;
    AscendC::GlobalTensor<scalar_t> gateUpGm_;
    AscendC::GlobalTensor<scalar_t> deltaGm_;
    AscendC::GlobalTensor<scalar_t> actGm_;
    AscendC::GlobalTensor<int8_t> yGm_;
    AscendC::GlobalTensor<float> scaleGm_;
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t width_;
    uint32_t hasDelta_;
    float swigluLimit_;
    uint32_t numRowsPerTile_;
    uint32_t reduceFull_;
    uint32_t reduceTail_;
    uint32_t reducePartials_;
    uint32_t reducePitch_;
};
}  // namespace

#define ADD_LORA_SWIGLU_QUANT_TYPE_DECLARE(TYPE)                                                     \
    extern "C" __global__ __aicore__ void add_lora_swiglu_quant_##TYPE(                              \
        __gm__ void* gateUp, __gm__ void* delta, __gm__ void* act, __gm__ void* y,                    \
        __gm__ void* scale, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t width,            \
        uint32_t hasDelta, float swigluLimit)                                                        \
    {                                                                                                \
        AscendC::TPipe pipe;                                                                         \
        AddLoraSwigluQuant<TYPE> op(&pipe);                                                          \
        op.Init(gateUp, delta, act, y, scale, batchSize, numTokensPerCore, width, hasDelta,           \
                swigluLimit);                                                                        \
        op.Process();                                                                                \
    }

// declare all dtype kernel
ADD_LORA_SWIGLU_QUANT_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ADD_LORA_SWIGLU_QUANT_TYPE_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {
extern void add_lora_swiglu_quant_impl(AscendType type, void *stream, void *gate_up, void *delta,
                                       void *act, void *y, void *scale, uint32_t batch,
                                       uint32_t width, uint32_t has_delta, float swiglu_limit,
                                       uint32_t aiv_num)
{
    if (width == 0 || width > MAX_WIDTH) {
        fprintf(stderr, "add_lora_swiglu_quant: width %u not in [1, %u]\n", width, MAX_WIDTH);
        abort();
    }

    uint32_t numTokensPerCore = (batch + aiv_num - 1) / aiv_num;
    if (numTokensPerCore == 0) {
        numTokensPerCore = 1;
    }
    uint32_t blockDim = (batch + numTokensPerCore - 1) / numTokensPerCore;

    if (type == AscendType::FP16) {
        add_lora_swiglu_quant_half<<<blockDim, nullptr, stream>>>(
            gate_up, delta, act, y, scale, batch, numTokensPerCore, width, has_delta, swiglu_limit);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        add_lora_swiglu_quant_bfloat16_t<<<blockDim, nullptr, stream>>>(
            gate_up, delta, act, y, scale, batch, numTokensPerCore, width, has_delta, swiglu_limit);
#endif
    }
}
}  // namespace vllm_ascend
