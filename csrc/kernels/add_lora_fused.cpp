/*
 * Copyright (c) 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused LoRA apply, split-kernel edition. Replaces the single
 * per-token kernel: at small batch (decode) one-token-per-block left most of
 * the 40 AIVs idle; this split fills the machine at any B and degrades to
 * the v1 mapping (one token per block) once B >= core count.
 *
 *   z1 kernel: z1[s][t][r] = scale * sum_h x[t,h] * A_s[slot,r,h]
 *     grid: (slice, token, rank-group) units; each block streams its rank
 *     group's A rows over H1 tiles with the x tile cast once per tile.
 *     z1 is staged to a GM fp32 workspace [S][B][R] (32B-aligned writes).
 *   z2 kernel: y[t, off+j] (+)= sum_r z1[s][t][r] * B_s[slot,j,r]
 *     grid: (token, chunk) units over the CONCATENATED output range of up to
 *     4 slices (qkv / gate_up / o_proj / qkvz); chunks are multiples of the
 *     W-tile quantum (8192/R) so tiles never split across blocks; a chunk may
 *     straddle slice boundaries (handled by per-slice intersection). z2 ports
 *     bgmv_expand's repeat-mask structure (dup z1 across a 256B repeat, Mul
 *     with src0RepStride=0, Block/PairReduceSum tree).
 *
 * Constraints (enforced by add_lora_eligible in torch_binding.cpp):
 *   - R in {8, 16, 32, 64}: must divide 64 (z2 replicates z1 into a
 *     64-element repeat; also selects the reduce-tree shape below) and be a
 *     multiple of 8 (z1out row writes / z1 loads must stay 32B-aligned).
 *     R=8 reduces with a single BlockReduceSum (one output per 32B block),
 *     mirroring bgmv_expand's rank-8 path.
 *   - H1 / every H2_s / y width are multiples of 16; <= 4 slices;
 *     offset_start == 0 (the kernels address y from column 0)
 *   - indices are int64; -1 skips the token (overwrite mode zeroes it)
 * Native fp16/bf16 (template T), fp32 accumulate, deterministic.
 *
 * EDITING NOTE: these kernels are parsed by the build's auto_gen tool —
 * pointer parameters MUST be written `__gm__ void* name` (star on the type,
 * not the name) or the generated launchers dereference them; and any kernel
 * signature change requires wiping the generated chain (build/.../auto_gen,
 * include/vllm_ascend_kernels, the *_precompile/preprocess/aic/aiv
 * device-prefix dirs) or stale launchers poison the rebuild.
 */

#include "kernel_operator.h"
#include "types.h"

namespace {
constexpr uint32_t TILE_H = 4096;               // x / A-row tile (z1 phase)
constexpr uint32_t Z1_RANK_SUB = 4;             // batch size for MTE2 load
constexpr uint32_t Z1_TOK_SUB = 4;              // tokens per z1 unit-run
constexpr uint32_t W_IN_TILE = 8192;            // B elements per z2 compute tile
constexpr uint32_t Y_OUT_TILE = 4096;           // max outputs per block (tmpY_ size)
constexpr uint32_t NUM_BYTES_PER_REPEAT = 256;  // vector unit read granularity
constexpr uint32_t NUM_BLOCKS_PER_REPEAT = 8;
constexpr uint32_t NUM_ELEMENTS_PER_REPEAT = NUM_BYTES_PER_REPEAT / sizeof(float);
constexpr uint32_t BLOCK_REDUCE_NUM_REPEATS = W_IN_TILE / NUM_ELEMENTS_PER_REPEAT;
constexpr uint32_t PAIR_REDUCE_NUM_REPEATS_16 =
    (BLOCK_REDUCE_NUM_REPEATS * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1)
    / NUM_ELEMENTS_PER_REPEAT;
constexpr uint32_t PAIR_REDUCE_NUM_REPEATS_32 = (PAIR_REDUCE_NUM_REPEATS_16 + 1) / 2;
constexpr uint32_t R_MAX = 64;
// reduce-tree strides (values mirror bgmv_expand's reduceSumParams):
// dst advances one 32B block per repeat; src advances 8 blocks (= 256B repeat)
constexpr uint16_t REDUCE_DST_REP_STRIDE = 1;
constexpr uint16_t REDUCE_SRC_BLK_STRIDE = 1;
constexpr uint16_t REDUCE_SRC_REP_STRIDE = 8;
}  // namespace

// ============================ kernel 1: z1 =================================
template <typename scalar_t>
class AddLoraZ1 {
public:
    using T = scalar_t;

    __aicore__ inline AddLoraZ1(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *x, __gm__ void *wa0, __gm__ void *wa1,
                                __gm__ void *wa2, __gm__ void *wa3, __gm__ void *indices,
                                __gm__ void* z1out, uint32_t batch, uint32_t units,
                                uint32_t unitsPerCore, uint32_t H1, uint32_t R,
                                uint32_t rankBlock, float scale)
    {
        batch_ = batch;
        units_ = units;
        unitsPerCore_ = unitsPerCore;
        H1_ = H1;
        R_ = R;
        rankBlock_ = rankBlock;
        scale_ = scale;
        wa_[0].SetGlobalBuffer((__gm__ T *)wa0);
        wa_[1].SetGlobalBuffer((__gm__ T *)wa1);
        wa_[2].SetGlobalBuffer((__gm__ T *)wa2);
        wa_[3].SetGlobalBuffer((__gm__ T *)wa3);
        groups_ = (R_ + rankBlock_ - 1) / rankBlock_;

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, batch);
        z1Gm_.SetGlobalBuffer((__gm__ float *)z1out);

        pipe_->InitBuffer(inQueueX_, 1, Z1_TOK_SUB * TILE_H * sizeof(T));
        pipe_->InitBuffer(inQueueWA_, 1, Z1_RANK_SUB * TILE_H * sizeof(T));
        pipe_->InitBuffer(tmpX_, TILE_H * sizeof(float));
        pipe_->InitBuffer(tmpWA_, Z1_RANK_SUB * TILE_H * sizeof(float));
        pipe_->InitBuffer(z1Prod_, TILE_H * sizeof(float));
        pipe_->InitBuffer(z1Stage_, Z1_TOK_SUB * R_MAX * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t end = (int64_t)(blockIdx + 1) * unitsPerCore_;
        if (end > (int64_t)units_) {
            end = units_;
        }
        uint32_t L = 1;
        for (int64_t u = (int64_t)blockIdx * unitsPerCore_; u < end; u += L) {
            // unit -> (slice, token, rank group); units are planned only for
            // ACTIVE slices (h2 != 0), so s < nSlices <= 4 always holds
            const int64_t perSlice = (int64_t)batch_ * groups_;
            uint32_t s = (uint32_t)(u / perSlice);
            int64_t rem = u % perSlice;
            int64_t t = rem / groups_;
            uint32_t g = (uint32_t)(rem % groups_);
            L = 1;
            int64_t slot = indicesGm_.GetValue(t);
            if (slot < 0) {
                continue;  // no-lora token: z2 skips it too
            }
            // run of following units sharing this slice, rank group and slot, so
            // one A load serves all of them; L == 1 is the old behaviour
            int64_t tok[Z1_TOK_SUB];
            tok[0] = t;
            while (L < Z1_TOK_SUB && u + (int64_t)L < end) {
                int64_t uu = u + (int64_t)L;
                if ((uint32_t)(uu / perSlice) != s) {
                    break;
                }
                int64_t rem2 = uu % perSlice;
                if ((uint32_t)(rem2 % groups_) != g) {
                    break;
                }
                int64_t t2 = rem2 / groups_;
                if (indicesGm_.GetValue(t2) != slot) {
                    break;
                }
                tok[L] = t2;
                ++L;
            }
            uint32_t r0 = g * rankBlock_;
            uint32_t r1 = r0 + rankBlock_;
            if (r1 > R_) {
                r1 = R_;
            }
            AscendC::LocalTensor<float> stage = z1Stage_.Get<float>();
            for (uint32_t j = 0; j < L; ++j) {
                AscendC::LocalTensor<float> stageJ = stage[(int32_t)(j * R_MAX)];
                for (uint32_t r = 0; r < (r1 - r0); ++r) {
                    stageJ.SetValue(r, 0.0f);
                }
            }
            const int64_t aBase = slot * (int64_t)R_ * H1_;
            for (uint32_t h0 = 0; h0 < H1_; h0 += TILE_H) {
                uint32_t n = (H1_ - h0 < TILE_H) ? (H1_ - h0) : TILE_H;
                AscendC::LocalTensor<T> xAll = inQueueX_.AllocTensor<T>();
                for (uint32_t j = 0; j < L; ++j) {
                    DataCopy(xAll[j * n], xGm_[tok[j] * H1_ + h0], n);
                }
                inQueueX_.EnQue(xAll);
                xAll = inQueueX_.DeQue<T>();
                AscendC::LocalTensor<float> xF = tmpX_.Get<float>();
                AscendC::LocalTensor<float> wF = tmpWA_.Get<float>();
                AscendC::LocalTensor<float> prod = z1Prod_.Get<float>();
                for (uint32_t rb = r0; rb < r1; rb += Z1_RANK_SUB) {
                    uint32_t sub = (r1 - rb < Z1_RANK_SUB) ? (r1 - rb) : Z1_RANK_SUB;
                    AscendC::LocalTensor<T> wLocal = inQueueWA_.AllocTensor<T>();
                    if (H1_ <= TILE_H) {
                        DataCopy(wLocal, wa_[s][aBase + (int64_t)rb * H1_], sub * n);
                    } else {
                        // we are copying 4 rows, but each rows is longer than TILE_H
                        // TODO: replace it with 1 strided datacppy 
                        for (uint32_t i = 0; i < sub; ++i)
                            DataCopy(wLocal[i * n], wa_[s][aBase + (int64_t)(rb + i) * H1_ + h0], n);
                    }
                    inQueueWA_.EnQue(wLocal);
                    wLocal = inQueueWA_.DeQue<T>();
                    Cast(wF, wLocal, AscendC::RoundMode::CAST_NONE, sub * n);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueueWA_.FreeTensor(wLocal);
                    for (uint32_t j = 0; j < L; ++j) {
                        AscendC::LocalTensor<float> stageJ = stage[(int32_t)(j * R_MAX)];
                        Cast(xF, xAll[j * n], AscendC::RoundMode::CAST_NONE, n);
                        AscendC::PipeBarrier<PIPE_V>();
                        for (uint32_t i = 0; i < sub; ++i) {
                            Mul(prod, xF, wF[i * n], n);
                            AscendC::PipeBarrier<PIPE_V>();
                            ReduceSum<float>(prod, prod, prod, n);
                            AscendC::PipeBarrier<PIPE_V>();
                            stageJ.SetValue(rb + i - r0,
                                            stageJ.GetValue(rb + i - r0) + prod.GetValue(0));
                        }
                    }
                }
                inQueueX_.FreeTensor(xAll);
            }
            // fold scale (fp32, same rounding as the v1 vector Muls) and
            // write this group's segment to the GM workspace (32B-aligned:
            // rankBlock is a multiple of 8 and R % 8 == 0)
            for (uint32_t j = 0; j < L; ++j) {
                AscendC::LocalTensor<float> stageJ = stage[(int32_t)(j * R_MAX)];
                for (uint32_t r = 0; r < (r1 - r0); ++r) {
                    stageJ.SetValue(r, stageJ.GetValue(r) * scale_);
                }
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(z1Gm_[((int64_t)s * batch_ + tok[j]) * R_ + r0], stageJ, r1 - r0);
            }
        }
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_, inQueueWA_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpX_, tmpWA_, z1Stage_, z1Prod_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> wa_[4];
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<float> z1Gm_;
    uint32_t batch_, units_, unitsPerCore_, H1_, R_, rankBlock_, groups_;
    float scale_;
};

// ============================ kernel 2: z2 =================================
template <typename scalar_t>
class AddLoraZ2 {
public:
    using T = scalar_t;

    __aicore__ inline AddLoraZ2(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *z1in, __gm__ void *wb0, __gm__ void *wb1,
                                __gm__ void *wb2, __gm__ void *wb3, __gm__ void *indices,
                                __gm__ void *y, uint32_t batch, uint32_t units,
                                uint32_t unitsPerCore, uint32_t R, uint32_t h2_0,
                                uint32_t h2_1, uint32_t h2_2, uint32_t h2_3, uint32_t chunk,
                                uint32_t chunksPerToken, uint32_t yWidth, uint32_t addInputs)
    {
        batch_ = batch;
        units_ = units;
        unitsPerCore_ = unitsPerCore;
        R_ = R;
        chunk_ = chunk;
        chunksPerToken_ = chunksPerToken;
        yWidth_ = yWidth;
        addInputs_ = addInputs;
        h2_[0] = h2_0;
        h2_[1] = h2_1;
        h2_[2] = h2_2;
        h2_[3] = h2_3;
        offs_[0] = 0;
        offs_[1] = h2_0;
        offs_[2] = h2_0 + h2_1;
        offs_[3] = h2_0 + h2_1 + h2_2;
        totalH2_ = offs_[3] + h2_[3];
        wb_[0].SetGlobalBuffer((__gm__ T *)wb0);
        wb_[1].SetGlobalBuffer((__gm__ T *)wb1);
        wb_[2].SetGlobalBuffer((__gm__ T *)wb2);
        wb_[3].SetGlobalBuffer((__gm__ T *)wb3);

        z1Gm_.SetGlobalBuffer((__gm__ float *)z1in);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, batch);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        pipe_->InitBuffer(z1Buf_, R_MAX * sizeof(float));
        pipe_->InitBuffer(dupBuf_, NUM_ELEMENTS_PER_REPEAT * sizeof(float));
        pipe_->InitBuffer(inQueueW_, 1, W_IN_TILE * sizeof(T));
        pipe_->InitBuffer(tmpW_, W_IN_TILE * sizeof(float));
        pipe_->InitBuffer(inQueueY_, 1, Y_OUT_TILE * sizeof(T));
        pipe_->InitBuffer(outQueueY_, 1, Y_OUT_TILE * sizeof(T));
        pipe_->InitBuffer(tmpY_, Y_OUT_TILE * sizeof(float));
        pipe_->InitBuffer(yInF_, Y_OUT_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t end = (int64_t)(blockIdx + 1) * unitsPerCore_;
        if (end > (int64_t)units_) {
            end = units_;
        }
        for (int64_t u = (int64_t)blockIdx * unitsPerCore_; u < end; ++u) {
            int64_t t = u / chunksPerToken_;
            uint32_t c = (uint32_t)(u % chunksPerToken_);
            int64_t slot = indicesGm_.GetValue(t);
            uint32_t gs = c * chunk_;
            uint32_t ge = gs + chunk_;
            if (ge > totalH2_) {
                ge = totalH2_;
            }
            const uint64_t yRow = (uint64_t)t * yWidth_;
            if (slot < 0) {
                // no-lora token: accumulate leaves y untouched; overwrite
                // zeroes this block's chunk across all slices (chunk-local
                // ZeroRow; the union over chunks covers the full row).
                if (!addInputs_) {
                    for (uint32_t s = 0; s < 4; ++s) {
                        uint32_t a, b;
                        if (IntersectSlice(s, gs, ge, a, b)) {
                            ZeroRange(yRow + a, b - a);
                        }
                    }
                }
                continue;
            }
            // process the chunk's intersection with each slice (z1 is
            // per-slice: A_s differs, so each intersection loads its own
            // z1 row from the [S][B][R] workspace)
            for (uint32_t s = 0; s < 4; ++s) {
                uint32_t a, b;
                if (IntersectSlice(s, gs, ge, a, b)) {
                    RangeSlice(t, s, slot, a - offs_[s], b - offs_[s], yRow + a);
                }
            }
        }
    }

private:
    // intersect the chunk [gs, ge) with slice s's range; empty for padding
    // slices (h2 == 0 gives b == offs_[s] <= a) — no separate skip needed
    __aicore__ inline bool IntersectSlice(uint32_t s, uint32_t gs, uint32_t ge, uint32_t &a,
                                           uint32_t &b)
    {
        a = gs > offs_[s] ? gs : offs_[s];
        b = ge < (offs_[s] + h2_[s]) ? ge : (offs_[s] + h2_[s]);
        return a < b;
    }

    // z1 workspace is [S][B][R]: each slice intersection reads its own row,
    // then replicates it into one 256B repeat for the repeat-mask dot
    __aicore__ inline void PrepareZ1(int64_t t, uint32_t s)
    {
        AscendC::LocalTensor<float> z1 = z1Buf_.Get<float>();
        AscendC::LocalTensor<float> dup = dupBuf_.Get<float>();
        DataCopy(z1, z1Gm_[((int64_t)s * batch_ + t) * R_], R_);
        AscendC::PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < NUM_ELEMENTS_PER_REPEAT; i += R_) {
            for (uint32_t j = 0; j < R_; ++j) {
                dup.SetValue(i + j, z1.GetValue(j));
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInW(int64_t slot, uint32_t slice, uint32_t wElemOff,
                                   int32_t numElements)
    {
        AscendC::LocalTensor<T> wLocal = inQueueW_.AllocTensor<T>();
        const uint64_t slotOff = (uint64_t)slot * h2_[slice] * R_;
        DataCopy(wLocal, wb_[slice][slotOff + wElemOff], numElements);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void CopyInY(uint32_t yAddr, int32_t numElements)
    {
        AscendC::LocalTensor<T> yLocal = inQueueY_.AllocTensor<T>();
        DataCopy(yLocal, yGm_[yAddr], numElements);
        inQueueY_.EnQue(yLocal);
    }

    // dot(dup vs one B tile) -> yTile[progress...]; bgmv_expand Compute
    __aicore__ inline void ComputeTile(int32_t progress,
                                       int32_t blockReduceRepeatCount = BLOCK_REDUCE_NUM_REPEATS,
                                       int32_t pairReduceRepeat16 = PAIR_REDUCE_NUM_REPEATS_16,
                                       int32_t pairReduceRepeat32 = PAIR_REDUCE_NUM_REPEATS_32)
    {
        AscendC::LocalTensor<float> yLocal = tmpY_.Get<float>();
        AscendC::LocalTensor<float> dup = dupBuf_.Get<float>();
        AscendC::LocalTensor<T> wLocal = inQueueW_.DeQue<T>();
        AscendC::LocalTensor<float> wTmp = tmpW_.Get<float>();

        Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, NUM_ELEMENTS_PER_REPEAT, blockReduceRepeatCount,
             castParams_);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueW_.FreeTensor(wLocal);

        Mul(wTmp, dup, wTmp, NUM_ELEMENTS_PER_REPEAT, blockReduceRepeatCount, dotProductParams_);
        AscendC::PipeBarrier<PIPE_V>();

        if (R_ == 8) {
            // one output per 32B block: the intra-block reduction IS the
            // full dot (bgmv_expand's rank-8 shape)
            BlockReduceSum(yLocal[progress], wTmp, blockReduceRepeatCount,
                           NUM_ELEMENTS_PER_REPEAT, REDUCE_DST_REP_STRIDE,
                           REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (R_ == 16) {
            BlockReduceSum(wTmp, wTmp, blockReduceRepeatCount, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(yLocal[progress], wTmp, pairReduceRepeat16, NUM_ELEMENTS_PER_REPEAT,
                          REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
                          REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (R_ == 32) {
            BlockReduceSum(wTmp, wTmp, blockReduceRepeatCount, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(wTmp, wTmp, pairReduceRepeat16, NUM_ELEMENTS_PER_REPEAT,
                          REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
                          REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(yLocal[progress], wTmp, pairReduceRepeat32, NUM_ELEMENTS_PER_REPEAT,
                          REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE,
                          REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        } else {  // R_ == 64
            BlockReduceSum(wTmp, wTmp, blockReduceRepeatCount, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            BlockReduceSum(yLocal[progress], wTmp, pairReduceRepeat16, NUM_ELEMENTS_PER_REPEAT,
                           REDUCE_DST_REP_STRIDE, REDUCE_SRC_BLK_STRIDE, REDUCE_SRC_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // y[t, yAddr..yAddr+len) (+)= tmpY (cast fp32 -> T)
    __aicore__ inline void ScaleOutput(int32_t numElements)
    {
        AscendC::LocalTensor<float> yLocal = tmpY_.Get<float>();
        if (addInputs_) {
            AscendC::LocalTensor<T> yInLocal = inQueueY_.DeQue<T>();
            AscendC::LocalTensor<float> yInF = yInF_.Get<float>();
            Cast(yInF, yInLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueY_.FreeTensor(yInLocal);
            Add(yLocal, yLocal, yInF, numElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::LocalTensor<T> yOutLocal = outQueueY_.AllocTensor<T>();
        Cast(yOutLocal, yLocal, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY_.EnQue(yOutLocal);
    }

    __aicore__ inline void CopyOut(uint32_t yAddr, int32_t numElements)
    {
        AscendC::LocalTensor<T> yOutLocal = outQueueY_.DeQue<T>();
        DataCopy(yGm_[yAddr], yOutLocal, numElements);
        outQueueY_.FreeTensor(yOutLocal);
    }

    __aicore__ inline void ZeroRange(uint32_t yAddr, uint32_t len)
    {
        AscendC::LocalTensor<T> yOut = outQueueY_.AllocTensor<T>();
        Duplicate(yOut, (T)0, len);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY_.EnQue(yOut);
        yOut = outQueueY_.DeQue<T>();
        DataCopy(yGm_[yAddr], yOut, len);
        outQueueY_.FreeTensor(yOut);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // Process outputs [la, lb) of slice `s`, writing y at yAddr (= row + la).
    // Unified full-tile + remainder path (merges v1's main loop and
    // ComputeLastIteration); start offsets are multiples of 16 (chunk and
    // slice boundaries), W reads at la*R elements are 32B-aligned, and the
    // remainder W count rem*R is a multiple of 64 (rem % 16, R % 8).
    __aicore__ inline void RangeSlice(int64_t t, uint32_t s, int64_t slot, uint32_t la,
                                      uint32_t lb, uint32_t yAddr)
    {
        PrepareZ1(t, s);
        const uint32_t len = lb - la;
        const uint32_t outPerTile = W_IN_TILE / R_;
        if (addInputs_) {
            CopyInY(yAddr, len);
        }
        uint32_t nFull = len / outPerTile;
        uint32_t rem = len - nFull * outPerTile;
        for (uint32_t i = 0; i < nFull; ++i) {
            CopyInW(slot, s, (la + i * outPerTile) * R_, W_IN_TILE);
            ComputeTile(i * outPerTile);
        }
        if (rem != 0) {
            uint32_t remW = rem * R_;
            CopyInW(slot, s, (la + nFull * outPerTile) * R_, remW);
            int32_t lastRepeatCount = remW / NUM_ELEMENTS_PER_REPEAT;
            int32_t pair16 = (lastRepeatCount * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1)
                             / NUM_ELEMENTS_PER_REPEAT;
            int32_t pair32 = (pair16 + 1) / 2;
            ComputeTile(nFull * outPerTile, lastRepeatCount, pair16, pair32);
        }
        ScaleOutput(len);
        CopyOut(yAddr, len);
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueW_, inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> z1Buf_, dupBuf_, tmpW_, tmpY_, yInF_;
    AscendC::GlobalTensor<T> wb_[4];
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<float> z1Gm_;
    uint32_t batch_, units_, unitsPerCore_, R_, chunk_, chunksPerToken_, yWidth_, addInputs_;
    uint32_t h2_[4];
    uint32_t offs_[4];
    uint32_t totalH2_;

    // repeat layouts copied from bgmv_expand (block=32B, 8 blocks per
    // repeat); reduce strides live in the named constants above
    AscendC::UnaryRepeatParams castParams_ = {1, 1, 8, 4};
    AscendC::BinaryRepeatParams dotProductParams_ = {1, 1, 1, 8, 0, 8};
};

#define ADD_LORA_Z1_DECLARE(TYPE)                                                                        \
    extern "C" __global__ __aicore__ void add_lora_z1_##TYPE(                                            \
        __gm__ void* x, __gm__ void* wa0, __gm__ void* wa1, __gm__ void* wa2, __gm__ void* wa3,          \
        __gm__ void* indices, __gm__ void* z1out, uint32_t batch, uint32_t units, uint32_t unitsPerCore, \
        uint32_t H1, uint32_t R, uint32_t rankBlock, float scale)                                        \
    {                                                                                                    \
        AscendC::TPipe pipe;                                                                             \
        AddLoraZ1<TYPE> op(&pipe);                                                                       \
        op.Init(x, wa0, wa1, wa2, wa3, indices, z1out, batch, units, unitsPerCore, H1, R, rankBlock,     \
                scale);                                                                                  \
        op.Process();                                                                                    \
    }

#define ADD_LORA_Z2_DECLARE(TYPE)                                                                        \
    extern "C" __global__ __aicore__ void add_lora_z2_##TYPE(                                            \
        __gm__ void* z1in, __gm__ void* wb0, __gm__ void* wb1, __gm__ void* wb2, __gm__ void* wb3,       \
        __gm__ void* indices, __gm__ void* y, uint32_t batch, uint32_t units, uint32_t unitsPerCore,     \
        uint32_t R, uint32_t h2_0, uint32_t h2_1, uint32_t h2_2, uint32_t h2_3, uint32_t chunk,          \
        uint32_t chunksPerToken, uint32_t yWidth, uint32_t addInputs)                                    \
    {                                                                                                    \
        AscendC::TPipe pipe;                                                                             \
        AddLoraZ2<TYPE> op(&pipe);                                                                       \
        op.Init(z1in, wb0, wb1, wb2, wb3, indices, y, batch, units, unitsPerCore, R, h2_0, h2_1, h2_2,    \
                h2_3, chunk, chunksPerToken, yWidth, addInputs);                                         \
        op.Process();                                                                                    \
    }

ADD_LORA_Z1_DECLARE(half)
ADD_LORA_Z2_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ADD_LORA_Z1_DECLARE(bfloat16_t)
ADD_LORA_Z2_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {

// Host-side grid planning shared by both launches.
struct AddLoraGrid {
    uint32_t units;
    uint32_t unitsPerCore;
    uint32_t gridDim;
};

static inline AddLoraGrid PlanGrid(uint32_t units, uint32_t aivNum)
{
    AddLoraGrid g;
    g.units = units;
    g.unitsPerCore = (units + aivNum - 1) / aivNum;
    if (g.unitsPerCore == 0) {
        g.unitsPerCore = 1;
    }
    g.gridDim = (units + g.unitsPerCore - 1) / g.unitsPerCore;
    return g;
}

// Host-side entry. Slice descriptors arrive as plain arrays (wa/wb/h2, first
// nSlices entries valid, nSlices <= 4); the fixed-signature AscendC launchers
// are fed from them at the single unrolled launch site below.
extern void add_lora_fused_impl(AscendType type, void *stream, void *x,
                                void *const *wa, void *const *wb, void *indices, void *y,
                                void *z1ws, uint32_t batch, uint32_t H1, uint32_t R,
                                const uint32_t *h2, uint32_t nSlices, uint32_t yWidth,
                                float scale, uint32_t addInputs, uint32_t aivNum)
{
    // ---- z1: (slice, token, rank-group) units; rank groups sized to fill
    // the machine at small B (rankBlock multiple of 8 keeps GM writes
    // 32B-aligned; R % 8 == 0 keeps the tail group aligned too)
    uint32_t targetGroups = (aivNum + batch - 1) / batch;
    if (targetGroups < 1) {
        targetGroups = 1;
    }
    if (targetGroups > R) {
        targetGroups = R;
    }
    uint32_t rankBlock = (R + targetGroups - 1) / targetGroups;
    rankBlock = (rankBlock + 7) / 8 * 8;  // round up to multiple of 8
    if (rankBlock > R) {
        rankBlock = R;
    }
    uint32_t groups = (R + rankBlock - 1) / rankBlock;
    AddLoraGrid g1 = PlanGrid(nSlices * batch * groups, aivNum);

    // ---- z2: (token, chunk) units over the concatenated output range;
    // chunk is a multiple of the W-tile quantum (8192/R) and capped at
    // Y_OUT_TILE (tmpY_ size)
    const uint32_t totalH2 = h2[0] + h2[1] + h2[2] + h2[3];
    const uint32_t outPerTile = W_IN_TILE / R;
    uint32_t wantChunks = (aivNum + batch - 1) / batch;
    if (wantChunks < 1) {
        wantChunks = 1;
    }
    // outPerTile in {128, 256, 512, 1024} (R in {8,16,32,64}) divides
    // Y_OUT_TILE, so a plain cap preserves the W-tile quantum — no alignment
    // fixups needed
    uint32_t chunk = outPerTile * ((totalH2 + wantChunks * outPerTile - 1) / (wantChunks * outPerTile));
    if (chunk > Y_OUT_TILE) {
        chunk = Y_OUT_TILE;
    }
    const uint32_t chunksPerToken = (totalH2 + chunk - 1) / chunk;
    AddLoraGrid g2 = PlanGrid(batch * chunksPerToken, aivNum);

    // pad descriptors to the fixed 4-slot launcher signature: unused slots
    // repeat slot 0 (harmless — z2 skips h2 == 0 slices)
    void *waf[4];
    void *wbf[4];
    uint32_t h2f[4];
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t src = i < nSlices ? i : 0;
        waf[i] = wa[src];
        wbf[i] = wb[src];
        h2f[i] = i < nSlices ? h2[i] : 0;
    }

    if (type == AscendType::FP16) {
        add_lora_z1_half<<<g1.gridDim, nullptr, stream>>>(
            x, waf[0], waf[1], waf[2], waf[3], indices, z1ws, batch, g1.units, g1.unitsPerCore,
            H1, R, rankBlock, scale);
        add_lora_z2_half<<<g2.gridDim, nullptr, stream>>>(
            z1ws, wbf[0], wbf[1], wbf[2], wbf[3], indices, y, batch, g2.units, g2.unitsPerCore,
            R, h2f[0], h2f[1], h2f[2], h2f[3], chunk, chunksPerToken, yWidth, addInputs);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        add_lora_z1_bfloat16_t<<<g1.gridDim, nullptr, stream>>>(
            x, waf[0], waf[1], waf[2], waf[3], indices, z1ws, batch, g1.units, g1.unitsPerCore,
            H1, R, rankBlock, scale);
        add_lora_z2_bfloat16_t<<<g2.gridDim, nullptr, stream>>>(
            z1ws, wbf[0], wbf[1], wbf[2], wbf[3], indices, y, batch, g2.units, g2.unitsPerCore,
            R, h2f[0], h2f[1], h2f[2], h2f[3], chunk, chunksPerToken, yWidth, addInputs);
#endif
    }
}
}  // namespace vllm_ascend
