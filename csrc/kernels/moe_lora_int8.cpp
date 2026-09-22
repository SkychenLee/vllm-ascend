// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include "kernel_operator.h"
#include "moe_lora_int8.h"

using namespace AscendC;
namespace {
constexpr uint32_t SHRINK_TILE = 4096;
constexpr uint32_t EXPAND_TILE = 256;
constexpr uint32_t MAX_RANK = 512;
constexpr uint32_t MAX_WIDTH = 8192;

template <HardEvent event>
__aicore__ inline void Sync() {
  SetFlag<event>(EVENT_ID0);
  WaitFlag<event>(EVENT_ID0);
}
template <typename T>
__aicore__ inline void Load(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count) {
  Sync<HardEvent::V_MTE2>();
  DataCopyPad(dst, src, DataCopyExtParams{1, count * sizeof(T), 0, 0, 0}, DataCopyPadExtParams<T>{false, 0, 0, 0});
  Sync<HardEvent::MTE2_V>();
}
template <typename T>
__aicore__ inline void Store(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
  Sync<HardEvent::V_MTE3>();
  Sync<HardEvent::S_MTE3>();
  DataCopyPad(dst, src, DataCopyExtParams{1, count * sizeof(T), 0, 0, 0});
  Sync<HardEvent::MTE3_V>();
  Sync<HardEvent::MTE3_S>();
}
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace

template <typename T, bool PAIRED = false>
class Int8Shrink {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR w, GM_ADDR ids, GM_ADDR scale, GM_ADDR out, uint32_t rows,
                             uint32_t hidden, uint32_t rank, uint32_t groups) {
    const uint32_t outputRank = PAIRED ? rank * 2 : rank;
    GlobalTensor<int8_t> xgm;
    xgm.SetGlobalBuffer((__gm__ int8_t*)x);
    GlobalTensor<T> wgm;
    wgm.SetGlobalBuffer((__gm__ T*)w);
    GlobalTensor<int64_t> igm;
    igm.SetGlobalBuffer((__gm__ int64_t*)ids);
    GlobalTensor<float> sgm, ygm;
    sgm.SetGlobalBuffer((__gm__ float*)scale);
    ygm.SetGlobalBuffer((__gm__ float*)out);
    pipe.InitBuffer(xq, SHRINK_TILE);
    pipe.InitBuffer(xh, SHRINK_TILE * 2);
    pipe.InitBuffer(xf, SHRINK_TILE * 4);
    pipe.InitBuffer(wb, SHRINK_TILE * 2);
    pipe.InitBuffer(wf, SHRINK_TILE * 4);
    pipe.InitBuffer(tmp, SHRINK_TILE * 4);
    pipe.InitBuffer(yb, MAX_RANK * (PAIRED ? 2 : 1) * 8 * 4);
    pipe.InitBuffer(meta, 128);
    auto qi = xq.Get<int8_t>();
    auto halfx = xh.Get<half>();
    auto xx = xf.Get<float>();
    auto ww = wb.Get<T>();
    auto ff = wf.Get<float>();
    auto scratch = tmp.Get<float>();
    auto output = yb.Get<float>();
    auto index = meta.Get<int64_t>();
    auto scales = meta.Get<float>()[16];
    // Decode uses the smallest row unit whose FP32 output is 32-byte
    // aligned. Larger batches amortize metadata DMA over eight rows.
    uint32_t unit = 8;
    if (rows <= GetBlockNum() * 8)
      unit = outputRank % 8 == 0 ? 1 : outputRank % 4 == 0 ? 2 : outputRank % 2 == 0 ? 4 : 8;
    for (uint32_t start = GetBlockIdx() * unit; start < rows; start += GetBlockNum() * unit) {
      uint32_t count = Min(unit, rows - start);
      Sync<HardEvent::S_MTE2>();
      Load(index, igm[start], count);
      Load(scales, sgm[start], count);
      Duplicate(output, 0.0f, (count * outputRank + 7) / 8 * 8);
      Sync<HardEvent::MTE2_S>();
      Sync<HardEvent::V_S>();
      for (uint32_t row = start; row < Min(start + unit, rows); ++row) {
        auto yy = output[(row - start) * outputRank];
        int64_t id = index.GetValue(row - start);
        float s = scales.GetValue(row - start);
        if (id >= 0 && id < groups) {
          // Short power-of-two K: keep the activation in UB and reduce a
          // batch of rank rows together. Odd ranks and long/tail K retain
          // the general tiled path below.
          if (hidden >= 64 && hidden <= 1024 && (hidden & (hidden - 1)) == 0 && rank % 8 == 0) {
            Load(qi, xgm[(uint64_t)row * hidden], hidden);
            Cast(halfx, qi, RoundMode::CAST_NONE, hidden);
            PipeBarrier<PIPE_V>();
            Cast(xx, halfx, RoundMode::CAST_NONE, hidden);
            PipeBarrier<PIPE_V>();
            Muls(xx, xx, s, hidden);
            PipeBarrier<PIPE_V>();
            uint32_t rankTile = Min(rank, SHRINK_TILE / hidden);
            for (uint32_t j = 1; j < rankTile; ++j) DataCopy(xx[j * hidden], xx, hidden);
            PipeBarrier<PIPE_V>();
            for (uint32_t first = 0; first < outputRank;) {
              uint32_t localFirst = first;
              uint64_t weightRow = (uint64_t)id * rank + first;
              if constexpr (PAIRED) {
                localFirst = first % rank;
                weightRow = (uint64_t)(first / rank) * groups * rank + id * rank + localFirst;
              }
              uint32_t nr = Min(rankTile, rank - localFirst);
              Load(ww, wgm[weightRow * hidden], nr * hidden);
              Cast(ff, ww, RoundMode::CAST_NONE, nr * hidden);
              PipeBarrier<PIPE_V>();
              Mul(ff, ff, xx, nr * hidden);
              PipeBarrier<PIPE_V>();
              auto src = ff;
              auto dst = scratch;
              for (uint32_t width = hidden; width > 1;) {
                uint32_t elements = nr * width;
                if (width >= 8) {
                  BlockReduceSum(dst, src, (elements + 63) / 64, Min(64, elements), 1, 1, 8);
                  width /= 8;
                } else {
                  PairReduceSum(dst, src, (elements + 63) / 64, Min(64, elements), 1, 1, 8);
                  width /= 2;
                }
                PipeBarrier<PIPE_V>();
                auto previous = src;
                src = dst;
                dst = previous;
              }
              if (nr >= 8) {
                DataCopy(yy[first], src, nr);
                PipeBarrier<PIPE_V>();
              } else {
                // K=1024 leaves fewer than one 32-byte rank block.
                // Stage these scalars in the row-owned output, then perform
                // the same aligned GM store as the general path.
                Sync<HardEvent::V_S>();
                for (uint32_t j = 0; j < nr; ++j) yy.SetValue(first + j, src.GetValue(j));
                Sync<HardEvent::S_V>();
              }
              first += nr;
            }
            continue;
          }
          for (uint32_t k = 0; k < hidden; k += SHRINK_TILE) {
            uint32_t n = Min(SHRINK_TILE, hidden - k);
            Load(qi, xgm[(uint64_t)row * hidden + k], n);
            Cast(halfx, qi, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            Cast(xx, halfx, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            Muls(xx, xx, s, n);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < outputRank; ++r) {
              uint64_t weightRow = (uint64_t)id * rank + r;
              if constexpr (PAIRED) weightRow = (uint64_t)(r / rank) * groups * rank + id * rank + r % rank;
              Load(ww, wgm[weightRow * hidden + k], n);
              Cast(ff, ww, RoundMode::CAST_NONE, n);
              PipeBarrier<PIPE_V>();
              Mul(ff, ff, xx, n);
              PipeBarrier<PIPE_V>();
              ReduceSum(scratch, ff, scratch, n);
              Sync<HardEvent::V_S>();
              yy.SetValue(r, yy.GetValue(r) + scratch.GetValue(0));
            }
          }
        }
      }
      Store(ygm[(uint64_t)start * outputRank], output, count * outputRank);
    }
  }

 private:
  TPipe pipe;
  TBuf<TPosition::VECCALC> xq, xh, xf, wb, wf, tmp, yb, meta;
};

template <typename T, bool PAIRED = false>
class ExpandSwigluQuant {
 public:
  __aicore__ inline void Run(GM_ADDR base, GM_ADDR gate, GM_ADDR up, GM_ADDR bg, GM_ADDR bu, GM_ADDR ids, GM_ADDR topk,
                             GM_ADDR out, GM_ADDR scales, uint32_t rows, uint32_t hidden, uint32_t rank,
                             uint32_t groups, float limit, uint32_t shards = 1) {
    tp = shards;
    m = rows;
    h = hidden;
    r = rank;
    basegm.SetGlobalBuffer((__gm__ T*)base);
    bgm[0].SetGlobalBuffer((__gm__ T*)bg);
    bgm[1].SetGlobalBuffer((__gm__ T*)bu);
    agm[0].SetGlobalBuffer((__gm__ float*)gate);
    agm[1].SetGlobalBuffer((__gm__ float*)up);
    igm.SetGlobalBuffer((__gm__ int64_t*)ids);
    tgm.SetGlobalBuffer((__gm__ float*)topk);
    ygm.SetGlobalBuffer((__gm__ int8_t*)out);
    sgm.SetGlobalBuffer((__gm__ float*)scales);
    bool vectorRank = r >= 8 && (r & (r - 1)) == 0;
    uint32_t columnTile = vectorRank ? Min(EXPAND_TILE, EXPAND_TILE * 64 / r) : EXPAND_TILE;
    if constexpr (PAIRED) {
      // Leave room for cached gate/up ranks and their gather offsets at
      // the largest activation widths. The ordinary model tile is unchanged.
      if (hidden > 2048 && vectorRank) columnTile /= 2;
    }
    uint32_t weightElements = vectorRank ? columnTile * r : MAX_RANK;
    if constexpr (PAIRED) {
      // Reuse weight scratch for padded, rank-major AllGather fragments.
      if (weightElements < r * 8) weightElements = r * 8;
    }
    pipe.InitBuffer(wb, weightElements * 2);
    pipe.InitBuffer(wf, weightElements * 4);
    pipe.InitBuffer(ab, MAX_RANK * (PAIRED ? 2 : 1) * 4);
    if constexpr (PAIRED) pipe.InitBuffer(offsets, MAX_RANK * 2 * 4);
    pipe.InitBuffer(dup, 64 * 4);
    pipe.InitBuffer(bbuf, EXPAND_TILE * 2);
    pipe.InitBuffer(vals, EXPAND_TILE * 4 * 4);
    uint32_t widthAligned = (h + 255) / 256 * 256;
    pipe.InitBuffer(activated, widthAligned * 4);
    pipe.InitBuffer(tmp, (widthAligned > MAX_RANK ? widthAligned : MAX_RANK) * 4);
    pipe.InitBuffer(castbuf, widthAligned * 2);
    pipe.InitBuffer(outbuf, widthAligned);
    pipe.InitBuffer(meta, 544);
    auto idx = meta.Get<int64_t>();
    auto routeScales = meta.Get<float>()[64];
    auto outputScales = meta.Get<float>()[96];
    auto scalar = meta.Get<float>()[128];
    auto a = activated.Get<float>();
    auto work = tmp.Get<float>();
    auto g = vals.Get<float>();
    auto u = vals.Get<float>()[EXPAND_TILE];
    auto den = vals.Get<float>()[EXPAND_TILE * 2];
    if constexpr (PAIRED) {
      auto indices = offsets.Get<uint32_t>();
      uint32_t localRank = r / tp;
      uint32_t paddedPair = (2 * localRank + 7) / 8 * 8;
      uint32_t paddedRank = (r + 7) / 8 * 8;
      for (uint32_t j = 0; j < 2 * paddedRank; ++j) {
        uint32_t plane = j / paddedRank, rankIndex = j % paddedRank;
        if (rankIndex >= r) rankIndex = 0;
        indices.SetValue(j, ((rankIndex / localRank) * paddedPair + plane * localRank + rankIndex % localRank) * 4);
      }
      Sync<HardEvent::S_V>();
    }
    // A whole 32-row unit owns both byte outputs (including odd H) and
    // FP32 scales. Aligned H only needs eight rows per unit.
    uint32_t rowsPerUnit = (h % 32 == 0) ? 8 : 32;
    for (uint32_t start = GetBlockIdx() * rowsPerUnit; start < rows; start += GetBlockNum() * rowsPerUnit) {
      uint32_t count = Min(rowsPerUnit, rows - start);
      Sync<HardEvent::S_MTE2>();
      Load(idx, igm[start], count);
      if (topk != nullptr) Load(routeScales, tgm[start], count);
      Sync<HardEvent::MTE2_S>();
      for (uint32_t row = start; row < Min(start + rowsPerUnit, rows); ++row) {
        int64_t id = idx.GetValue(row - start);
        float route = 1.0f;
        if (topk != nullptr) {
          route = routeScales.GetValue(row - start);
        }
        if constexpr (PAIRED) {
          if (id >= 0 && id < groups) LoadPaired(row);
        }
        for (uint32_t col = 0; col < h; col += columnTile) {
          uint32_t n = Min(columnTile, h - col);
          Project(g, 0, row, col, n, id, groups);
          Project(u, 1, row, col, n, id, groups);
          if (limit > 0) {
            Mins(g, g, limit, n);
            Mins(u, u, limit, n);
            PipeBarrier<PIPE_V>();
            Maxs(u, u, -limit, n);
            PipeBarrier<PIPE_V>();
            // torch.clamp writes the model dtype, including nonrepresentable
            // scalar limits. Retain that boundary before the activation.
            RoundFloat(g, n);
            RoundFloat(u, n);
          }
          Muls(den, g, -1.0f, n);
          PipeBarrier<PIPE_V>();
          Exp(den, den, n);
          PipeBarrier<PIPE_V>();
          Adds(den, den, 1.0f, n);
          PipeBarrier<PIPE_V>();
          Div(g, g, den, n);
          PipeBarrier<PIPE_V>();
          Mul(g, g, u, n);
          PipeBarrier<PIPE_V>();
          RoundFloat(g, n);
          if (topk != nullptr) {
            Muls(g, g, route, n);
            PipeBarrier<PIPE_V>();
            RoundFloat(g, n);
          }
          // Tile offsets and padded copies are whole 32-byte UB blocks.
          DataCopy(a[col], g, (n + 7) / 8 * 8);
          PipeBarrier<PIPE_V>();
        }
        Abs(work, a, h);
        PipeBarrier<PIPE_V>();
        ReduceMax(scalar, work, work, h, false);
        Sync<HardEvent::V_S>();
        float maxval = scalar.GetValue(0);
        float scale = maxval / 127.0f;
        outputScales.SetValue(row - start, scale);
        if (maxval > 0) {
          Muls(a, a, 127.0f / maxval, h);
          PipeBarrier<PIPE_V>();
        }
        // Round in FP32 before FP16 -> INT8 to avoid double rounding.
        auto ints = tmp.Get<int32_t>();
        Cast(ints, a, RoundMode::CAST_RINT, h);
        PipeBarrier<PIPE_V>();
        Cast(a, ints, RoundMode::CAST_NONE, h);
        PipeBarrier<PIPE_V>();
        auto halfout = castbuf.Get<half>();
        auto q = outbuf.Get<int8_t>();
        Cast(halfout, a, RoundMode::CAST_NONE, h);
        PipeBarrier<PIPE_V>();
        Cast(q, halfout, RoundMode::CAST_RINT, h);
        PipeBarrier<PIPE_V>();
        Store(ygm[(uint64_t)row * h], q, h);
      }
      Store(sgm[start], outputScales, count);
    }
  }

 private:
  __aicore__ inline void RoundFloat(LocalTensor<float> x, uint32_t n) {
    auto low = castbuf.Get<T>();
    Cast(low, x, RoundMode::CAST_RINT, n);
    PipeBarrier<PIPE_V>();
    Cast(x, low, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void LoadPaired(uint32_t row) {
    uint32_t localRank = r / tp;
    auto staging = wf.Get<float>();
    Sync<HardEvent::V_MTE2>();
    DataCopyPad(staging, agm[0][(uint64_t)row * 2 * localRank],
                DataCopyExtParams{static_cast<uint16_t>(tp), 2 * localRank * 4, (m - 1) * 2 * localRank * 4, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    Sync<HardEvent::MTE2_V>();
    Gather(ab.Get<float>(), staging, offsets.Get<uint32_t>(), 0, 2 * ((r + 7) / 8 * 8));
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Project(LocalTensor<float> result, uint32_t p, uint32_t row, uint32_t col, uint32_t n,
                                 int64_t id, uint32_t groups) {
    auto base = bbuf.Get<T>();
    auto basef = vals.Get<float>()[EXPAND_TILE * 3];
    Load(base, basegm[(uint64_t)row * h * 2 + p * h + col], n);
    Cast(basef, base, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    if (id < 0 || id >= groups) {
      DataCopy(result, basef, (n + 7) / 8 * 8);
      PipeBarrier<PIPE_V>();
      return;
    }
    auto av = ab.Get<float>();
    auto w = wb.Get<T>();
    auto f = wf.Get<float>();
    if constexpr (PAIRED) {
      av = ab.Get<float>()[p * ((r + 7) / 8 * 8)];
    } else {
      Load(av, agm[p][(uint64_t)row * r], r);
    }
    if (r >= 8 && (r & (r - 1)) == 0) {
      Load(w, bgm[p][((uint64_t)id * h + col) * r], n * r);
      Cast(f, w, RoundMode::CAST_NONE, n * r);
      PipeBarrier<PIPE_V>();
      uint32_t count = (n * r + 63) / 64;
      if (r <= 64) {
        auto repeated = dup.Get<float>();
        for (uint32_t j = 0; j < 64; j += r) DataCopy(repeated[j], av, r);
        PipeBarrier<PIPE_V>();
        // Each 64-element vector repeats the same LoRA rank vector.
        for (uint32_t rep = 0; rep < count; rep += 128) {
          Mul(f[rep * 64], f[rep * 64], repeated, 64, Min(128, count - rep), BinaryRepeatParams{1, 1, 1, 8, 8, 0});
        }
      } else {
        // Large ranks use strided repeats over columns; cap columnTile so
        // the BF16/FP16 and FP32 weight buffers keep the same UB budget.
        for (uint32_t k = 0; k < r; k += 64) {
          Mul(f[k], f[k], av[k], 64, n,
              BinaryRepeatParams{1, 1, 1, static_cast<uint8_t>(r / 8), static_cast<uint8_t>(r / 8), 0});
        }
      }
      PipeBarrier<PIPE_V>();
      for (uint32_t rep = 0; rep < count; rep += 128) {
        auto reduced = r == 8 ? result[rep * 8] : f[rep * 8];
        BlockReduceSum(reduced, f[rep * 64], Min(128, count - rep), 64, 1, 1, 8);
      }
      PipeBarrier<PIPE_V>();
      uint32_t partials = count * 8;
      for (uint32_t width = r / 8; width > 1;) {
        if (width >= 8) {
          BlockReduceSum(width == 8 ? result : f, f, (partials + 63) / 64, 64, 1, 1, 8);
          width /= 8;
          partials /= 8;
        } else {
          PairReduceSum(width == 2 ? result : f, f, (partials + 63) / 64, 64, 1, 1, 8);
          width /= 2;
          partials /= 2;
        }
        PipeBarrier<PIPE_V>();
      }
    } else {
      // Non-power-of-two and TP-local small ranks retain independent
      // reductions; full ranks 8 through 512 use the vector path.
      auto reduce = tmp.Get<float>();
      for (uint32_t j = 0; j < n; ++j) {
        // Gather into an aligned scratch row for unaligned ranks.
        Load(w, bgm[p][((uint64_t)id * h + col + j) * r], r);
        Cast(f, w, RoundMode::CAST_NONE, r);
        PipeBarrier<PIPE_V>();
        Mul(f, f, av, r);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduce, f, reduce, r);
        Sync<HardEvent::V_S>();
        result.SetValue(j, reduce.GetValue(0));
      }
      Sync<HardEvent::S_V>();
    }
    Add(result, result, basef, n);
    PipeBarrier<PIPE_V>();
    RoundFloat(result, n);
  }
  TPipe pipe;
  TBuf<TPosition::VECCALC> wb, wf, ab, dup, bbuf, vals, activated, tmp, castbuf, outbuf, meta, offsets;
  GlobalTensor<T> basegm, bgm[2];
  GlobalTensor<float> agm[2], tgm, sgm;
  GlobalTensor<int64_t> igm;
  GlobalTensor<int8_t> ygm;
  uint32_t h, r, tp, m;
};

#define DECLARE_LORA_INT8(T)                                                                                        \
  extern "C" __global__ __aicore__ void lora_shrink_int8_##T(GM_ADDR x, GM_ADDR w, GM_ADDR i, GM_ADDR s, GM_ADDR y, \
                                                             uint32_t m, uint32_t k, uint32_t r, uint32_t groups) { \
    Int8Shrink<T> op;                                                                                               \
    op.Run(x, w, i, s, y, m, k, r, groups);                                                                         \
  }                                                                                                                 \
  extern "C" __global__ __aicore__ void lora_expand_swiglu_quant_##T(                                               \
      GM_ADDR base, GM_ADDR g, GM_ADDR u, GM_ADDR bg, GM_ADDR bu, GM_ADDR i, GM_ADDR t, GM_ADDR y, GM_ADDR s,       \
      uint32_t m, uint32_t h, uint32_t r, uint32_t groups, float limit) {                                           \
    ExpandSwigluQuant<T> op;                                                                                        \
    op.Run(base, g, u, bg, bu, i, t, y, s, m, h, r, groups, limit);                                                 \
  }
DECLARE_LORA_INT8(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
DECLARE_LORA_INT8(bfloat16_t)
#endif

#define DECLARE_LORA_PAIR(T)                                                                                        \
  extern "C" __global__ __aicore__ void lora_shrink_int8_pair_##T(                                                  \
      GM_ADDR x, GM_ADDR w, GM_ADDR i, GM_ADDR s, GM_ADDR y, uint32_t m, uint32_t k, uint32_t r, uint32_t groups) { \
    Int8Shrink<T, true> op;                                                                                         \
    op.Run(x, w, i, s, y, m, k, r, groups);                                                                         \
  }                                                                                                                 \
  extern "C" __global__ __aicore__ void lora_expand_swiglu_quant_pair_##T(                                          \
      GM_ADDR base, GM_ADDR a, GM_ADDR bg, GM_ADDR bu, GM_ADDR i, GM_ADDR t, GM_ADDR y, GM_ADDR s, uint32_t m,      \
      uint32_t h, uint32_t r, uint32_t groups, float limit, uint32_t tp) {                                          \
    ExpandSwigluQuant<T, true> op;                                                                                  \
    op.Run(base, a, a, bg, bu, i, t, y, s, m, h, r, groups, limit, tp);                                             \
  }
DECLARE_LORA_PAIR(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
DECLARE_LORA_PAIR(bfloat16_t)
#endif

namespace vllm_ascend {
void bgmv_shrink_int8_pair_impl(AscendType type, void* stream, void* x, void* w, void* ids, void* s, void* y,
                                uint32_t rows, uint32_t hidden, uint32_t rank, uint32_t groups, uint32_t cores) {
  uint32_t outputRank = rank * 2;
  uint32_t unit = 8;
  if (rows <= cores * 8) unit = outputRank % 8 == 0 ? 1 : outputRank % 4 == 0 ? 2 : outputRank % 2 == 0 ? 4 : 8;
  uint32_t grid = (rows + unit - 1) / unit;
  if (grid > cores) grid = cores;
  if (type == AscendType::FP16)
    lora_shrink_int8_pair_half<<<grid, nullptr, stream>>>((GM_ADDR)x, (GM_ADDR)w, (GM_ADDR)ids, (GM_ADDR)s, (GM_ADDR)y,
                                                          rows, hidden, rank, groups);
  else {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    lora_shrink_int8_pair_bfloat16_t<<<grid, nullptr, stream>>>((GM_ADDR)x, (GM_ADDR)w, (GM_ADDR)ids, (GM_ADDR)s,
                                                                (GM_ADDR)y, rows, hidden, rank, groups);
#endif
  }
}
void moe_lora_expand_swiglu_quant_pair_impl(AscendType type, void* stream, void* base, void* a, void* bg, void* bu,
                                            void* ids, void* topk, void* y, void* s, uint32_t rows, uint32_t hidden,
                                            uint32_t rank, uint32_t groups, uint32_t cores, float limit, uint32_t tp) {
  uint32_t unit = hidden % 32 == 0 ? 8 : 32;
  uint32_t grid = (rows + unit - 1) / unit;
  if (grid > cores) grid = cores;
  if (type == AscendType::FP16)
    lora_expand_swiglu_quant_pair_half<<<grid, nullptr, stream>>>((GM_ADDR)base, (GM_ADDR)a, (GM_ADDR)bg, (GM_ADDR)bu,
                                                                  (GM_ADDR)ids, (GM_ADDR)topk, (GM_ADDR)y, (GM_ADDR)s,
                                                                  rows, hidden, rank, groups, limit, tp);
  else {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    lora_expand_swiglu_quant_pair_bfloat16_t<<<grid, nullptr, stream>>>(
        (GM_ADDR)base, (GM_ADDR)a, (GM_ADDR)bg, (GM_ADDR)bu, (GM_ADDR)ids, (GM_ADDR)topk, (GM_ADDR)y, (GM_ADDR)s, rows,
        hidden, rank, groups, limit, tp);
#endif
  }
}
void bgmv_shrink_int8_impl(AscendType type, void* stream, void* x, void* w, void* ids, void* s, void* y, uint32_t rows,
                           uint32_t hidden, uint32_t rank, uint32_t groups, uint32_t cores) {
  uint32_t unit = 8;
  if (rows <= cores * 8) unit = rank % 8 == 0 ? 1 : rank % 4 == 0 ? 2 : rank % 2 == 0 ? 4 : 8;
  uint32_t grid = (rows + unit - 1) / unit;
  if (grid > cores) grid = cores;
  if (type == AscendType::FP16)
    lora_shrink_int8_half<<<grid, nullptr, stream>>>((GM_ADDR)x, (GM_ADDR)w, (GM_ADDR)ids, (GM_ADDR)s, (GM_ADDR)y, rows,
                                                     hidden, rank, groups);
  else {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    lora_shrink_int8_bfloat16_t<<<grid, nullptr, stream>>>((GM_ADDR)x, (GM_ADDR)w, (GM_ADDR)ids, (GM_ADDR)s, (GM_ADDR)y,
                                                           rows, hidden, rank, groups);
#endif
  }
}
void moe_lora_expand_swiglu_quant_impl(AscendType type, void* stream, void* base, void* g, void* u, void* bg, void* bu,
                                       void* ids, void* topk, void* y, void* s, uint32_t rows, uint32_t hidden,
                                       uint32_t rank, uint32_t groups, uint32_t cores, float limit) {
  uint32_t unit = hidden % 32 == 0 ? 8 : 32;
  uint32_t grid = (rows + unit - 1) / unit;
  if (grid > cores) grid = cores;
  if (type == AscendType::FP16)
    lora_expand_swiglu_quant_half<<<grid, nullptr, stream>>>((GM_ADDR)base, (GM_ADDR)g, (GM_ADDR)u, (GM_ADDR)bg,
                                                             (GM_ADDR)bu, (GM_ADDR)ids, (GM_ADDR)topk, (GM_ADDR)y,
                                                             (GM_ADDR)s, rows, hidden, rank, groups, limit);
  else {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    lora_expand_swiglu_quant_bfloat16_t<<<grid, nullptr, stream>>>((GM_ADDR)base, (GM_ADDR)g, (GM_ADDR)u, (GM_ADDR)bg,
                                                                   (GM_ADDR)bu, (GM_ADDR)ids, (GM_ADDR)topk, (GM_ADDR)y,
                                                                   (GM_ADDR)s, rows, hidden, rank, groups, limit);
#endif
  }
}
}  // namespace vllm_ascend
