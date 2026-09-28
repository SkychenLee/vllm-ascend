// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include "kernel_operator.h"
#include "moe_lora_recover_large.h"
#include <cmath>
#include <limits>

namespace {
constexpr uint32_t TILE = MOE_LORA_RECOVER_LARGE_TILE;
template <AscendC::HardEvent Event>
__aicore__ inline void Sync() {
  AscendC::SetFlag<Event>(0);
  AscendC::WaitFlag<Event>(0);
}
template <typename T>
__aicore__ inline void CopyIn(AscendC::LocalTensor<T> local, AscendC::GlobalTensor<T> global, uint32_t count) {
  AscendC::DataCopyPad(local, global, {1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                       AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
}
}  // namespace

namespace {
template <typename Index, typename Expert>
class RecoverScatter {
 public:
  __aicore__ inline void Init(GM_ADDR expanded, GM_ADDR expert, GM_ADDR slots, GM_ADDR scratch, uint32_t rows,
                              uint64_t topK, uint64_t slotCount, float reciprocal) {
    rows_ = rows;
    topK_ = topK;
    slotCount_ = slotCount;
    reciprocal_ = reciprocal;
    expanded_.SetGlobalBuffer(reinterpret_cast<__gm__ Index*>(expanded));
    expert_.SetGlobalBuffer(reinterpret_cast<__gm__ Expert*>(expert));
    slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
    scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(scratch));
    pipe_.InitBuffer(expandedBuf_, TILE * sizeof(Index));
    pipe_.InitBuffer(payloadBuf_, (2 * TILE + 4) * sizeof(int64_t));
    // Each 16B payload starts at a 32B-aligned UB address.
    pipe_.InitBuffer(pairBuf_, TILE * 32);
    pipe_.InitBuffer(vectorBuf_, TILE * 160);
  }

  __aicore__ inline void Process() {
    auto expanded = expandedBuf_.Get<Index>();
    auto payload = payloadBuf_.Get<uint32_t>();
    auto expert = payload.ReinterpretCast<Expert>();
    auto slots = payload[2 * TILE].ReinterpretCast<int64_t>();
    auto pairs = pairBuf_.Get<int64_t>();
    using namespace AscendC;
    // Each 32B UB record contains the four payload words twice. Generate
    // their Gather addresses directly: this device has no vector Scatter.
    // The 16B GM write below selects only the first copy of the payload.
    constexpr uint32_t WORDS = 8 * TILE;
    auto row = vectorBuf_.Get<int32_t>();
    auto slotMask = row[WORDS];
    auto slotWord = row[2 * WORDS];
    auto expertOffset = row[3 * WORDS];
    auto address = row[4 * WORDS];
    ArithProgression(address, int32_t{0}, int32_t{1}, WORDS);
    PipeBarrier<PIPE_V>();
    ShiftRight(row, address, int32_t{3}, WORDS);
    ShiftRight(expertOffset, address, int32_t{2}, WORDS);
    PipeBarrier<PIPE_V>();
    ShiftLeft(expertOffset, expertOffset, int32_t{2}, WORDS);
    PipeBarrier<PIPE_V>();
    Sub(slotWord, address, expertOffset, WORDS);  // word lane modulo four
    PipeBarrier<PIPE_V>();
    ShiftRight(slotMask, slotWord, int32_t{1}, WORDS);
    PipeBarrier<PIPE_V>();
    ShiftLeft(expertOffset, slotMask, int32_t{1}, WORDS);
    PipeBarrier<PIPE_V>();
    Sub(slotWord, slotWord, expertOffset, WORDS);  // low/high word in int64
    PipeBarrier<PIPE_V>();
    if constexpr (sizeof(Expert) == 4) {
      ShiftLeft(expertOffset, row, int32_t{2}, WORDS);
      Muls(address, slotWord, int32_t{TILE * 4}, WORDS);
    } else {
      ShiftLeft(expertOffset, row, int32_t{3}, WORDS);
      ShiftLeft(address, slotWord, int32_t{2}, WORDS);
    }
    PipeBarrier<PIPE_V>();
    Add(expertOffset, expertOffset, address, WORDS);
    ShiftLeft(slotWord, slotWord, int32_t{2}, WORDS);
    PipeBarrier<PIPE_V>();
    Adds(slotWord, slotWord, int32_t{TILE * 8}, WORDS);
    Sync<HardEvent::V_MTE2>();
    for (uint32_t base = GetBlockIdx() * TILE; base < rows_; base += GetBlockNum() * TILE) {
      const uint32_t count = rows_ - base < TILE ? rows_ - base : TILE;
      const uint64_t firstToken = base / topK_;
      const uint64_t lastToken = (static_cast<uint64_t>(base) + count - 1) / topK_;
      const uint64_t firstSlot = firstToken < slotCount_ ? firstToken : slotCount_ - 1;
      const uint64_t lastSlot = lastToken < slotCount_ ? lastToken : slotCount_ - 1;
      CopyIn(expanded, expanded_[base], count);
      CopyIn(expert, expert_[base], count);
      CopyIn(slots, slots_[firstSlot], static_cast<uint32_t>(lastSlot - firstSlot + 1));
      Sync<HardEvent::MTE2_V>();
      Sync<HardEvent::MTE2_S>();
      auto expertWords = expert.template ReinterpretCast<int32_t>();
      if constexpr (sizeof(Expert) == 4) {
        ShiftRight(expertWords[TILE], expertWords, int32_t{31}, count);
        PipeBarrier<PIPE_V>();
      }
      // Only routing positions use FP32. topK <= rows <= 2**18 and
      // x < topK+TILE bound the upward reciprocal's error below 1/topK;
      // integer quotients cannot round below themselves. Payloads are
      // gathered as raw 32-bit halves and retain every int64 bit.
      auto tokenFloat = address.ReinterpretCast<float>();
      Cast(tokenFloat, row, RoundMode::CAST_ROUND, count * 8);
      PipeBarrier<PIPE_V>();
      Adds(tokenFloat, tokenFloat, static_cast<float>(static_cast<int32_t>(base % topK_)), count * 8);
      PipeBarrier<PIPE_V>();
      Muls(tokenFloat, tokenFloat, reciprocal_, count * 8);
      PipeBarrier<PIPE_V>();
      Mins(tokenFloat, tokenFloat, static_cast<float>(static_cast<int32_t>(lastSlot - firstSlot)), count * 8);
      PipeBarrier<PIPE_V>();
      Cast(address, tokenFloat, RoundMode::CAST_FLOOR, count * 8);
      PipeBarrier<PIPE_V>();
      ShiftLeft(address, address, int32_t{3}, count * 8);
      PipeBarrier<PIPE_V>();
      Add(address, address, slotWord, count * 8);
      PipeBarrier<PIPE_V>();
      Sub(address, address, expertOffset, count * 8);
      PipeBarrier<PIPE_V>();
      Mul(address, address, slotMask, count * 8);
      PipeBarrier<PIPE_V>();
      Add(address, address, expertOffset, count * 8);
      PipeBarrier<PIPE_V>();
      // Keep each Gather's count within the source tensor extent. CANN's
      // CPU model truncates a larger gather at that extent, despite repeated
      // in-range addresses. Splitting preserves the same vector payload.
      constexpr uint32_t GATHER_WORDS = 4 * TILE;
      const uint32_t words = count * 8;
      const uint32_t firstWords = words < GATHER_WORDS ? words : GATHER_WORDS;
      auto pairWords = pairs.ReinterpretCast<uint32_t>();
      auto addresses = address.ReinterpretCast<uint32_t>();
      Gather(pairWords, payload, addresses, 0, firstWords);
      if (words > GATHER_WORDS) {
        Gather(pairWords[GATHER_WORDS], payload, addresses[GATHER_WORDS], 0, words - GATHER_WORDS);
      }
      Sync<HardEvent::V_MTE3>();
      for (uint32_t i = 0; i < count; ++i) {
        const int64_t signedDestination = static_cast<int64_t>(expanded.GetValue(i));
        const uint64_t destination = signedDestination < 0 ? uint64_t{0} - static_cast<uint64_t>(signedDestination)
                                                           : static_cast<uint64_t>(signedDestination);
        if (destination < rows_) {
          DataCopyPad(scratch_[destination * 8], pairs[i * 4], {1, 16, 0, 0, 0});
        }
      }
      Sync<HardEvent::MTE3_V>();
      Sync<HardEvent::V_MTE2>();
    }
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> expandedBuf_, payloadBuf_, pairBuf_, vectorBuf_;
  AscendC::GlobalTensor<Index> expanded_;
  AscendC::GlobalTensor<Expert> expert_;
  AscendC::GlobalTensor<int64_t> slots_, scratch_;
  uint32_t rows_;
  uint64_t topK_, slotCount_;
  float reciprocal_;
};

class RecoverPack {
 public:
  __aicore__ inline void Init(GM_ADDR scratch, GM_ADDR expert, GM_ADDR slots, uint32_t rows) {
    rows_ = rows;
    scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(scratch));
    expert_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(expert));
    slots_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(slots));
    pipe_.InitBuffer(inputBuf_, TILE * 64);
    pipe_.InitBuffer(offsetBuf_, TILE * 24);
    pipe_.InitBuffer(outputBuf_, TILE * 16);
  }
  __aicore__ inline void Process() {
    using namespace AscendC;
    auto input = inputBuf_.Get<uint32_t>();
    auto offsets = offsetBuf_.Get<uint32_t>();
    auto slotOffsets = offsets[TILE * 2];
    auto output = outputBuf_.Get<uint32_t>();
    auto sequence = offsets.ReinterpretCast<int32_t>();
    auto row = offsets[TILE * 4].ReinterpretCast<int32_t>();
    // word i selects byte 4*i + 56*floor(i/2) from each 64B record.
    ArithProgression(sequence, int32_t{0}, int32_t{1}, TILE * 2);
    PipeBarrier<PIPE_V>();
    ShiftRight(row, sequence, int32_t{1}, TILE * 2);
    PipeBarrier<PIPE_V>();
    Muls(row, row, int32_t{56}, TILE * 2);
    ShiftLeft(sequence, sequence, int32_t{2}, TILE * 2);
    PipeBarrier<PIPE_V>();
    Add(sequence, sequence, row, TILE * 2);
    PipeBarrier<PIPE_V>();
    Adds(slotOffsets.ReinterpretCast<int32_t>(), sequence, int32_t{8}, TILE * 2);
    PipeBarrier<PIPE_V>();
    auto slotOutput = output[TILE * 2];
    for (uint32_t base = GetBlockIdx() * TILE; base < rows_; base += GetBlockNum() * TILE) {
      const uint32_t count = rows_ - base < TILE ? rows_ - base : TILE;
      CopyIn(input, scratch_[base * 16], count * 16);
      Sync<HardEvent::MTE2_V>();
      Gather(output, input, offsets, 0, count * 2);
      Gather(slotOutput, input, slotOffsets, 0, count * 2);
      Sync<HardEvent::V_MTE3>();
      DataCopyPad(expert_[base * 2], output, {1, count * 8, 0, 0, 0});
      DataCopyPad(slots_[base * 2], slotOutput, {1, count * 8, 0, 0, 0});
      Sync<HardEvent::MTE3_V>();
      Sync<HardEvent::V_MTE2>();
    }
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_, offsetBuf_, outputBuf_;
  AscendC::GlobalTensor<uint32_t> scratch_, expert_, slots_;
  uint32_t rows_;
};

}  // namespace

#define SCATTER_KERNEL(NAME, I, E)                                                                             \
  extern "C" __global__ __aicore__ void NAME(GM_ADDR expanded, GM_ADDR expert, GM_ADDR slots, GM_ADDR scratch, \
                                             uint32_t rows, uint64_t topK, uint64_t slotCount, float reciprocal) { \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                            \
    RecoverScatter<I, E> op;                                                                                   \
    op.Init(expanded, expert, slots, scratch, rows, topK, slotCount, reciprocal);                                          \
    op.Process();                                                                                              \
  }
SCATTER_KERNEL(moe_lora_recover_large_scatter_ii, int32_t, int32_t)
SCATTER_KERNEL(moe_lora_recover_large_scatter_il, int32_t, int64_t)
SCATTER_KERNEL(moe_lora_recover_large_scatter_li, int64_t, int32_t)
SCATTER_KERNEL(moe_lora_recover_large_scatter_ll, int64_t, int64_t)

extern "C" __global__ __aicore__ void moe_lora_recover_large_pack(GM_ADDR scratch, GM_ADDR expert, GM_ADDR slots,
                                                                  uint32_t rows) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  RecoverPack op;
  op.Init(scratch, expert, slots, rows);
  op.Process();
}

void moe_lora_recover_large_impl(void* stream, void* expanded, void* expert, void* slots, void* scratch,
                                 void* expertOut, void* slotOut, uint32_t rows, uint64_t topK, uint64_t slotCount,
                                 uint32_t indexBytes, uint32_t expertBytes, uint32_t cores) {
  const float reciprocal = std::nextafter(static_cast<float>(1.0 / static_cast<double>(topK)),
                                           std::numeric_limits<float>::infinity());
#define LAUNCH(NAME)                                                                             \
  NAME<<<cores, nullptr, stream>>>(static_cast<GM_ADDR>(expanded), static_cast<GM_ADDR>(expert), \
                                   static_cast<GM_ADDR>(slots), static_cast<GM_ADDR>(scratch), rows, topK, slotCount, reciprocal)
  if (indexBytes == 4 && expertBytes == 4) {
    LAUNCH(moe_lora_recover_large_scatter_ii);
  } else if (indexBytes == 4) {
    LAUNCH(moe_lora_recover_large_scatter_il);
  } else if (expertBytes == 4) {
    LAUNCH(moe_lora_recover_large_scatter_li);
  } else {
    LAUNCH(moe_lora_recover_large_scatter_ll);
  }
  moe_lora_recover_large_pack<<<cores, nullptr, stream>>>(
      static_cast<GM_ADDR>(scratch), static_cast<GM_ADDR>(expertOut), static_cast<GM_ADDR>(slotOut), rows);
}

bool moe_lora_recover_large_supported(uint64_t rows, uint64_t ubBytes) {
  // Scatter: 8B index, 16B payload, 32B pair, 160B vector work per row.
  // Pack uses 64B input + 24B offsets + 16B output per row.
  constexpr uint64_t ubReserve = 4096;
  constexpr uint64_t packBytes = uint64_t{TILE} * (8 + 16 + 32 + 160) + 32;
  return rows >= MOE_LORA_RECOVER_LARGE_MIN_ROWS && rows <= MOE_LORA_RECOVER_LARGE_MAX_ROWS &&
         ubBytes >= packBytes + ubReserve;
}
