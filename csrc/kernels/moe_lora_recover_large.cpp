// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include "kernel_operator.h"
#include "moe_lora_recover_large.h"

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
                              uint64_t topK, uint64_t slotCount) {
    rows_ = rows;
    topK_ = topK;
    slotCount_ = slotCount;
    expanded_.SetGlobalBuffer(reinterpret_cast<__gm__ Index*>(expanded));
    expert_.SetGlobalBuffer(reinterpret_cast<__gm__ Expert*>(expert));
    slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
    scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(scratch));
    pipe_.InitBuffer(expandedBuf_, TILE * sizeof(Index));
    pipe_.InitBuffer(expertBuf_, TILE * sizeof(Expert));
    pipe_.InitBuffer(slotsBuf_, (TILE + 4) * sizeof(int64_t));
    // Each 16B payload starts at a 32B-aligned UB address.
    pipe_.InitBuffer(pairBuf_, TILE * 32);
  }

  __aicore__ inline void Process() {
    auto expanded = expandedBuf_.Get<Index>();
    auto expert = expertBuf_.Get<Expert>();
    auto slots = slotsBuf_.Get<int64_t>();
    auto pairs = pairBuf_.Get<int64_t>();
    for (uint32_t base = AscendC::GetBlockIdx() * TILE; base < rows_; base += AscendC::GetBlockNum() * TILE) {
      const uint32_t count = rows_ - base < TILE ? rows_ - base : TILE;
      const uint64_t firstToken = base / topK_;
      const uint64_t lastToken = (static_cast<uint64_t>(base) + count - 1) / topK_;
      const uint64_t firstSlot = firstToken < slotCount_ ? firstToken : slotCount_ - 1;
      const uint64_t lastSlot = lastToken < slotCount_ ? lastToken : slotCount_ - 1;
      CopyIn(expanded, expanded_[base], count);
      CopyIn(expert, expert_[base], count);
      CopyIn(slots, slots_[firstSlot], static_cast<uint32_t>(lastSlot - firstSlot + 1));
      Sync<AscendC::HardEvent::MTE2_S>();
      uint32_t p = 0;
      uint64_t token = firstToken;
      uint64_t remaining = topK_ - base % topK_;
      while (p < count) {
        const uint64_t slot = token < slotCount_ ? token : slotCount_ - 1;
        const int64_t value = slots.GetValue(static_cast<uint32_t>(slot - firstSlot));
        const uint32_t group = remaining < count - p ? static_cast<uint32_t>(remaining) : count - p;
        for (uint32_t j = 0; j < group; ++j) {
          pairs.SetValue((p + j) * 4, static_cast<int64_t>(expert.GetValue(p + j)));
          pairs.SetValue((p + j) * 4 + 1, value);
        }
        p += group;
        ++token;
        remaining = topK_;
      }
      Sync<AscendC::HardEvent::S_MTE3>();
      for (uint32_t i = 0; i < count; ++i) {
        const int64_t signedDestination = static_cast<int64_t>(expanded.GetValue(i));
        const uint64_t destination = signedDestination < 0 ? uint64_t{0} - static_cast<uint64_t>(signedDestination)
                                                           : static_cast<uint64_t>(signedDestination);
        if (destination < rows_) {
          // One writer per 64B destination line. Exactly 16 valid
          // bytes are written; padding is never selected as payload.
          AscendC::DataCopyPad(scratch_[destination * 8], pairs[i * 4], {1, 16, 0, 0, 0});
        }
      }
      Sync<AscendC::HardEvent::MTE3_S>();
      Sync<AscendC::HardEvent::S_MTE2>();
    }
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> expandedBuf_, expertBuf_, slotsBuf_, pairBuf_;
  AscendC::GlobalTensor<Index> expanded_;
  AscendC::GlobalTensor<Expert> expert_;
  AscendC::GlobalTensor<int64_t> slots_, scratch_;
  uint32_t rows_;
  uint64_t topK_, slotCount_;
};

class RecoverPack {
 public:
  __aicore__ inline void Init(GM_ADDR scratch, GM_ADDR expert, GM_ADDR slots, uint32_t rows) {
    rows_ = rows;
    scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(scratch));
    expert_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(expert));
    slots_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(slots));
    pipe_.InitBuffer(inputBuf_, TILE * 64);
    pipe_.InitBuffer(offsetBuf_, TILE * 16);
    pipe_.InitBuffer(outputBuf_, TILE * 8);
  }
  __aicore__ inline void Process() {
    using namespace AscendC;
    auto input = inputBuf_.Get<uint32_t>();
    auto offsets = offsetBuf_.Get<uint32_t>();
    auto slotOffsets = offsets[TILE * 2];
    auto output = outputBuf_.Get<uint32_t>();
    for (uint32_t i = 0; i < TILE; ++i) {
      offsets.SetValue(2 * i, i * 64);
      offsets.SetValue(2 * i + 1, i * 64 + 4);
      slotOffsets.SetValue(2 * i, i * 64 + 8);
      slotOffsets.SetValue(2 * i + 1, i * 64 + 12);
    }
    Sync<HardEvent::S_V>();
    for (uint32_t base = GetBlockIdx() * TILE; base < rows_; base += GetBlockNum() * TILE) {
      const uint32_t count = rows_ - base < TILE ? rows_ - base : TILE;
      CopyIn(input, scratch_[base * 16], count * 16);
      Sync<HardEvent::MTE2_V>();
      Gather(output, input, offsets, 0, count * 2);
      Sync<HardEvent::V_MTE3>();
      DataCopyPad(expert_[base * 2], output, {1, count * 8, 0, 0, 0});
      Sync<HardEvent::MTE3_V>();
      Gather(output, input, slotOffsets, 0, count * 2);
      Sync<HardEvent::V_MTE3>();
      DataCopyPad(slots_[base * 2], output, {1, count * 8, 0, 0, 0});
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
                                             uint32_t rows, uint64_t topK, uint64_t slotCount) {               \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                            \
    RecoverScatter<I, E> op;                                                                                   \
    op.Init(expanded, expert, slots, scratch, rows, topK, slotCount);                                          \
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
#define LAUNCH(NAME)                                                                             \
  NAME<<<cores, nullptr, stream>>>(static_cast<GM_ADDR>(expanded), static_cast<GM_ADDR>(expert), \
                                   static_cast<GM_ADDR>(slots), static_cast<GM_ADDR>(scratch), rows, topK, slotCount)
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
  // Pack holds 64B input, 16B offsets, and 8B output per row.
  constexpr uint64_t ubReserve = 4096;
  constexpr uint64_t packBytes = uint64_t{TILE} * (64 + 16 + 8);
  return rows >= MOE_LORA_RECOVER_LARGE_MIN_ROWS && rows <= MOE_LORA_RECOVER_LARGE_MAX_ROWS &&
         ubBytes >= packBytes + ubReserve;
}
