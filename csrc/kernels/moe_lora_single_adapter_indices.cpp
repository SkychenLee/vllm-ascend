// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include "kernel_operator.h"
#include "moe_lora_single_adapter_indices.h"

namespace {
constexpr uint64_t BLOCK_BYTES = 32;
constexpr uint64_t UB_RESERVE_BYTES = 4096;
constexpr uint32_t TILE = vllm_ascend::MOE_LORA_SINGLE_ADAPTER_LARGE_TILE;
constexpr uint32_t MAX_ADAPTERS = vllm_ascend::MOE_LORA_SINGLE_ADAPTER_MAX_ADAPTERS;

constexpr uint64_t AlignBlock(uint64_t bytes) {
    return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}
__aicore__ inline uint64_t RoundBytes(uint64_t bytes) {
    return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}
template <AscendC::HardEvent Event>
__aicore__ inline void Sync() {
    AscendC::SetFlag<Event>(0);
    AscendC::WaitFlag<Event>(0);
}
template <typename T>
__aicore__ inline void CopyIn(AscendC::LocalTensor<T> local, AscendC::GlobalTensor<T> global, uint32_t count) {
    AscendC::DataCopyPad(local, global,
        AscendC::DataCopyExtParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
        AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
}
__aicore__ inline uint64_t Destination(int64_t value) {
    // Retain the TP recovery's integer abs, without signed INT64_MIN overflow.
    return value < 0 ? uint64_t{0} - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
}

class SlotCache {
public:
    __aicore__ inline void Init(int64_t firstSlot, AscendC::LocalTensor<int32_t> enabled,
        uint32_t adapters, uint64_t numExperts) {
        firstSlot_ = firstSlot;
        enabled_ = enabled;
        adapters_ = adapters;
        numExperts_ = numExperts;
        firstEnabled_ = firstSlot >= 0 && static_cast<uint64_t>(firstSlot) < adapters &&
                        enabled.GetValue(static_cast<uint32_t>(firstSlot)) != 0;
        firstBase_ = firstEnabled_ ? static_cast<uint64_t>(firstSlot) * numExperts : 0;
    }
    __aicore__ inline bool Base(int64_t slot, uint64_t& base) {
        if (slot == firstSlot_) {
            base = firstBase_;
            return firstEnabled_;
        }
        // The first slot is just a hint, never a promise that the whole batch
        // has one adapter. Every different slot keeps the generic row result.
        if (slot < 0 || static_cast<uint64_t>(slot) >= adapters_ ||
            enabled_.GetValue(static_cast<uint32_t>(slot)) == 0) {
            return false;
        }
        base = static_cast<uint64_t>(slot) * numExperts_;
        return true;
    }
private:
    AscendC::LocalTensor<int32_t> enabled_;
    int64_t firstSlot_;
    uint64_t firstBase_, numExperts_;
    uint32_t adapters_;
    bool firstEnabled_;
};

template <typename RowIndexType, typename ExpertType>
class SingleAdapterSmall {
public:
    __aicore__ inline void Init(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots, GM_ADDR enabled,
        GM_ADDR output, uint32_t rows, uint64_t topK, uint64_t slotCount,
        uint32_t adapters, uint64_t numExperts) {
        rows_ = rows;
        topK_ = topK;
        slotCount_ = slotCount;
        adapters_ = adapters;
        numExperts_ = numExperts;
        const uint64_t tokens = 1 + (static_cast<uint64_t>(rows) - 1) / topK;
        readSlots_ = static_cast<uint32_t>(slotCount < tokens ? slotCount : tokens);
        expanded_.SetGlobalBuffer(reinterpret_cast<__gm__ RowIndexType*>(expanded));
        topk_.SetGlobalBuffer(reinterpret_cast<__gm__ ExpertType*>(topk));
        slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
        enabled_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(enabled));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(output));
        pipe_.InitBuffer(expandedBuf_, static_cast<uint32_t>(RoundBytes(rows * sizeof(RowIndexType))));
        pipe_.InitBuffer(topkBuf_, static_cast<uint32_t>(RoundBytes(rows * sizeof(ExpertType))));
        pipe_.InitBuffer(slotsBuf_, static_cast<uint32_t>(RoundBytes(readSlots_ * sizeof(int64_t))));
        pipe_.InitBuffer(enabledBuf_, MAX_ADAPTERS * sizeof(int32_t));
        pipe_.InitBuffer(outputBuf_, static_cast<uint32_t>(RoundBytes(rows * sizeof(int64_t))));
    }
    __aicore__ inline void Process() {
        auto expanded = expandedBuf_.template Get<RowIndexType>();
        auto topk = topkBuf_.template Get<ExpertType>();
        auto slots = slotsBuf_.Get<int64_t>();
        auto enabled = enabledBuf_.Get<int32_t>();
        auto output = outputBuf_.Get<int64_t>();
        CopyIn(expanded, expanded_, rows_);
        CopyIn(topk, topk_, rows_);
        CopyIn(slots, slots_, readSlots_);
        CopyIn(enabled, enabled_, adapters_);
        // Deterministic holes for out-of-contract destinations in the small
        // path; malformed duplicate permutations are still unsupported.
        AscendC::Duplicate(output.ReinterpretCast<int32_t>(), int32_t{-1}, rows_ * 2);
        Sync<AscendC::HardEvent::MTE2_S>();
        Sync<AscendC::HardEvent::V_S>();
        SlotCache cache;
        cache.Init(slots.GetValue(0), enabled, adapters_, numExperts_);
        uint32_t source = 0;
        uint64_t token = 0;
        while (source < rows_) {
            const uint64_t slotIndex = token < slotCount_ ? token : slotCount_ - 1;
            uint64_t base = 0;
            const bool active = cache.Base(slots.GetValue(static_cast<uint32_t>(slotIndex)), base);
            const uint32_t remaining = rows_ - source;
            const uint32_t group = static_cast<uint32_t>(topK_ < remaining ? topK_ : remaining);
            for (uint32_t lane = 0; lane < group; ++lane) {
                const uint32_t index = source + lane;
                const uint64_t destination = Destination(static_cast<int64_t>(expanded.GetValue(index)));
                if (destination < rows_) {
                    // Unsigned arithmetic retains torch INT64 wraparound;
                    // never convert adapter/expert payloads to floating point.
                    const int64_t result = active ? static_cast<int64_t>(
                        base + static_cast<uint64_t>(static_cast<int64_t>(topk.GetValue(index)))) : int64_t{-1};
                    output.SetValue(static_cast<uint32_t>(destination), result);
                }
            }
            source += group;
            ++token;
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(output_, output,
            AscendC::DataCopyExtParams{1, rows_ * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0});
        Sync<AscendC::HardEvent::MTE3_S>();
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expandedBuf_, topkBuf_, slotsBuf_, enabledBuf_, outputBuf_;
    AscendC::GlobalTensor<RowIndexType> expanded_;
    AscendC::GlobalTensor<ExpertType> topk_;
    AscendC::GlobalTensor<int64_t> slots_, output_;
    AscendC::GlobalTensor<int32_t> enabled_;
    uint32_t rows_, readSlots_, adapters_;
    uint64_t topK_, slotCount_, numExperts_;
};

template <typename RowIndexType, typename ExpertType>
class SingleAdapterScatter {
public:
    __aicore__ inline void Init(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots, GM_ADDR enabled,
        GM_ADDR scratch, uint32_t rows, uint64_t topK, uint64_t slotCount,
        uint32_t adapters, uint64_t numExperts) {
        rows_ = rows;
        topK_ = topK;
        slotCount_ = slotCount;
        adapters_ = adapters;
        numExperts_ = numExperts;
        expanded_.SetGlobalBuffer(reinterpret_cast<__gm__ RowIndexType*>(expanded));
        topk_.SetGlobalBuffer(reinterpret_cast<__gm__ ExpertType*>(topk));
        slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
        enabled_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(enabled));
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(scratch));
        pipe_.InitBuffer(expandedBuf_, TILE * sizeof(RowIndexType));
        pipe_.InitBuffer(topkBuf_, TILE * sizeof(ExpertType));
        pipe_.InitBuffer(slotsBuf_, static_cast<uint32_t>(RoundBytes((TILE + 1) * sizeof(int64_t))));
        pipe_.InitBuffer(enabledBuf_, MAX_ADAPTERS * sizeof(int32_t));
        pipe_.InitBuffer(firstSlotBuf_, BLOCK_BYTES);
        // A 32B-aligned UB source for each 8B payload; the destination is a
        // private 64B GM record, independent of the producer's permutation.
        pipe_.InitBuffer(recordBuf_, TILE * BLOCK_BYTES);
    }
    __aicore__ inline void Process() {
        auto expanded = expandedBuf_.template Get<RowIndexType>();
        auto topk = topkBuf_.template Get<ExpertType>();
        auto slots = slotsBuf_.Get<int64_t>();
        auto enabled = enabledBuf_.Get<int32_t>();
        auto firstSlot = firstSlotBuf_.Get<int64_t>();
        auto records = recordBuf_.Get<int64_t>();
        CopyIn(enabled, enabled_, adapters_);
        CopyIn(firstSlot, slots_, 1);
        Sync<AscendC::HardEvent::MTE2_S>();
        SlotCache cache;
        cache.Init(firstSlot.GetValue(0), enabled, adapters_, numExperts_);
        for (uint32_t tileBase = AscendC::GetBlockIdx() * TILE; tileBase < rows_;
             tileBase += AscendC::GetBlockNum() * TILE) {
            const uint32_t count = rows_ - tileBase < TILE ? rows_ - tileBase : TILE;
            const uint64_t firstToken = tileBase / topK_;
            const uint64_t lastToken = (static_cast<uint64_t>(tileBase) + count - 1) / topK_;
            const uint64_t firstSlotIndex = firstToken < slotCount_ ? firstToken : slotCount_ - 1;
            const uint64_t lastSlotIndex = lastToken < slotCount_ ? lastToken : slotCount_ - 1;
            Sync<AscendC::HardEvent::S_MTE2>();
            CopyIn(expanded, expanded_[tileBase], count);
            CopyIn(topk, topk_[tileBase], count);
            CopyIn(slots, slots_[firstSlotIndex], static_cast<uint32_t>(lastSlotIndex - firstSlotIndex + 1));
            Sync<AscendC::HardEvent::MTE2_S>();
            uint32_t source = 0;
            uint64_t token = firstToken;
            while (source < count) {
                const uint64_t slotIndex = token < slotCount_ ? token : slotCount_ - 1;
                uint64_t base = 0;
                const bool active = cache.Base(slots.GetValue(static_cast<uint32_t>(slotIndex - firstSlotIndex)), base);
                const uint64_t untilNextToken = topK_ - (static_cast<uint64_t>(tileBase) + source) % topK_;
                const uint32_t remaining = count - source;
                const uint32_t group = static_cast<uint32_t>(untilNextToken < remaining ? untilNextToken : remaining);
                for (uint32_t lane = 0; lane < group; ++lane) {
                    const uint32_t index = source + lane;
                    const int64_t result = active ? static_cast<int64_t>(
                        base + static_cast<uint64_t>(static_cast<int64_t>(topk.GetValue(index)))) : int64_t{-1};
                    records.SetValue(index * 4, result);
                }
                source += group;
                ++token;
            }
            Sync<AscendC::HardEvent::S_MTE3>();
            for (uint32_t index = 0; index < count; ++index) {
                const uint64_t destination = Destination(static_cast<int64_t>(expanded.GetValue(index)));
                if (destination < rows_) {
                    AscendC::DataCopyPad(scratch_[destination * 8], records[index * 4],
                        AscendC::DataCopyExtParams{1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0});
                }
            }
            Sync<AscendC::HardEvent::MTE3_S>();
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expandedBuf_, topkBuf_, slotsBuf_, enabledBuf_, firstSlotBuf_, recordBuf_;
    AscendC::GlobalTensor<RowIndexType> expanded_;
    AscendC::GlobalTensor<ExpertType> topk_;
    AscendC::GlobalTensor<int64_t> slots_, scratch_;
    AscendC::GlobalTensor<int32_t> enabled_;
    uint32_t rows_, adapters_;
    uint64_t topK_, slotCount_, numExperts_;
};

class SingleAdapterPack {
public:
    __aicore__ inline void Init(GM_ADDR scratch, GM_ADDR output, uint32_t rows) {
        rows_ = rows;
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(scratch));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
        pipe_.InitBuffer(inputBuf_, TILE * 64);
        pipe_.InitBuffer(offsetBuf_, TILE * 16);
        pipe_.InitBuffer(outputBuf_, TILE * sizeof(int64_t));
    }
    __aicore__ inline void Process() {
        using namespace AscendC;
        auto input = inputBuf_.Get<uint32_t>();
        auto offsets = offsetBuf_.Get<int32_t>();
        auto row = offsets[TILE * 2];
        auto output = outputBuf_.Get<uint32_t>();
        // Preserve both raw words: 4*i + 56*floor(i/2) selects int64 from
        // the first eight bytes of each padded 64B destination record.
        ArithProgression(offsets, int32_t{0}, int32_t{1}, TILE * 2);
        PipeBarrier<PIPE_V>();
        ShiftRight(row, offsets, int32_t{1}, TILE * 2);
        PipeBarrier<PIPE_V>();
        Muls(row, row, int32_t{56}, TILE * 2);
        ShiftLeft(offsets, offsets, int32_t{2}, TILE * 2);
        PipeBarrier<PIPE_V>();
        Add(offsets, offsets, row, TILE * 2);
        Sync<HardEvent::V_MTE2>();
        for (uint32_t tileBase = GetBlockIdx() * TILE; tileBase < rows_; tileBase += GetBlockNum() * TILE) {
            const uint32_t count = rows_ - tileBase < TILE ? rows_ - tileBase : TILE;
            CopyIn(input, scratch_[tileBase * 16], count * 16);
            Sync<HardEvent::MTE2_V>();
            Gather(output, input, offsets.ReinterpretCast<uint32_t>(), 0, count * 2);
            Sync<HardEvent::V_MTE3>();
            // Tile starts own disjoint whole cache lines. Only the final tile
            // writes a partial line, with an exact byte count and no padding.
            DataCopyPad(output_[tileBase * 2], output,
                DataCopyExtParams{1, count * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0});
            Sync<HardEvent::MTE3_V>();
            Sync<HardEvent::V_MTE2>();
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_, offsetBuf_, outputBuf_;
    AscendC::GlobalTensor<uint32_t> scratch_, output_;
    uint32_t rows_;
};
}  // namespace

#define SINGLE_SMALL(NAME, I, E) \
extern "C" __global__ __aicore__ void NAME(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots, \
    GM_ADDR enabled, GM_ADDR output, uint32_t rows, uint64_t topK, uint64_t slotCount, \
    uint32_t adapters, uint64_t numExperts) { \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
    SingleAdapterSmall<I, E> op; \
    op.Init(expanded, topk, slots, enabled, output, rows, topK, slotCount, adapters, numExperts); \
    op.Process(); \
}
SINGLE_SMALL(moe_lora_single_small_ii, int32_t, int32_t)
SINGLE_SMALL(moe_lora_single_small_il, int32_t, int64_t)
SINGLE_SMALL(moe_lora_single_small_li, int64_t, int32_t)
SINGLE_SMALL(moe_lora_single_small_ll, int64_t, int64_t)
#undef SINGLE_SMALL

#define SINGLE_SCATTER(NAME, I, E) \
extern "C" __global__ __aicore__ void NAME(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots, \
    GM_ADDR enabled, GM_ADDR scratch, uint32_t rows, uint64_t topK, uint64_t slotCount, \
    uint32_t adapters, uint64_t numExperts) { \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
    SingleAdapterScatter<I, E> op; \
    op.Init(expanded, topk, slots, enabled, scratch, rows, topK, slotCount, adapters, numExperts); \
    op.Process(); \
}
SINGLE_SCATTER(moe_lora_single_scatter_ii, int32_t, int32_t)
SINGLE_SCATTER(moe_lora_single_scatter_il, int32_t, int64_t)
SINGLE_SCATTER(moe_lora_single_scatter_li, int64_t, int32_t)
SINGLE_SCATTER(moe_lora_single_scatter_ll, int64_t, int64_t)
#undef SINGLE_SCATTER

extern "C" __global__ __aicore__ void moe_lora_single_pack(GM_ADDR scratch, GM_ADDR output, uint32_t rows) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SingleAdapterPack op;
    op.Init(scratch, output, rows);
    op.Process();
}

namespace vllm_ascend {
bool moe_lora_single_adapter_indices_small_supported(uint64_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t indexBytes, uint32_t expertBytes, uint64_t ubBytes) {
    if ((indexBytes != 4 && indexBytes != 8) || (expertBytes != 4 && expertBytes != 8) || topK == 0) return false;
    if (rows == 0) return true;
    const uint64_t tokens = 1 + (rows - 1) / topK;
    if (slotCount == 0 || rows > MOE_LORA_SINGLE_ADAPTER_SMALL_WORK_BUDGET ||
        tokens > MOE_LORA_SINGLE_ADAPTER_SMALL_WORK_BUDGET - rows || ubBytes <= UB_RESERVE_BYTES) return false;
    const uint64_t readSlots = slotCount < tokens ? slotCount : tokens;
    const uint64_t required = AlignBlock(rows * indexBytes) + AlignBlock(rows * expertBytes) +
        AlignBlock(readSlots * sizeof(int64_t)) + AlignBlock(rows * sizeof(int64_t)) + MAX_ADAPTERS * sizeof(int32_t);
    return required <= ubBytes - UB_RESERVE_BYTES;
}
bool moe_lora_single_adapter_indices_large_supported(uint64_t rows, uint64_t ubBytes) {
    // Worst-case scalar scatter: two 8B index tiles, a clamped slot tile,
    // enabled cache, slot-zero hint and 32B aligned UB records. Pack is larger.
    constexpr uint64_t scatterBytes = TILE * 16 + AlignBlock((TILE + 1) * sizeof(int64_t)) +
        MAX_ADAPTERS * sizeof(int32_t) + BLOCK_BYTES + TILE * BLOCK_BYTES;
    constexpr uint64_t packBytes = TILE * (64 + 16 + sizeof(int64_t));
    constexpr uint64_t required = scatterBytes > packBytes ? scatterBytes : packBytes;
    return rows >= MOE_LORA_SINGLE_ADAPTER_LARGE_MIN_ROWS && rows <= MOE_LORA_SINGLE_ADAPTER_LARGE_MAX_ROWS &&
        ubBytes >= required + UB_RESERVE_BYTES;
}

void moe_lora_single_adapter_indices_small_impl(void* stream, void* expanded, void* topk, void* slots,
    void* enabled, void* output, uint32_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t adapters, uint64_t numExperts, uint32_t indexBytes, uint32_t expertBytes) {
#define LAUNCH(NAME) NAME<<<1, nullptr, stream>>>((GM_ADDR)expanded, (GM_ADDR)topk, (GM_ADDR)slots, \
    (GM_ADDR)enabled, (GM_ADDR)output, rows, topK, slotCount, adapters, numExperts)
    if (indexBytes == 4 && expertBytes == 4) { LAUNCH(moe_lora_single_small_ii); }
    else if (indexBytes == 4) { LAUNCH(moe_lora_single_small_il); }
    else if (expertBytes == 4) { LAUNCH(moe_lora_single_small_li); }
    else { LAUNCH(moe_lora_single_small_ll); }
#undef LAUNCH
}
void moe_lora_single_adapter_indices_large_impl(void* stream, void* expanded, void* topk, void* slots,
    void* enabled, void* scratch, void* output, uint32_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t adapters, uint64_t numExperts, uint32_t indexBytes, uint32_t expertBytes, uint32_t cores) {
#define LAUNCH(NAME) NAME<<<cores, nullptr, stream>>>((GM_ADDR)expanded, (GM_ADDR)topk, (GM_ADDR)slots, \
    (GM_ADDR)enabled, (GM_ADDR)scratch, rows, topK, slotCount, adapters, numExperts)
    if (indexBytes == 4 && expertBytes == 4) { LAUNCH(moe_lora_single_scatter_ii); }
    else if (indexBytes == 4) { LAUNCH(moe_lora_single_scatter_il); }
    else if (expertBytes == 4) { LAUNCH(moe_lora_single_scatter_li); }
    else { LAUNCH(moe_lora_single_scatter_ll); }
#undef LAUNCH
    moe_lora_single_pack<<<cores, nullptr, stream>>>((GM_ADDR)scratch, (GM_ADDR)output, rows);
}
}  // namespace vllm_ascend
