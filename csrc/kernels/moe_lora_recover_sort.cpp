// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "moe_lora_recover_sort.h"
#include <cmath>
#include <limits>

namespace {
constexpr uint32_t SORT_GROUP = 32;
constexpr uint32_t MAX_SORT_REPEATS = 255;
constexpr uint32_t GATHER_SCRATCH_ARRAYS = 5;
constexpr uint32_t UB_RESERVE_BYTES = 4096;
constexpr uint32_t SORT_BYTES_PER_ROW = 24;
constexpr uint64_t AlignRows(uint64_t rows)
{
    return (rows + SORT_GROUP - 1) / SORT_GROUP * SORT_GROUP;
}
} // namespace

template <typename IndexType, typename ExpertType>
class MoeLoraRecoverSort {
public:
    __aicore__ inline explicit MoeLoraRecoverSort(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots,
        GM_ADDR expertOut, GM_ADDR slotOut, uint32_t rows, uint32_t topK,
        uint32_t readSlots, float reciprocal)
    {
        rows_ = rows;
        aligned_ = (rows + SORT_GROUP - 1) / SORT_GROUP * SORT_GROUP;
        tile_ = aligned_ / GATHER_SCRATCH_ARRAYS / SORT_GROUP * SORT_GROUP;
        readSlots_ = readSlots;
        reciprocal_ = reciprocal;
        expandedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IndexType*>(expanded), rows);
        expertGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ExpertType*>(topk), rows);
        slotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots), readSlots);
        expertOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(expertOut), 2 * rows);
        slotOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(slotOut), 2 * rows);
        pipe_->InitBuffer(bankA_, 8 * aligned_);
        pipe_->InitBuffer(bankB_, 8 * aligned_);
        pipe_->InitBuffer(bankC_, 8 * aligned_);
    }

    __aicore__ inline void Process()
    {
        using namespace AscendC;
        auto a = bankA_.Get<float>();
        auto b = bankB_.Get<float>();
        auto c = bankC_.Get<float>();
        auto keys = a;
        auto inverse = a[aligned_].ReinterpretCast<uint32_t>();
        // expanded is a signed permutation with magnitude < rows <= 8160.
        // Thus its signed low 32-bit word equals the original int64 value.
        // Only routing keys are narrowed; expert/slot payloads never are.
        CopyIn(b.ReinterpretCast<IndexType>(), expandedGm_, rows_);
        Sync<HardEvent::MTE2_V>();
        if constexpr (sizeof(IndexType) == 8) {
            ArithProgression<int32_t>(inverse.ReinterpretCast<int32_t>(), 0, 8, aligned_);
            PipeBarrier<PIPE_V>();
            Gather(a.ReinterpretCast<int32_t>(), b.ReinterpretCast<int32_t>(), inverse, 0, rows_);
        } else {
            DataCopy(a, b, aligned_);
        }
        PipeBarrier<PIPE_V>();
        Cast(keys, a.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND, rows_);
        PipeBarrier<PIPE_V>();
        Abs(keys, keys, rows_);
        PipeBarrier<PIPE_V>();
        Muls(keys, keys, -1.0f, rows_);
        PipeBarrier<PIPE_V>();
        // Scalar stores only initialize at most 31 padding keys.
        Sync<HardEvent::V_S>();
        for (uint32_t i = rows_; i < aligned_; ++i) {
            keys.SetValue(i, -static_cast<float>(static_cast<int32_t>(aligned_)));
        }
        Sync<HardEvent::S_V>();
        ArithProgression<int32_t>(inverse.ReinterpretCast<int32_t>(), 0, 1, aligned_);
        PipeBarrier<PIPE_V>();
        Concat(keys, keys, c, aligned_ / SORT_GROUP);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(b, keys, inverse, c, aligned_ / SORT_GROUP);
        PipeBarrier<PIPE_V>();
        Extract(keys, inverse, b, aligned_ / SORT_GROUP);
        PipeBarrier<PIPE_V>();

        // A: inverse in its upper half, token lookup in its lower half.
        // B: one payload at a time. C is no longer needed after sorting.
        InitWordOffsets();
        Sync<HardEvent::V_MTE2>();
        CopyIn(b.ReinterpretCast<ExpertType>(), expertGm_, rows_);
        Sync<HardEvent::MTE2_V>();
        if constexpr (sizeof(ExpertType) == 4) {
            // Store low and sign-extension words as separate source planes.
            ShiftRight(b[aligned_].ReinterpretCast<int32_t>(), b.ReinterpretCast<int32_t>(),
                int32_t{31}, rows_);
            PipeBarrier<PIPE_V>();
        }
        WriteGathered(b.ReinterpretCast<uint32_t>(), inverse, expertOutGm_, sizeof(ExpertType) == 8);

        // Exact floor(p/k) on the bounded sort domain: the host rounds 1/k
        // upwards. For p < 8160, FP32 multiply error is < 1/k, so it cannot
        // cross the next integer; an exact integer quotient cannot round
        // below itself. This is not an unchecked reciprocal approximation.
        Cast(keys, inverse.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND, rows_);
        PipeBarrier<PIPE_V>();
        Muls(keys, keys, reciprocal_, rows_);
        PipeBarrier<PIPE_V>();
        Mins(keys, keys, static_cast<float>(static_cast<int32_t>(readSlots_ - 1)), rows_);
        PipeBarrier<PIPE_V>();
        auto token = keys.ReinterpretCast<int32_t>();
        Cast(token, keys, RoundMode::CAST_FLOOR, rows_);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_MTE2>();
        CopyIn(b.ReinterpretCast<int64_t>(), slotsGm_, readSlots_);
        Sync<HardEvent::MTE2_V>();
        WriteGathered(b.ReinterpretCast<uint32_t>(), token.ReinterpretCast<uint32_t>(), slotOutGm_, true);
    }

private:
    __aicore__ inline void InitWordOffsets()
    {
        using namespace AscendC;
        auto s = bankC_.Get<int32_t>();
        auto sequence = s;
        auto pairs = s[2 * tile_];
        auto lanes = s[4 * tile_];
        auto offsets = s[6 * tile_];
        ArithProgression<int32_t>(sequence, 0, 1, 2 * tile_);
        PipeBarrier<PIPE_V>();
        ShiftRight(pairs, sequence, int32_t{1}, 2 * tile_);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, pairs, int32_t{2}, 2 * tile_);
        ShiftLeft(pairs, pairs, int32_t{1}, 2 * tile_);
        PipeBarrier<PIPE_V>();
        Sub(lanes, sequence, pairs, 2 * tile_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteGathered(AscendC::LocalTensor<uint32_t> source,
        AscendC::LocalTensor<uint32_t> indices, AscendC::GlobalTensor<uint32_t> destination,
        bool interleaved)
    {
        using namespace AscendC;
        auto s = bankC_.Get<int32_t>();
        auto byteOffsets = s;
        auto laneBytes = s[2 * tile_];
        auto lanes = s[4 * tile_];
        auto pairOffsets = s[6 * tile_].ReinterpretCast<uint32_t>();
        auto out = s[8 * tile_].ReinterpretCast<uint32_t>();
        Muls(laneBytes, lanes, static_cast<int32_t>(interleaved ? 4 : aligned_ * 4), 2 * tile_);
        PipeBarrier<PIPE_V>();
        for (uint32_t start = 0; start < rows_; start += tile_) {
            const uint32_t count = rows_ - start < tile_ ? rows_ - start : tile_;
            Gather(byteOffsets.ReinterpretCast<uint32_t>(), indices[start], pairOffsets, 0, 2 * count);
            PipeBarrier<PIPE_V>();
            ShiftLeft(byteOffsets, byteOffsets, static_cast<int32_t>(interleaved ? 3 : 2), 2 * count);
            PipeBarrier<PIPE_V>();
            Add(byteOffsets, byteOffsets, laneBytes, 2 * count);
            PipeBarrier<PIPE_V>();
            Gather(out, source, byteOffsets.ReinterpretCast<uint32_t>(), 0, 2 * count);
            Sync<HardEvent::V_MTE3>();
            const DataCopyExtParams params{1, count * 8, 0, 0, 0};
            DataCopyPad(destination[2 * start], out, params);
            Sync<HardEvent::MTE3_V>();
        }
    }

    template <typename T>
    __aicore__ inline void CopyIn(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint32_t count)
    {
        const AscendC::DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, params, padding);
    }

    template <AscendC::HardEvent Event>
    __aicore__ inline void Sync()
    {
        const auto id = pipe_->FetchEventID(Event);
        AscendC::SetFlag<Event>(id);
        AscendC::WaitFlag<Event>(id);
    }

    AscendC::TPipe* pipe_;
    uint32_t rows_, aligned_, readSlots_, tile_;
    float reciprocal_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bankA_, bankB_, bankC_;
    AscendC::GlobalTensor<IndexType> expandedGm_;
    AscendC::GlobalTensor<ExpertType> expertGm_;
    AscendC::GlobalTensor<int64_t> slotsGm_;
    AscendC::GlobalTensor<uint32_t> expertOutGm_, slotOutGm_;
};

#define DEFINE_SORT_KERNEL(name, index_type, expert_type) \
extern "C" __global__ __aicore__ void name(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots, \
    GM_ADDR expertOut, GM_ADDR slotOut, uint32_t rows, uint32_t topK, uint32_t readSlots, float reciprocal) \
{ \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
    AscendC::TPipe pipe; \
    MoeLoraRecoverSort<index_type, expert_type> op(&pipe); \
    op.Init(expanded, topk, slots, expertOut, slotOut, rows, topK, readSlots, reciprocal); \
    op.Process(); \
}
DEFINE_SORT_KERNEL(moe_lora_recover_sort_i32_i32, int32_t, int32_t)
DEFINE_SORT_KERNEL(moe_lora_recover_sort_i32_i64, int32_t, int64_t)
DEFINE_SORT_KERNEL(moe_lora_recover_sort_i64_i32, int64_t, int32_t)
DEFINE_SORT_KERNEL(moe_lora_recover_sort_i64_i64, int64_t, int64_t)
#undef DEFINE_SORT_KERNEL

namespace vllm_ascend {
bool moe_lora_recover_sort_supported(uint64_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t indexBytes, uint32_t expertBytes, uint64_t ubBytes)
{
    if (rows < SORT_GROUP * GATHER_SCRATCH_ARRAYS || !topK || topK > rows || rows % topK || !slotCount ||
        (indexBytes != 4 && indexBytes != 8) || (expertBytes != 4 && expertBytes != 8) ||
        rows > SORT_GROUP * MAX_SORT_REPEATS) {
        return false;
    }
    return AlignRows(rows) * SORT_BYTES_PER_ROW + UB_RESERVE_BYTES <= ubBytes;
}

void moe_lora_recover_sort_impl(void* stream, void* expanded, void* topk, void* slots,
    void* expertOut, void* slotOut, uint64_t rows, uint64_t topK, uint64_t slotCount,
    uint32_t indexBytes, uint32_t expertBytes, uint64_t ubBytes)
{
    if (!moe_lora_recover_sort_supported(rows, topK, slotCount, indexBytes, expertBytes, ubBytes)) {
        return;
    }
    const uint32_t n = static_cast<uint32_t>(rows);
    const uint32_t k = static_cast<uint32_t>(topK);
    const uint32_t readSlots = static_cast<uint32_t>(slotCount < rows / topK ? slotCount : rows / topK);
    const float reciprocal = std::nextafter(static_cast<float>(1.0 / static_cast<double>(topK)),
        std::numeric_limits<float>::infinity());
    if (indexBytes == 4 && expertBytes == 4) {
        moe_lora_recover_sort_i32_i32<<<1, nullptr, stream>>>(expanded, topk, slots, expertOut, slotOut, n, k, readSlots, reciprocal);
    } else if (indexBytes == 4) {
        moe_lora_recover_sort_i32_i64<<<1, nullptr, stream>>>(expanded, topk, slots, expertOut, slotOut, n, k, readSlots, reciprocal);
    } else if (expertBytes == 4) {
        moe_lora_recover_sort_i64_i32<<<1, nullptr, stream>>>(expanded, topk, slots, expertOut, slotOut, n, k, readSlots, reciprocal);
    } else {
        moe_lora_recover_sort_i64_i64<<<1, nullptr, stream>>>(expanded, topk, slots, expertOut, slotOut, n, k, readSlots, reciprocal);
    }
}
} // namespace vllm_ascend
