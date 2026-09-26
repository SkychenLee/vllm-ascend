/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel_operator.h"
#include "moe_lora_recover_small.h"

namespace {
constexpr uint64_t BLOCK_BYTES = 32;
// Reserve capacity for future pipeline/control buffers; this is a resource
// margin, not a whitelist of model dimensions or ranks.
constexpr uint64_t UB_RESERVE_BYTES = 4096;
// Original code sorts integer magnitudes after conversion to float32. Retain
// the exact-key domain, even though the UB constraint is much tighter in use.
constexpr uint64_t MAX_EXACT_FLOAT32_PERMUTATION_ROWS = (uint64_t{1} << 24) + 1;

constexpr uint64_t AlignBlock(uint64_t bytes)
{
    return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}
} // namespace

template <typename IndexType, typename ExpertT, bool Filtered = false>
class MoeLoraRecoverSmall {
public:
    __aicore__ inline explicit MoeLoraRecoverSmall(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots,
                                GM_ADDR expertOut, GM_ADDR slotOut, uint32_t rows,
                                uint64_t topK, uint64_t slotCount,
                                int64_t expertStart = 0, int64_t numLocalExperts = 0)
    {
        rows_ = rows;
        topK_ = topK;
        slotCount_ = slotCount;
        expertStart_ = expertStart;
        numLocalExperts_ = numLocalExperts;
        const uint64_t tokens = 1 + (static_cast<uint64_t>(rows) - 1) / topK;
        readSlots_ = static_cast<uint32_t>(slotCount < tokens ? slotCount : tokens);
        expandedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IndexType*>(expanded), rows);
        topkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ExpertT*>(topk), rows);
        slotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots), readSlots_);
        expertOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(expertOut), rows);
        slotOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slotOut), rows);
        pipe_->InitBuffer(expandedBuf_, RoundBytes(static_cast<uint64_t>(rows) * sizeof(IndexType)));
        pipe_->InitBuffer(topkBuf_, RoundBytes(static_cast<uint64_t>(rows) * sizeof(ExpertT)));
        pipe_->InitBuffer(slotsBuf_, RoundBytes(static_cast<uint64_t>(readSlots_) * sizeof(int64_t)));
        pipe_->InitBuffer(expertOutBuf_, RoundBytes(static_cast<uint64_t>(rows) * sizeof(int64_t)));
        pipe_->InitBuffer(slotOutBuf_, RoundBytes(static_cast<uint64_t>(rows) * sizeof(int64_t)));
    }

    __aicore__ inline void Process()
    {
        auto expanded = expandedBuf_.Get<IndexType>();
        auto topk = topkBuf_.Get<ExpertT>();
        auto slots = slotsBuf_.Get<int64_t>();
        auto expertOut = expertOutBuf_.Get<int64_t>();
        auto slotOut = slotOutBuf_.Get<int64_t>();
        CopyIn(expanded, expandedGm_, rows_);
        CopyIn(topk, topkGm_, rows_);
        CopyIn(slots, slotsGm_, readSlots_);

        // Values outside the caller's permutation contract must never index UB
        // out of bounds. Zeroing also avoids exposing uninitialized output when
        // such invalid data leaves a hole. It does not define duplicate-index
        // argsort semantics or turn malformed input into supported input.
        const int32_t invalid = Filtered ? -1 : 0;
        AscendC::Duplicate(expertOut.template ReinterpretCast<int32_t>(), invalid, rows_ * 2);
        AscendC::Duplicate(slotOut.template ReinterpretCast<int32_t>(), invalid, rows_ * 2);
        Sync<AscendC::HardEvent::MTE2_S>();
        Sync<AscendC::HardEvent::V_S>();

        uint32_t p = 0;
        uint64_t token = 0;
        while (p < rows_) {
            const uint64_t slot = token < slotCount_ ? token : slotCount_ - 1;
            const int64_t slotValue = slots.GetValue(static_cast<uint32_t>(slot));
            const uint32_t remaining = rows_ - p;
            const uint32_t group = static_cast<uint32_t>(topK_ < remaining ? topK_ : remaining);
            // Consecutive original rows share p/topK. This integer grouping
            // preserves floor/clamp exactly without a division for every row.
            for (uint32_t lane = 0; lane < group; ++lane) {
                const uint32_t source = p + lane;
                const int64_t signedDestination = static_cast<int64_t>(expanded.GetValue(source));
                if (Filtered && signedDestination < 0) {
                    continue;
                }
                // Unsigned subtraction is defined even for INT64_MIN; skip an
                // out-of-contract magnitude before any destination access.
                const uint64_t destination = signedDestination < 0
                    ? uint64_t{0} - static_cast<uint64_t>(signedDestination)
                    : static_cast<uint64_t>(signedDestination);
                if (destination >= rows_) {
                    continue;
                }
                int64_t expert = static_cast<int64_t>(topk.GetValue(source));
                if (Filtered) {
                    if (expert < expertStart_ || expert - expertStart_ >= numLocalExperts_) {
                        continue;
                    }
                    expert -= expertStart_;
                }
                expertOut.SetValue(static_cast<uint32_t>(destination), expert);
                slotOut.SetValue(static_cast<uint32_t>(destination), slotValue);
            }
            p += group;
            ++token;
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        const AscendC::DataCopyExtParams copyOut{1, rows_ * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPad(expertOutGm_, expertOut, copyOut);
        AscendC::DataCopyPad(slotOutGm_, slotOut, copyOut);
        // These TBuf objects are shared for the whole kernel. Wait before their
        // lifetime ends; there are no inter-core dependencies or shared writes.
        Sync<AscendC::HardEvent::MTE3_S>();
    }

private:
    __aicore__ inline uint32_t RoundBytes(uint64_t bytes)
    {
        return static_cast<uint32_t>((bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES);
    }

    template <AscendC::HardEvent Event>
    __aicore__ inline void Sync()
    {
        const event_t id = static_cast<event_t>(pipe_->FetchEventID(Event));
        AscendC::SetFlag<Event>(id);
        AscendC::WaitFlag<Event>(id);
    }

    template <typename T>
    __aicore__ inline void CopyIn(AscendC::LocalTensor<T>& local, AscendC::GlobalTensor<T>& global,
                                  uint32_t count)
    {
        const AscendC::DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        AscendC::DataCopyPad(local, global, params, padding);
    }

    AscendC::TPipe* pipe_;
    uint32_t rows_;
    uint32_t readSlots_;
    uint64_t topK_;
    uint64_t slotCount_;
    int64_t expertStart_;
    int64_t numLocalExperts_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expandedBuf_, topkBuf_, slotsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expertOutBuf_, slotOutBuf_;
    AscendC::GlobalTensor<IndexType> expandedGm_;
    AscendC::GlobalTensor<ExpertT> topkGm_;
    AscendC::GlobalTensor<int64_t> slotsGm_, expertOutGm_, slotOutGm_;
};

#define DEFINE_RECOVER_KERNEL(name, index_type, expert_type) \
extern "C" __global__ __aicore__ void name(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots, \
    GM_ADDR expertOut, GM_ADDR slotOut, uint32_t rows, uint64_t topK, uint64_t slotCount) \
{ \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
    if (rows == 0 || topK == 0 || slotCount == 0) { return; } \
    AscendC::TPipe pipe; \
    MoeLoraRecoverSmall<index_type, expert_type> op(&pipe); \
    op.Init(expanded, topk, slots, expertOut, slotOut, rows, topK, slotCount); \
    op.Process(); \
}

DEFINE_RECOVER_KERNEL(moe_lora_recover_small_i32_i32, int32_t, int32_t)
DEFINE_RECOVER_KERNEL(moe_lora_recover_small_i32_i64, int32_t, int64_t)
DEFINE_RECOVER_KERNEL(moe_lora_recover_small_i64_i32, int64_t, int32_t)
DEFINE_RECOVER_KERNEL(moe_lora_recover_small_i64_i64, int64_t, int64_t)
#undef DEFINE_RECOVER_KERNEL

#define DEFINE_RECOVER_EP_KERNEL(name, index_type, expert_type) \
extern "C" __global__ __aicore__ void name(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots, \
    GM_ADDR expertOut, GM_ADDR slotOut, uint32_t rows, uint64_t topK, uint64_t slotCount, \
    int64_t expertStart, int64_t numLocalExperts) \
{ \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
    if (rows == 0 || topK == 0 || slotCount == 0) { return; } \
    AscendC::TPipe pipe; \
    MoeLoraRecoverSmall<index_type, expert_type, true> op(&pipe); \
    op.Init(expanded, topk, slots, expertOut, slotOut, rows, topK, slotCount, expertStart, numLocalExperts); \
    op.Process(); \
}

DEFINE_RECOVER_EP_KERNEL(moe_lora_recover_ep_small_i32_i32, int32_t, int32_t)
DEFINE_RECOVER_EP_KERNEL(moe_lora_recover_ep_small_i32_i64, int32_t, int64_t)
DEFINE_RECOVER_EP_KERNEL(moe_lora_recover_ep_small_i64_i32, int64_t, int32_t)
DEFINE_RECOVER_EP_KERNEL(moe_lora_recover_ep_small_i64_i64, int64_t, int64_t)
#undef DEFINE_RECOVER_EP_KERNEL

namespace vllm_ascend {

bool moe_lora_recover_small_supported(uint64_t rows, uint64_t topK, uint64_t slotCount,
                                      uint32_t expandedElementBytes, uint32_t expertElementBytes,
                                      uint64_t ubBytes)
{
    if ((expandedElementBytes != 4 && expandedElementBytes != 8) ||
        (expertElementBytes != 4 && expertElementBytes != 8) || topK == 0) {
        return false;
    }
    if (rows == 0) {
        return true;
    }
    if (slotCount == 0 || rows > MAX_EXACT_FLOAT32_PERMUTATION_ROWS || ubBytes <= UB_RESERVE_BYTES) {
        return false;
    }
    const uint64_t tokens = 1 + (rows - 1) / topK;
    const uint64_t readSlots = slotCount < tokens ? slotCount : tokens;
    const uint64_t required = AlignBlock(rows * expandedElementBytes) + AlignBlock(rows * expertElementBytes) +
        AlignBlock(readSlots * sizeof(int64_t)) + 2 * AlignBlock(rows * sizeof(int64_t));
    return required <= ubBytes - UB_RESERVE_BYTES;
}

void moe_lora_recover_small_impl(void* stream, void* expanded, void* topk, void* slots,
                                void* expertOut, void* slotOut, uint64_t rows, uint64_t topK,
                                uint64_t slotCount, uint32_t expandedElementBytes,
                                uint32_t expertElementBytes, uint64_t ubBytes)
{
    if (rows == 0 || !moe_lora_recover_small_supported(rows, topK, slotCount,
        expandedElementBytes, expertElementBytes, ubBytes)) {
        return;
    }
    const uint32_t kernelRows = static_cast<uint32_t>(rows);
    if (expandedElementBytes == sizeof(int32_t)) {
        if (expertElementBytes == sizeof(int32_t)) {
            moe_lora_recover_small_i32_i32<<<1, nullptr, stream>>>(expanded, topk, slots, expertOut, slotOut,
                kernelRows, topK, slotCount);
        } else {
            moe_lora_recover_small_i32_i64<<<1, nullptr, stream>>>(expanded, topk, slots, expertOut, slotOut,
                kernelRows, topK, slotCount);
        }
    } else if (expertElementBytes == sizeof(int32_t)) {
        moe_lora_recover_small_i64_i32<<<1, nullptr, stream>>>(expanded, topk, slots, expertOut, slotOut,
            kernelRows, topK, slotCount);
    } else {
        moe_lora_recover_small_i64_i64<<<1, nullptr, stream>>>(expanded, topk, slots, expertOut, slotOut,
            kernelRows, topK, slotCount);
    }
}

void moe_lora_recover_ep_small_impl(void* stream, void* expanded, void* topk, void* slots,
                                   void* expertOut, void* slotOut, uint64_t rows, uint64_t topK,
                                   uint64_t slotCount, uint32_t expandedElementBytes,
                                   uint32_t expertElementBytes, uint64_t ubBytes,
                                   int64_t expertStart, int64_t numLocalExperts)
{
    if (rows == 0 || numLocalExperts <= 0 || !moe_lora_recover_small_supported(rows, topK, slotCount,
        expandedElementBytes, expertElementBytes, ubBytes)) {
        return;
    }
    const uint32_t kernelRows = static_cast<uint32_t>(rows);
    if (expandedElementBytes == sizeof(int32_t)) {
        if (expertElementBytes == sizeof(int32_t)) {
            moe_lora_recover_ep_small_i32_i32<<<1, nullptr, stream>>>(expanded, topk, slots,
                expertOut, slotOut, kernelRows, topK, slotCount, expertStart, numLocalExperts);
        } else {
            moe_lora_recover_ep_small_i32_i64<<<1, nullptr, stream>>>(expanded, topk, slots,
                expertOut, slotOut, kernelRows, topK, slotCount, expertStart, numLocalExperts);
        }
    } else if (expertElementBytes == sizeof(int32_t)) {
        moe_lora_recover_ep_small_i64_i32<<<1, nullptr, stream>>>(expanded, topk, slots,
            expertOut, slotOut, kernelRows, topK, slotCount, expertStart, numLocalExperts);
    } else {
        moe_lora_recover_ep_small_i64_i64<<<1, nullptr, stream>>>(expanded, topk, slots,
            expertOut, slotOut, kernelRows, topK, slotCount, expertStart, numLocalExperts);
    }
}

} // namespace vllm_ascend
