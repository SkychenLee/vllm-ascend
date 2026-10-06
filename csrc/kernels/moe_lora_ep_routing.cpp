/* SPDX-License-Identifier: Apache-2.0 */
#include "kernel_operator.h"
#include "moe_lora_ep_routing.h"

namespace {
template <typename RowIndexType, typename ExpertT, typename MapT>
class EpRouting {
public:
    __aicore__ inline explicit EpRouting(AscendC::TPipe* pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots,
        GM_ADDR expertMap, GM_ADDR expertOut, GM_ADDR slotOut, uint32_t pairs,
        uint32_t capacity, uint32_t topK, uint32_t experts)
    {
        pairs_ = pairs;
        capacity_ = capacity;
        topK_ = topK;
        experts_ = experts;
        tokens_ = pairs / topK;
        expandedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RowIndexType*>(expanded), pairs);
        topkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ExpertT*>(topk), pairs);
        slotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots), tokens_);
        mapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MapT*>(expertMap), experts);
        expertOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(expertOut), capacity);
        slotOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slotOut), capacity);
        pipe_->InitBuffer(expandedBuf_, RoundBytes(pairs * sizeof(RowIndexType)));
        pipe_->InitBuffer(topkBuf_, RoundBytes(pairs * sizeof(ExpertT)));
        pipe_->InitBuffer(slotsBuf_, RoundBytes(tokens_ * sizeof(int64_t)));
        pipe_->InitBuffer(mapBuf_, RoundBytes(experts * sizeof(MapT)));
        pipe_->InitBuffer(expertOutBuf_, RoundBytes(capacity * sizeof(int64_t)));
        pipe_->InitBuffer(slotOutBuf_, RoundBytes(capacity * sizeof(int64_t)));
        pipe_->InitBuffer(seenBuf_, RoundBytes(capacity * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        auto expanded = expandedBuf_.Get<RowIndexType>();
        auto topk = topkBuf_.Get<ExpertT>();
        auto slots = slotsBuf_.Get<int64_t>();
        auto map = mapBuf_.Get<MapT>();
        auto expertOut = expertOutBuf_.Get<int64_t>();
        auto slotOut = slotOutBuf_.Get<int64_t>();
        auto seen = seenBuf_.Get<int32_t>();
        CopyIn(expanded, expandedGm_, pairs_);
        CopyIn(topk, topkGm_, pairs_);
        CopyIn(slots, slotsGm_, tokens_);
        CopyIn(map, mapGm_, experts_);
        // Both 32-bit halves are -1, retaining INT64 sentinels bit-for-bit.
        AscendC::Duplicate(expertOut.template ReinterpretCast<int32_t>(), int32_t{-1}, capacity_ * 2);
        AscendC::Duplicate(slotOut.template ReinterpretCast<int32_t>(), int32_t{-1}, capacity_ * 2);
        AscendC::Duplicate(seen, int32_t{0}, capacity_);
        Sync<AscendC::HardEvent::MTE2_S>();
        Sync<AscendC::HardEvent::V_S>();
        uint32_t source = 0;
        for (uint32_t token = 0; token < tokens_; ++token) {
            const int64_t slot = slots.GetValue(token);
            for (uint32_t k = 0; k < topK_; ++k, ++source) {
                const int64_t destination = static_cast<int64_t>(expanded.GetValue(source));
                // Never abs() a producer sentinel, even INT64_MIN.
                if (destination < 0 || destination >= capacity_) continue;
                seen.SetValue(static_cast<uint32_t>(destination), 1);
                const int64_t global = static_cast<int64_t>(topk.GetValue(source));
                if (global < 0 || global >= experts_) continue;
                const int64_t local = static_cast<int64_t>(map.GetValue(static_cast<uint32_t>(global)));
                if (local < 0) continue;
                expertOut.SetValue(static_cast<uint32_t>(destination), local);
                slotOut.SetValue(static_cast<uint32_t>(destination), slot);
            }
        }
        // Original argsort compares its ordinal against the destination. For
        // unique destinations an internal hole invalidates every later row;
        // reproduce this even for malformed input. Valid producers have a
        // contiguous prefix and only unwritten capacity tails.
        bool prefix = true;
        for (uint32_t row = 0; row < capacity_; ++row) {
            prefix = prefix && seen.GetValue(row) != 0;
            if (!prefix) {
                expertOut.SetValue(row, int64_t{-1});
                slotOut.SetValue(row, int64_t{-1});
            }
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        const AscendC::DataCopyExtParams out{1, capacity_ * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPad(expertOutGm_, expertOut, out);
        AscendC::DataCopyPad(slotOutGm_, slotOut, out);
        Sync<AscendC::HardEvent::MTE3_S>();
    }

private:
    __aicore__ inline uint32_t RoundBytes(uint32_t bytes) { return (bytes + 31) / 32 * 32; }
    template <AscendC::HardEvent Event> __aicore__ inline void Sync()
    {
        const event_t id = static_cast<event_t>(pipe_->FetchEventID(Event));
        AscendC::SetFlag<Event>(id);
        AscendC::WaitFlag<Event>(id);
    }
    template <typename T> __aicore__ inline void CopyIn(AscendC::LocalTensor<T>& dst,
        AscendC::GlobalTensor<T>& src, uint32_t count)
    {
        const AscendC::DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, params, pad);
    }
    AscendC::TPipe* pipe_;
    uint32_t pairs_, capacity_, topK_, experts_, tokens_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expandedBuf_, topkBuf_, slotsBuf_, mapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expertOutBuf_, slotOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> seenBuf_;
    AscendC::GlobalTensor<RowIndexType> expandedGm_;
    AscendC::GlobalTensor<ExpertT> topkGm_;
    AscendC::GlobalTensor<MapT> mapGm_;
    AscendC::GlobalTensor<int64_t> slotsGm_, expertOutGm_, slotOutGm_;
};
}  // namespace

#define DEFINE_EP_ROUTING(name, index_t, expert_t, map_t) \
extern "C" __global__ __aicore__ void name(GM_ADDR expanded, GM_ADDR topk, GM_ADDR slots, \
    GM_ADDR expertMap, GM_ADDR expertOut, GM_ADDR slotOut, uint32_t pairs, \
    uint32_t capacity, uint32_t topK, uint32_t experts) \
{ \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); \
    AscendC::TPipe pipe; \
    EpRouting<index_t, expert_t, map_t> op(&pipe); \
    op.Init(expanded, topk, slots, expertMap, expertOut, slotOut, pairs, capacity, topK, experts); \
    op.Process(); \
}
DEFINE_EP_ROUTING(ep_routing_iii, int32_t, int32_t, int32_t)
DEFINE_EP_ROUTING(ep_routing_iil, int32_t, int32_t, int64_t)
DEFINE_EP_ROUTING(ep_routing_ili, int32_t, int64_t, int32_t)
DEFINE_EP_ROUTING(ep_routing_ill, int32_t, int64_t, int64_t)
DEFINE_EP_ROUTING(ep_routing_lii, int64_t, int32_t, int32_t)
DEFINE_EP_ROUTING(ep_routing_lil, int64_t, int32_t, int64_t)
DEFINE_EP_ROUTING(ep_routing_lli, int64_t, int64_t, int32_t)
DEFINE_EP_ROUTING(ep_routing_lll, int64_t, int64_t, int64_t)
#undef DEFINE_EP_ROUTING

namespace vllm_ascend {
void moe_lora_ep_routing_impl(void* stream, void* expanded, void* topk,
    void* slots, void* expertMap, void* expertOut, void* slotOut,
    uint32_t pairs, uint32_t capacity, uint32_t topK, uint32_t experts,
    uint32_t expandedBytes, uint32_t topkBytes, uint32_t mapBytes)
{
    if (capacity == 0) return;
#define LAUNCH(name) name<<<1, nullptr, stream>>>(expanded, topk, slots, expertMap, expertOut, slotOut, pairs, capacity, topK, experts)
    if (expandedBytes == 4) {
        if (topkBytes == 4) {
            if (mapBytes == 4) { LAUNCH(ep_routing_iii); } else { LAUNCH(ep_routing_iil); }
        } else {
            if (mapBytes == 4) { LAUNCH(ep_routing_ili); } else { LAUNCH(ep_routing_ill); }
        }
    } else if (topkBytes == 4) {
        if (mapBytes == 4) { LAUNCH(ep_routing_lii); } else { LAUNCH(ep_routing_lil); }
    } else {
        if (mapBytes == 4) { LAUNCH(ep_routing_lli); } else { LAUNCH(ep_routing_lll); }
    }
#undef LAUNCH
}
}  // namespace vllm_ascend
