// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "moe_lora_combined_indices.h"

namespace {
template <AscendC::HardEvent event>
__aicore__ inline void Sync() {
    AscendC::SetFlag<event>(EVENT_ID0);
    AscendC::WaitFlag<event>(EVENT_ID0);
}
template <typename T>
__aicore__ inline void Load(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint32_t n) {
    AscendC::DataCopyPad(dst, src, AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0},
                       AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
}
}

template <typename ExpertT>
class CombinedIndices {
public:
    __aicore__ inline void Run(GM_ADDR experts, GM_ADDR slots, GM_ADDR enabled, GM_ADDR output,
                               uint64_t rows, uint64_t numExperts, uint32_t adapters) {
        AscendC::GlobalTensor<ExpertT> egm;
        AscendC::GlobalTensor<int64_t> sgm;
        AscendC::GlobalTensor<int32_t> mgm;
        AscendC::GlobalTensor<int64_t> ogm;
        egm.SetGlobalBuffer(reinterpret_cast<__gm__ ExpertT*>(experts));
        sgm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
        mgm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(enabled));
        ogm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(output));
        pipe.InitBuffer(eb, MOE_LORA_COMBINED_TILE_ROWS * sizeof(ExpertT));
        pipe.InitBuffer(sb, MOE_LORA_COMBINED_TILE_ROWS * sizeof(int64_t));
        pipe.InitBuffer(ob, MOE_LORA_COMBINED_TILE_ROWS * sizeof(int64_t));
        pipe.InitBuffer(mb, MOE_LORA_COMBINED_MAX_ADAPTERS * sizeof(int32_t));
        auto e = eb.Get<ExpertT>();
        auto s = sb.Get<int64_t>();
        auto o = ob.Get<int64_t>();
        auto mask = mb.Get<int32_t>();
        Load(mask, mgm, adapters);
        Sync<AscendC::HardEvent::MTE2_S>();
        // Each core owns whole 64-byte output lines. Only the final core
        // writes a partial line, with exact DataCopyPad byte counts.
        uint64_t coreRows = (rows + AscendC::GetBlockNum() * 8 - 1) / (AscendC::GetBlockNum() * 8) * 8;
        uint64_t begin = AscendC::GetBlockIdx() * coreRows;
        uint64_t end = begin + coreRows < rows ? begin + coreRows : rows;
        for (uint64_t start = begin; start < end; start += MOE_LORA_COMBINED_TILE_ROWS) {
            uint32_t n = static_cast<uint32_t>(end - start < MOE_LORA_COMBINED_TILE_ROWS
                                             ? end - start : MOE_LORA_COMBINED_TILE_ROWS);
            Sync<AscendC::HardEvent::S_MTE2>();
            Load(e, egm[start], n);
            Load(s, sgm[start], n);
            Sync<AscendC::HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < n; ++i) {
                int64_t slot = s.GetValue(i);
                int64_t result = -1;
                // Guard before indexing UB; malformed positive slots are disabled.
                if (slot >= 0 && static_cast<uint64_t>(slot) < adapters && mask.GetValue(slot) != 0) {
                    // Unsigned arithmetic defines two's-complement overflow just
                    // as tensor INT64 arithmetic does; no floating-point payload cast.
                    uint64_t bits = static_cast<uint64_t>(slot) * numExperts +
                                    static_cast<uint64_t>(static_cast<int64_t>(e.GetValue(i)));
                    result = static_cast<int64_t>(bits);
                }
                o.SetValue(i, result);
            }
            Sync<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(ogm[start], o,
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(int64_t)), 0, 0, 0});
            Sync<AscendC::HardEvent::MTE3_S>();
        }
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> eb, sb, mb, ob;
};

#define COMBINED_KERNEL(NAME, T) \
extern "C" __global__ __aicore__ void NAME(GM_ADDR e, GM_ADDR s, GM_ADDR m, GM_ADDR o, \
    uint64_t rows, uint64_t experts, uint32_t adapters) { \
    CombinedIndices<T> op; op.Run(e, s, m, o, rows, experts, adapters); \
}
COMBINED_KERNEL(moe_lora_combined_i32, int32_t)
COMBINED_KERNEL(moe_lora_combined_i64, int64_t)

void moe_lora_combined_indices_impl(void* stream, void* experts, void* slots,
    void* enabled, void* output, uint64_t rows, uint64_t num_experts,
    uint32_t adapters, uint32_t expert_bytes, uint32_t cores) {
    if (expert_bytes == 4) {
        moe_lora_combined_i32<<<cores, nullptr, stream>>>((GM_ADDR)experts, (GM_ADDR)slots,
            (GM_ADDR)enabled, (GM_ADDR)output, rows, num_experts, adapters);
    } else {
        moe_lora_combined_i64<<<cores, nullptr, stream>>>((GM_ADDR)experts, (GM_ADDR)slots,
            (GM_ADDR)enabled, (GM_ADDR)output, rows, num_experts, adapters);
    }
}
