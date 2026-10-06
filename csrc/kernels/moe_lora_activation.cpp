// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "moe_lora_activation.h"

using namespace AscendC;
namespace {
constexpr uint32_t WIDTH = vllm_ascend::MOE_LORA_ACTIVATION_WIDTH;
constexpr uint32_t ROW_UNIT = 1;

template <HardEvent event>
__aicore__ inline void Sync() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

template <typename T>
class ActivationQuant {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR activated,
        GM_ADDR quantized, GM_ADDR scales, uint32_t rows, float limit) {
        xgm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        fgm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(activated));
        qgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quantized));
        sgm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
        pipe.InitBuffer(inputBuf, 2 * WIDTH * sizeof(T));
        pipe.InitBuffer(outputBuf, WIDTH * sizeof(T));
        pipe.InitBuffer(gateBuf, WIDTH * sizeof(float));
        pipe.InitBuffer(upBuf, WIDTH * sizeof(float));
        pipe.InitBuffer(workBuf, WIDTH * sizeof(float));
        pipe.InitBuffer(quantBuf, WIDTH);
        pipe.InitBuffer(halfBuf, WIDTH * sizeof(half));
        pipe.InitBuffer(scalarBuf, 128);
        auto lowGate = inputBuf.Get<T>();
        auto lowUp = inputBuf.Get<T>()[WIDTH];
        auto lowOutput = outputBuf.Get<T>();
        auto gate = gateBuf.Get<float>();
        auto up = upBuf.Get<float>();
        auto work = workBuf.Get<float>();
        auto quant = quantBuf.Get<int8_t>();
        auto halfQuant = halfBuf.Get<half>();
        auto reduction = scalarBuf.Get<float>();
        auto rowScales = scalarBuf.Get<float>()[8];
        auto quantConstant = scalarBuf.Get<float>()[16];
        auto dequantScale = scalarBuf.Get<float>()[24];
        Duplicate(quantConstant, 127.0f, 8);
        PipeBarrier<PIPE_V>();
        for (uint32_t start = GetBlockIdx() * ROW_UNIT;
             start < rows; start += GetBlockNum() * ROW_UNIT) {
            uint32_t count = rows - start < ROW_UNIT ? rows - start : ROW_UNIT;
            for (uint32_t j = 0; j < count; ++j) {
                uint32_t row = start + j;
                Sync<HardEvent::V_MTE2>();
                DataCopy(lowGate, xgm[static_cast<uint64_t>(row) * 2 * WIDTH], 2 * WIDTH);
                Sync<HardEvent::MTE2_V>();
                Cast(gate, lowGate, RoundMode::CAST_NONE, WIDTH);
                Cast(up, lowUp, RoundMode::CAST_NONE, WIDTH);
                PipeBarrier<PIPE_V>();
                if (limit > 0) {
                    Mins(gate, gate, limit, WIDTH);
                    Mins(up, up, limit, WIDTH);
                    PipeBarrier<PIPE_V>();
                    Maxs(up, up, -limit, WIDTH);
                    PipeBarrier<PIPE_V>();
                    // torch.clamp stores to the original model dtype before
                    // torch.cat/npu_swiglu. Preserve that rounding boundary.
                    Cast(lowGate, gate, RoundMode::CAST_RINT, WIDTH);
                    Cast(lowUp, up, RoundMode::CAST_RINT, WIDTH);
                    PipeBarrier<PIPE_V>();
                    Cast(gate, lowGate, RoundMode::CAST_NONE, WIDTH);
                    Cast(up, lowUp, RoundMode::CAST_NONE, WIDTH);
                    PipeBarrier<PIPE_V>();
                }
                Muls(work, gate, -1.0f, WIDTH);
                PipeBarrier<PIPE_V>();
                Exp(work, work, WIDTH);
                PipeBarrier<PIPE_V>();
                Adds(work, work, 1.0f, WIDTH);
                PipeBarrier<PIPE_V>();
                Div(gate, gate, work, WIDTH);
                PipeBarrier<PIPE_V>();
                Mul(gate, gate, up, WIDTH);
                PipeBarrier<PIPE_V>();
                // npu_swiglu returns BF16/FP16. W2 LoRA consumes this exact
                // floating tensor; dynamic quant also reads this rounded value.
                Cast(lowOutput, gate, RoundMode::CAST_RINT, WIDTH);
                PipeBarrier<PIPE_V>();
                Sync<HardEvent::V_MTE3>();
                DataCopy(fgm[static_cast<uint64_t>(row) * WIDTH], lowOutput, WIDTH);
                Sync<HardEvent::MTE3_V>();
                Cast(gate, lowOutput, RoundMode::CAST_NONE, WIDTH);
                PipeBarrier<PIPE_V>();
                Abs(work, gate, WIDTH);
                PipeBarrier<PIPE_V>();
                ReduceMax(reduction, work, work, WIDTH, false);
                Sync<HardEvent::V_S>();
                float maxAbs = reduction.GetValue(0);
                // Match DynamicQuant's vector division, which rounds
                // differently from scalar CCE division. Its output scale
                // uses maxAbs * float(1/127), not scalar maxAbs / 127.
                Muls(dequantScale, reduction, 1.0f / 127.0f, 1);
                Div(reduction, quantConstant, reduction, 1);
                Sync<HardEvent::V_S>();
                float quantScale = reduction.GetValue(0);
                rowScales.SetValue(j, dequantScale.GetValue(0));
                if (maxAbs > 0) {
                    Muls(gate, gate, quantScale, WIDTH);
                    PipeBarrier<PIPE_V>();
                }
                // Round in FP32 first; FP16 -> INT8 alone double-rounds values
                // near half integers. Reuse work after max reduction.
                auto integral = workBuf.Get<int32_t>();
                Cast(integral, gate, RoundMode::CAST_RINT, WIDTH);
                PipeBarrier<PIPE_V>();
                Cast(gate, integral, RoundMode::CAST_NONE, WIDTH);
                PipeBarrier<PIPE_V>();
                Cast(halfQuant, gate, RoundMode::CAST_NONE, WIDTH);
                PipeBarrier<PIPE_V>();
                Cast(quant, halfQuant, RoundMode::CAST_RINT, WIDTH);
                PipeBarrier<PIPE_V>();
                Sync<HardEvent::V_MTE3>();
                DataCopy(qgm[static_cast<uint64_t>(row) * WIDTH], quant, WIDTH);
                Sync<HardEvent::MTE3_V>();
            }
            // Write exactly one FP32 scale, without padding adjacent rows.
            // This is the same disjoint-byte DMA operation used by CANN's
            // single-row DynamicQuant output; UB source remains 32B aligned.
            Sync<HardEvent::S_MTE3>();
            DataCopyPad(sgm[start], rowScales,
                DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
            Sync<HardEvent::MTE3_S>();
        }
    }
private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuf, outputBuf, gateBuf, upBuf,
        workBuf, quantBuf, halfBuf, scalarBuf;
    GlobalTensor<T> xgm, fgm;
    GlobalTensor<int8_t> qgm;
    GlobalTensor<float> sgm;
};
}  // namespace

#define ACTIVATION_KERNEL(NAME, T) \
extern "C" __global__ __aicore__ void NAME(GM_ADDR x, GM_ADDR f, GM_ADDR q, GM_ADDR s, \
    uint32_t rows, float limit) { \
    ActivationQuant<T> op; op.Run(x, f, q, s, rows, limit); \
}
ACTIVATION_KERNEL(moe_lora_activation_half, half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ACTIVATION_KERNEL(moe_lora_activation_bfloat16, bfloat16_t)
#endif

namespace vllm_ascend {
void moe_lora_activation_impl(AscendType type, void* stream, void* input,
    void* activated, void* quantized, void* scales, uint32_t rows,
    uint32_t cores, float limit) {
    if (type == AscendType::FP16) {
        moe_lora_activation_half<<<cores, nullptr, stream>>>((GM_ADDR)input,
            (GM_ADDR)activated, (GM_ADDR)quantized, (GM_ADDR)scales, rows, limit);
    } else {
        moe_lora_activation_bfloat16<<<cores, nullptr, stream>>>((GM_ADDR)input,
            (GM_ADDR)activated, (GM_ADDR)quantized, (GM_ADDR)scales, rows, limit);
    }
}
}  // namespace vllm_ascend
