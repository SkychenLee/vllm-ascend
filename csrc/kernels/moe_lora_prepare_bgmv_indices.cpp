/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"

namespace {

constexpr uint32_t ROUTING_TILE_ROWS = 4096;

class MoeLoraPrepareBgmvIndices {
public:
    __aicore__ inline MoeLoraPrepareBgmvIndices(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void* routedLoraSlots, __gm__ void* groupList,
                                __gm__ void* adapterEnabled, __gm__ void* output,
                                uint32_t numRows, uint32_t numExperts, uint32_t numLoras,
                                float numLorasFloat)
    {
        numRows_ = numRows;
        numExperts_ = numExperts;
        numLoras_ = numLoras;
        numLorasFloat_ = numLorasFloat;
        routedLoraSlotsGm_.SetGlobalBuffer((__gm__ float*)routedLoraSlots, numRows);
        groupListGm_.SetGlobalBuffer((__gm__ int64_t*)groupList, numExperts);
        adapterEnabledGm_.SetGlobalBuffer((__gm__ int32_t*)adapterEnabled, numLoras);
        outputGm_.SetGlobalBuffer((__gm__ int64_t*)output, numRows);

        pipe_->InitBuffer(
            routedLoraSlotsBuf_,
            AlignBytes(ROUTING_TILE_ROWS * sizeof(float)));
        pipe_->InitBuffer(groupListBuf_, AlignBytes(numExperts * sizeof(int64_t)));
        pipe_->InitBuffer(adapterEnabledBuf_, AlignBytes(numLoras * sizeof(int32_t)));
        pipe_->InitBuffer(
            outputBuf_,
            AlignBytes(ROUTING_TILE_ROWS * sizeof(int64_t)));
    }

    __aicore__ inline void Process()
    {
        CopyMetadata();
        AscendC::LocalTensor<int64_t> groupList = groupListBuf_.Get<int64_t>();

        // Init-routing output is expert-major. Assigning one AIV block to each
        // expert lets prefill segments run in parallel. Each block only needs
        // the short prefix of expert counts to locate its disjoint segment,
        // avoiding both a global cumsum/searchsorted chain and atomics.
        uint32_t expert = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (expert >= numExperts_) {
            return;
        }
        uint32_t row = 0;
        for (uint32_t previousExpert = 0;
             previousExpert < expert && row < numRows_;
             ++previousExpert) {
            int64_t previousCount = groupList.GetValue(previousExpert);
            if (previousCount <= 0) {
                continue;
            }
            uint64_t unclampedRow =
                static_cast<uint64_t>(row) + static_cast<uint64_t>(previousCount);
            row = unclampedRow < numRows_ ? static_cast<uint32_t>(unclampedRow) : numRows_;
        }

        uint32_t end = row;
        int64_t count = groupList.GetValue(expert);
        if (count > 0 && row < numRows_) {
            uint64_t unclampedEnd =
                static_cast<uint64_t>(row) + static_cast<uint64_t>(count);
            end = unclampedEnd < numRows_ ? static_cast<uint32_t>(unclampedEnd) : numRows_;
        }
        for (uint32_t offset = row; offset < end;) {
            uint32_t tileRows = end - offset;
            if (tileRows > ROUTING_TILE_ROWS) {
                tileRows = ROUTING_TILE_ROWS;
            }
            ProcessTile(offset, tileRows, expert);
            offset += tileRows;
        }

        // Only the final expert block owns the undefined padded tail.
        if (expert + 1 != numExperts_) {
            return;
        }
        for (uint32_t offset = end; offset < numRows_;) {
            uint32_t tileRows = numRows_ - offset;
            if (tileRows > ROUTING_TILE_ROWS) {
                tileRows = ROUTING_TILE_ROWS;
            }
            ProcessTailTile(offset, tileRows);
            offset += tileRows;
        }
    }

private:
    __aicore__ inline static uint32_t AlignBytes(uint32_t bytes)
    {
        constexpr uint32_t alignment = 32;
        return (bytes + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline void CopyMetadata()
    {
        AscendC::LocalTensor<int64_t> groupList = groupListBuf_.Get<int64_t>();
        AscendC::LocalTensor<int32_t> adapterEnabled = adapterEnabledBuf_.Get<int32_t>();
        AscendC::DataCopyPad(
            groupList,
            groupListGm_,
            {1, static_cast<uint32_t>(numExperts_ * sizeof(int64_t)), 0, 0, 0},
            {true, 0, 0, 0});
        AscendC::DataCopyPad(
            adapterEnabled,
            adapterEnabledGm_,
            {1, static_cast<uint32_t>(numLoras_ * sizeof(int32_t)), 0, 0, 0},
            {true, 0, 0, 0});
        event_t eventIdMte2ToS = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
    }

    __aicore__ inline void ProcessTile(
        uint32_t offset, uint32_t tileRows, uint32_t expert)
    {
        AscendC::LocalTensor<float> routedLoraSlots =
            routedLoraSlotsBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> adapterEnabled =
            adapterEnabledBuf_.Get<int32_t>();
        AscendC::LocalTensor<int64_t> output = outputBuf_.Get<int64_t>();
        AscendC::DataCopyPad(
            routedLoraSlots,
            routedLoraSlotsGm_[offset],
            {1, static_cast<uint32_t>(tileRows * sizeof(float)), 0, 0, 0},
            {true, 0, 0, 0});
        event_t eventIdMte2ToS =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        for (uint32_t localRow = 0; localRow < tileRows; ++localRow) {
            output.SetValue(
                localRow,
                PrepareIndex(
                    routedLoraSlots,
                    adapterEnabled,
                    localRow,
                    expert));
        }
        CopyOutput(offset, tileRows, output);
    }

    __aicore__ inline void ProcessTailTile(uint32_t offset, uint32_t tileRows)
    {
        AscendC::LocalTensor<int64_t> output = outputBuf_.Get<int64_t>();
        for (uint32_t localRow = 0; localRow < tileRows; ++localRow) {
            output.SetValue(localRow, static_cast<int64_t>(-1));
        }
        CopyOutput(offset, tileRows, output);
    }

    __aicore__ inline void CopyOutput(
        uint32_t offset,
        uint32_t tileRows,
        const AscendC::LocalTensor<int64_t>& output)
    {
        event_t eventIdSToMte3 =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::DataCopyPad(
            outputGm_[offset],
            output,
            {1, static_cast<uint32_t>(tileRows * sizeof(int64_t)), 0, 0, 0});
        event_t eventIdMte3ToS =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
    }

    __aicore__ inline int64_t PrepareIndex(
        const AscendC::LocalTensor<float>& routedLoraSlots,
        const AscendC::LocalTensor<int32_t>& adapterEnabled,
        uint32_t row, uint32_t expert)
    {
        float slotValue = routedLoraSlots.GetValue(row);
        // These comparisons also reject NaN/Inf before the float-to-integer
        // conversion. active_expert_range may leave such values in the tail.
        if (!(slotValue >= 0.0f && slotValue < numLorasFloat_)) {
            return -1;
        }
        int64_t slot = static_cast<int64_t>(slotValue);
        if (adapterEnabled.GetValue(slot) == 0) {
            return -1;
        }
        return slot * static_cast<int64_t>(numExperts_) + static_cast<int64_t>(expert);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::GlobalTensor<float> routedLoraSlotsGm_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    AscendC::GlobalTensor<int32_t> adapterEnabledGm_;
    AscendC::GlobalTensor<int64_t> outputGm_;
    AscendC::TBuf<AscendC::TPosition::VECIN> routedLoraSlotsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> groupListBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> adapterEnabledBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outputBuf_;
    uint32_t numRows_;
    uint32_t numExperts_;
    uint32_t numLoras_;
    float numLorasFloat_;
};

}  // namespace

extern "C" __global__ __aicore__ void moe_lora_prepare_bgmv_indices(
    __gm__ void* routedLoraSlots, __gm__ void* groupList, __gm__ void* adapterEnabled,
    __gm__ void* output, uint32_t numRows, uint32_t numExperts, uint32_t numLoras,
    float numLorasFloat)
{
    AscendC::TPipe pipe;
    MoeLoraPrepareBgmvIndices op(&pipe);
    op.Init(routedLoraSlots, groupList, adapterEnabled, output, numRows, numExperts,
            numLoras, numLorasFloat);
    op.Process();
}

namespace vllm_ascend {
extern void moe_lora_prepare_bgmv_indices_impl(
    void* stream, void* routedLoraSlots, void* groupList, void* adapterEnabled,
    void* output, uint32_t numRows, uint32_t numExperts, uint32_t numLoras)
{
    const uint32_t blockDim = numExperts;
    moe_lora_prepare_bgmv_indices<<<blockDim, nullptr, stream>>>(
        routedLoraSlots, groupList, adapterEnabled, output, numRows, numExperts,
        numLoras, static_cast<float>(numLoras));
}
}  // namespace vllm_ascend
