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

constexpr uint32_t ROUTING_TILE_ROWS = 2048;
constexpr uint32_t EXPERTS_PER_BLOCK = 4;

class MoeLoraPrepareCompositeGmmRouting {
public:
    __aicore__ inline MoeLoraPrepareCompositeGmmRouting(AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(
        __gm__ void* routedLoraSlots, __gm__ void* groupList,
        __gm__ void* adapterEnabled, __gm__ void* groupIds,
        __gm__ void* compositeGroupList, __gm__ void* enabled,
        uint32_t numRows, uint32_t numExperts, uint32_t numLoras,
        uint32_t numTailBlocks, float numLorasFloat)
    {
        numRows_ = numRows;
        numExperts_ = numExperts;
        numLoras_ = numLoras;
        numTailBlocks_ = numTailBlocks;
        numLorasFloat_ = numLorasFloat;
        blockIdx_ = AscendC::GetBlockIdx();

        routedLoraSlotsGm_.SetGlobalBuffer((__gm__ float*)routedLoraSlots, numRows);
        groupListGm_.SetGlobalBuffer((__gm__ int64_t*)groupList, numExperts);
        adapterEnabledGm_.SetGlobalBuffer((__gm__ int32_t*)adapterEnabled, numLoras);
        groupIdsGm_.SetGlobalBuffer((__gm__ int32_t*)groupIds, numRows);
        compositeGroupListGm_.SetGlobalBuffer(
            (__gm__ int64_t*)compositeGroupList, numExperts * numLoras);
        enabledGm_.SetGlobalBuffer((__gm__ uint8_t*)enabled, numRows);

        pipe_->InitBuffer(
            routedLoraSlotsBuf_, AlignBytes(ROUTING_TILE_ROWS * sizeof(float)));
        pipe_->InitBuffer(groupListBuf_, AlignBytes(numExperts * sizeof(int64_t)));
        pipe_->InitBuffer(adapterEnabledBuf_, AlignBytes(numLoras * sizeof(int32_t)));
        pipe_->InitBuffer(
            groupIdsBuf_, AlignBytes(ROUTING_TILE_ROWS * sizeof(int32_t)));
        pipe_->InitBuffer(enabledBuf_, AlignBytes(ROUTING_TILE_ROWS * sizeof(uint8_t)));
        pipe_->InitBuffer(loraCountsBuf_, AlignBytes(numLoras * sizeof(int64_t)));
        pipe_->InitBuffer(
            blockCountsBuf_,
            AlignBytes(numLoras * EXPERTS_PER_BLOCK * sizeof(int64_t)));
    }

    __aicore__ inline void Process()
    {
        CopyMetadata();
        uint32_t numExpertBlocks =
            (numExperts_ + EXPERTS_PER_BLOCK - 1) / EXPERTS_PER_BLOCK;
        if (blockIdx_ < numExpertBlocks) {
            uint32_t firstExpert = blockIdx_ * EXPERTS_PER_BLOCK;
            uint32_t lastExpert = firstExpert + EXPERTS_PER_BLOCK;
            if (lastExpert > numExperts_) {
                lastExpert = numExperts_;
            }
            AscendC::LocalTensor<int64_t> blockCounts = blockCountsBuf_.Get<int64_t>();
            for (uint32_t expert = firstExpert; expert < lastExpert; ++expert) {
                ProcessExpert(expert, firstExpert, blockCounts);
            }
            CopyBlockCountsOut(firstExpert, blockCounts);
        } else {
            ProcessTail(blockIdx_ - numExpertBlocks);
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
        event_t eventIdMte2ToS =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
    }

    __aicore__ inline uint32_t GetExpertStart(uint32_t expert)
    {
        AscendC::LocalTensor<int64_t> groupList = groupListBuf_.Get<int64_t>();
        uint64_t start = 0;
        for (uint32_t index = 0; index < expert; ++index) {
            int64_t count = groupList.GetValue(index);
            if (count > 0) {
                start += static_cast<uint64_t>(count);
            }
        }
        return start < numRows_ ? static_cast<uint32_t>(start) : numRows_;
    }

    __aicore__ inline uint32_t GetValidRows()
    {
        AscendC::LocalTensor<int64_t> groupList = groupListBuf_.Get<int64_t>();
        uint64_t validRows = 0;
        for (uint32_t expert = 0; expert < numExperts_; ++expert) {
            int64_t count = groupList.GetValue(expert);
            if (count > 0) {
                validRows += static_cast<uint64_t>(count);
            }
        }
        return validRows < numRows_ ? static_cast<uint32_t>(validRows) : numRows_;
    }

    __aicore__ inline void ProcessExpert(
        uint32_t expert,
        uint32_t firstExpert,
        const AscendC::LocalTensor<int64_t>& blockCounts)
    {
        AscendC::LocalTensor<int64_t> groupList = groupListBuf_.Get<int64_t>();
        AscendC::LocalTensor<int64_t> loraCounts = loraCountsBuf_.Get<int64_t>();
        for (uint32_t slot = 0; slot < numLoras_; ++slot) {
            loraCounts.SetValue(slot, 0);
        }

        uint32_t start = GetExpertStart(expert);
        int64_t count = groupList.GetValue(expert);
        uint64_t unclampedEnd = count > 0
            ? static_cast<uint64_t>(start) + static_cast<uint64_t>(count)
            : static_cast<uint64_t>(start);
        uint32_t end = unclampedEnd < numRows_
            ? static_cast<uint32_t>(unclampedEnd)
            : numRows_;

        for (uint32_t offset = start; offset < end; offset += ROUTING_TILE_ROWS) {
            uint32_t tileRows = end - offset;
            if (tileRows > ROUTING_TILE_ROWS) {
                tileRows = ROUTING_TILE_ROWS;
            }
            ProcessExpertTile(expert, offset, tileRows, loraCounts);
        }

        uint32_t expertOffset = expert - firstExpert;
        for (uint32_t slot = 0; slot < numLoras_; ++slot) {
            blockCounts.SetValue(
                slot * EXPERTS_PER_BLOCK + expertOffset,
                loraCounts.GetValue(slot));
        }
    }

    __aicore__ inline void CopyBlockCountsOut(
        uint32_t firstExpert,
        const AscendC::LocalTensor<int64_t>& blockCounts)
    {
        // Four int64 counts are one aligned 32-byte DMA block. Writing the
        // complete cache line once avoids false sharing and scalar GM cache
        // writeback loss while retaining the slot-major GMM weight layout.
        event_t eventIdSToMte3 =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        for (uint32_t slot = 0; slot < numLoras_; ++slot) {
            AscendC::DataCopy(
                compositeGroupListGm_[slot * numExperts_ + firstExpert],
                blockCounts[slot * EXPERTS_PER_BLOCK],
                EXPERTS_PER_BLOCK);
        }
        event_t eventIdMte3ToS =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
    }

    __aicore__ inline void ProcessExpertTile(
        uint32_t expert, uint32_t offset, uint32_t tileRows,
        const AscendC::LocalTensor<int64_t>& loraCounts)
    {
        AscendC::LocalTensor<float> routedLoraSlots = routedLoraSlotsBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> adapterEnabled = adapterEnabledBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> groupIds = groupIdsBuf_.Get<int32_t>();
        AscendC::LocalTensor<uint8_t> enabled = enabledBuf_.Get<uint8_t>();

        AscendC::DataCopyPad(
            routedLoraSlots,
            routedLoraSlotsGm_[offset],
            {1, static_cast<uint32_t>(tileRows * sizeof(float)), 0, 0, 0},
            {true, 0, 0, 0});
        event_t eventIdMte2ToS =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);

        int32_t sentinelGroup = static_cast<int32_t>(numLoras_ * numExperts_);
        for (uint32_t row = 0; row < tileRows; ++row) {
            float slotValue = routedLoraSlots.GetValue(row);
            bool rowEnabled = slotValue >= 0.0f && slotValue < numLorasFloat_;
            int32_t slot = rowEnabled ? static_cast<int32_t>(slotValue) : 0;
            rowEnabled = rowEnabled && adapterEnabled.GetValue(slot) != 0;
            if (rowEnabled) {
                groupIds.SetValue(
                    row, slot * static_cast<int32_t>(numExperts_) +
                             static_cast<int32_t>(expert));
                enabled.SetValue(row, static_cast<uint8_t>(1));
                loraCounts.SetValue(slot, loraCounts.GetValue(slot) + 1);
            } else {
                groupIds.SetValue(row, sentinelGroup);
                enabled.SetValue(row, static_cast<uint8_t>(0));
            }
        }
        CopyRoutingTileOut(offset, tileRows, groupIds, enabled);
    }

    __aicore__ inline void ProcessTail(uint32_t tailBlock)
    {
        if (tailBlock >= numTailBlocks_) {
            return;
        }
        uint32_t validRows = GetValidRows();
        uint32_t tailRows = numRows_ - validRows;
        uint32_t rowsPerBlock =
            (tailRows + numTailBlocks_ - 1) / numTailBlocks_;
        uint64_t unclampedStart =
            static_cast<uint64_t>(validRows) +
            static_cast<uint64_t>(tailBlock) * rowsPerBlock;
        uint32_t start = unclampedStart < numRows_
            ? static_cast<uint32_t>(unclampedStart)
            : numRows_;
        uint64_t unclampedEnd =
            static_cast<uint64_t>(start) + rowsPerBlock;
        uint32_t end = unclampedEnd < numRows_
            ? static_cast<uint32_t>(unclampedEnd)
            : numRows_;

        AscendC::LocalTensor<int32_t> groupIds = groupIdsBuf_.Get<int32_t>();
        AscendC::LocalTensor<uint8_t> enabled = enabledBuf_.Get<uint8_t>();
        int32_t sentinelGroup = static_cast<int32_t>(numLoras_ * numExperts_);
        for (uint32_t offset = start; offset < end; offset += ROUTING_TILE_ROWS) {
            uint32_t tileRows = end - offset;
            if (tileRows > ROUTING_TILE_ROWS) {
                tileRows = ROUTING_TILE_ROWS;
            }
            for (uint32_t row = 0; row < tileRows; ++row) {
                groupIds.SetValue(row, sentinelGroup);
                enabled.SetValue(row, static_cast<uint8_t>(0));
            }
            CopyRoutingTileOut(offset, tileRows, groupIds, enabled);
        }
    }

    __aicore__ inline void CopyRoutingTileOut(
        uint32_t offset, uint32_t tileRows,
        const AscendC::LocalTensor<int32_t>& groupIds,
        const AscendC::LocalTensor<uint8_t>& enabled)
    {
        event_t eventIdSToMte3 =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::DataCopyPad(
            groupIdsGm_[offset],
            groupIds,
            {1, static_cast<uint32_t>(tileRows * sizeof(int32_t)), 0, 0, 0});
        AscendC::DataCopyPad(
            enabledGm_[offset],
            enabled,
            {1, static_cast<uint32_t>(tileRows * sizeof(uint8_t)), 0, 0, 0});
        event_t eventIdMte3ToS =
            static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::GlobalTensor<float> routedLoraSlotsGm_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    AscendC::GlobalTensor<int32_t> adapterEnabledGm_;
    AscendC::GlobalTensor<int32_t> groupIdsGm_;
    AscendC::GlobalTensor<int64_t> compositeGroupListGm_;
    AscendC::GlobalTensor<uint8_t> enabledGm_;
    AscendC::TBuf<AscendC::TPosition::VECIN> routedLoraSlotsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> groupListBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> adapterEnabledBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> groupIdsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> enabledBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> loraCountsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> blockCountsBuf_;
    uint32_t numRows_;
    uint32_t numExperts_;
    uint32_t numLoras_;
    uint32_t numTailBlocks_;
    uint32_t blockIdx_;
    float numLorasFloat_;
};

}  // namespace

extern "C" __global__ __aicore__ void moe_lora_prepare_composite_gmm_routing(
    __gm__ void* routedLoraSlots, __gm__ void* groupList,
    __gm__ void* adapterEnabled, __gm__ void* groupIds,
    __gm__ void* compositeGroupList, __gm__ void* enabled,
    uint32_t numRows, uint32_t numExperts, uint32_t numLoras,
    uint32_t numTailBlocks, float numLorasFloat)
{
    AscendC::TPipe pipe;
    MoeLoraPrepareCompositeGmmRouting op(&pipe);
    op.Init(routedLoraSlots, groupList, adapterEnabled, groupIds,
            compositeGroupList, enabled, numRows, numExperts, numLoras,
            numTailBlocks, numLorasFloat);
    op.Process();
}

namespace vllm_ascend {
extern void moe_lora_prepare_composite_gmm_routing_impl(
    void* stream, void* routedLoraSlots, void* groupList,
    void* adapterEnabled, void* groupIds, void* compositeGroupList,
    void* enabled, uint32_t numRows, uint32_t numExperts,
    uint32_t numLoras)
{
    constexpr uint32_t numTailBlocks = 32;
    constexpr uint32_t expertsPerBlock = 4;
    uint32_t numExpertBlocks =
        (numExperts + expertsPerBlock - 1) / expertsPerBlock;
    uint32_t blockDim = numExpertBlocks + numTailBlocks;
    moe_lora_prepare_composite_gmm_routing<<<blockDim, nullptr, stream>>>(
        routedLoraSlots, groupList, adapterEnabled, groupIds,
        compositeGroupList, enabled, numRows, numExperts, numLoras,
        numTailBlocks, static_cast<float>(numLoras));
}
}  // namespace vllm_ascend
