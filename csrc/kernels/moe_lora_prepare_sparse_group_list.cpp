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

class MoeLoraPrepareSparseGroupList {
public:
    __aicore__ inline MoeLoraPrepareSparseGroupList(AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(
        __gm__ void* groupList, __gm__ void* sparseGroupList,
        uint32_t numExperts)
    {
        numExperts_ = numExperts;
        groupListGm_.SetGlobalBuffer((__gm__ int64_t*)groupList, numExperts);
        sparseGroupListGm_.SetGlobalBuffer(
            (__gm__ int64_t*)sparseGroupList, numExperts * 2);
        pipe_->InitBuffer(
            groupListBuf_, AlignBytes(numExperts * sizeof(int64_t)));
        pipe_->InitBuffer(
            sparseGroupListBuf_,
            AlignBytes(numExperts * 2 * sizeof(int64_t)));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<int64_t> groupList =
            groupListBuf_.Get<int64_t>();
        AscendC::LocalTensor<int64_t> sparseGroupList =
            sparseGroupListBuf_.Get<int64_t>();
        AscendC::DataCopyPad(
            groupList,
            groupListGm_,
            {1, static_cast<uint32_t>(numExperts_ * sizeof(int64_t)),
             0, 0, 0},
            {true, 0, 0, 0});
        event_t eventIdMte2ToS = static_cast<event_t>(
            pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);

        // GroupedMatmulV5 type-2 requires every non-empty group to precede
        // empty groups. Preserve ascending expert order inside each partition
        // so the already expert-major routed rows need no further permutation.
        uint32_t outputIndex = 0;
        for (uint32_t expert = 0; expert < numExperts_; ++expert) {
            int64_t count = groupList.GetValue(expert);
            if (count > 0) {
                WritePair(sparseGroupList, outputIndex++, expert, count);
            }
        }
        for (uint32_t expert = 0; expert < numExperts_; ++expert) {
            int64_t count = groupList.GetValue(expert);
            if (count <= 0) {
                WritePair(sparseGroupList, outputIndex++, expert, 0);
            }
        }

        event_t eventIdSToMte3 = static_cast<event_t>(
            pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIdSToMte3);
        AscendC::DataCopyPad(
            sparseGroupListGm_,
            sparseGroupList,
            {1, static_cast<uint32_t>(
                    numExperts_ * 2 * sizeof(int64_t)),
             0, 0, 0});
    }

private:
    __aicore__ inline static uint32_t AlignBytes(uint32_t bytes)
    {
        constexpr uint32_t alignment = 32;
        return (bytes + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline static void WritePair(
        const AscendC::LocalTensor<int64_t>& output,
        uint32_t outputIndex, uint32_t expert, int64_t count)
    {
        output.SetValue(outputIndex * 2, static_cast<int64_t>(expert));
        output.SetValue(outputIndex * 2 + 1, count);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    AscendC::GlobalTensor<int64_t> sparseGroupListGm_;
    AscendC::TBuf<AscendC::TPosition::VECIN> groupListBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> sparseGroupListBuf_;
    uint32_t numExperts_;
};

}  // namespace

extern "C" __global__ __aicore__ void moe_lora_prepare_sparse_group_list(
    __gm__ void* groupList, __gm__ void* sparseGroupList,
    uint32_t numExperts)
{
    AscendC::TPipe pipe;
    MoeLoraPrepareSparseGroupList op(&pipe);
    op.Init(groupList, sparseGroupList, numExperts);
    op.Process();
}

namespace vllm_ascend {
extern void moe_lora_prepare_sparse_group_list_impl(
    void* stream, void* groupList, void* sparseGroupList,
    uint32_t numExperts)
{
    constexpr uint32_t blockDim = 1;
    moe_lora_prepare_sparse_group_list<<<blockDim, nullptr, stream>>>(
        groupList, sparseGroupList, numExperts);
}
}  // namespace vllm_ascend
