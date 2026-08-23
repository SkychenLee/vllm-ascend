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

#include <cstdint>
#include <limits>

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "../../../ops.h"

namespace vllm_ascend {

namespace {

struct RoutingDeviceResources {
    uint32_t vectorCoreNum;
    uint64_t ubBytes;
};

const RoutingDeviceResources& GetRoutingDeviceResources()
{
    // vLLM workers are process-bound to one NPU. Cache these immutable
    // properties so the latency-sensitive routing op does not query them on
    // every launch.
    static const RoutingDeviceResources resources = [] {
        int32_t deviceId = 0;
        int64_t vectorCoreNum = 0;
        int64_t ubBytes = 0;
        TORCH_CHECK(
            aclrtGetDevice(&deviceId) == ACL_SUCCESS,
            "moe_lora_build_combined_idx: failed to query current device");
        TORCH_CHECK(
            aclrtGetDeviceInfo(
                static_cast<uint32_t>(deviceId),
                ACL_DEV_ATTR_VECTOR_CORE_NUM,
                &vectorCoreNum) == ACL_SUCCESS,
            "moe_lora_build_combined_idx: failed to query Vector Core count");
        TORCH_CHECK(
            aclrtGetDeviceInfo(
                static_cast<uint32_t>(deviceId),
                ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE,
                &ubBytes) == ACL_SUCCESS,
            "moe_lora_build_combined_idx: failed to query UB capacity");
        TORCH_CHECK(
            vectorCoreNum > 0,
            "moe_lora_build_combined_idx: invalid Vector Core count");
        // CANN 9.0 returns zero for ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE on
        // 910B3. Use a conservative cross-product floor in that case; the
        // 910B3 physical UB is much larger, while the kernel stays below this
        // floor by construction.
        constexpr int64_t conservativeUbBytes = 64 * 1024;
        if (ubBytes <= 0) {
            ubBytes = conservativeUbBytes;
        }
        return RoutingDeviceResources{
            static_cast<uint32_t>(vectorCoreNum),
            static_cast<uint64_t>(ubBytes)};
    }();
    return resources;
}

uint32_t AlignUpInt32Block(uint32_t value)
{
    constexpr uint32_t elementsPerBlock = 32 / sizeof(int32_t);
    return (value + elementsPerBlock - 1) /
        elementsPerBlock * elementsPerBlock;
}

}  // namespace

at::Tensor moe_lora_build_combined_idx(
    const at::Tensor& expandedRowIdx,
    const at::Tensor& topkIds,
    const at::Tensor& tokenLoraIndices,
    const at::Tensor& adapterEnabled,
    int64_t numExperts)
{
    TORCH_CHECK(
        expandedRowIdx.scalar_type() == at::kInt,
        "moe_lora_build_combined_idx: expanded_row_idx must be int32");
    TORCH_CHECK(
        topkIds.scalar_type() == at::kInt,
        "moe_lora_build_combined_idx: topk_ids must be int32");
    TORCH_CHECK(
        tokenLoraIndices.scalar_type() == at::kLong,
        "moe_lora_build_combined_idx: token_lora_indices must be int64");
    TORCH_CHECK(
        adapterEnabled.scalar_type() == at::kBool ||
            adapterEnabled.scalar_type() == at::kChar,
        "moe_lora_build_combined_idx: adapter_enabled must be bool or int8");
    TORCH_CHECK(
        expandedRowIdx.dim() == 1,
        "moe_lora_build_combined_idx: expanded_row_idx must be 1D");
    TORCH_CHECK(
        topkIds.dim() == 2 && topkIds.size(1) > 0,
        "moe_lora_build_combined_idx: topk_ids must be [tokens, top_k]");
    TORCH_CHECK(
        tokenLoraIndices.dim() == 1 && adapterEnabled.dim() == 1,
        "moe_lora_build_combined_idx: LoRA metadata must be 1D");
    TORCH_CHECK(
        expandedRowIdx.numel() == topkIds.numel(),
        "moe_lora_build_combined_idx: routing tensor sizes must match");
    TORCH_CHECK(
        tokenLoraIndices.numel() >= topkIds.size(0),
        "moe_lora_build_combined_idx: token LoRA mapping is too short");
    TORCH_CHECK(
        numExperts > 0 &&
            numExperts <= std::numeric_limits<uint32_t>::max(),
        "moe_lora_build_combined_idx: num_experts is out of range");
    TORCH_CHECK(
        adapterEnabled.numel() <=
            std::numeric_limits<int32_t>::max() / numExperts,
        "moe_lora_build_combined_idx: combined index exceeds int32");
    TORCH_CHECK(
        expandedRowIdx.is_contiguous() && topkIds.is_contiguous() &&
            tokenLoraIndices.is_contiguous() && adapterEnabled.is_contiguous(),
        "moe_lora_build_combined_idx: all inputs must be contiguous");
    TORCH_CHECK(
        expandedRowIdx.device() == topkIds.device() &&
            expandedRowIdx.device() == tokenLoraIndices.device() &&
            expandedRowIdx.device() == adapterEnabled.device(),
        "moe_lora_build_combined_idx: all inputs must be on the same device");

    const int64_t numPairs64 = expandedRowIdx.numel();
    TORCH_CHECK(
        numPairs64 <= std::numeric_limits<uint32_t>::max(),
        "moe_lora_build_combined_idx: too many routed pairs");
    at::Tensor output = at::empty(
        {numPairs64}, expandedRowIdx.options().dtype(at::kInt));
    if (numPairs64 == 0) {
        return output;
    }

    const uint32_t numPairs = static_cast<uint32_t>(numPairs64);
    const uint32_t topK = static_cast<uint32_t>(topkIds.size(1));
    const uint32_t numExperts32 = static_cast<uint32_t>(numExperts);
    const RoutingDeviceResources& resources = GetRoutingDeviceResources();
    constexpr uint32_t multiCoreThreshold = 512;
    constexpr uint32_t maxTilePairs = 4096;
    uint32_t tileLength = 0;
    uint32_t outputPairsPerCore = numPairs;
    uint32_t blockDim = 1;
    if (numPairs >= multiCoreThreshold && resources.vectorCoreNum > 1) {
        tileLength = numPairs < maxTilePairs ? numPairs : maxTilePairs;
        uint32_t desiredCoreNum = resources.vectorCoreNum;
        const uint32_t maxUsefulCoreNum = (numPairs + 63) / 64;
        if (desiredCoreNum > maxUsefulCoreNum) {
            desiredCoreNum = maxUsefulCoreNum;
        }
        outputPairsPerCore = AlignUpInt32Block(
            (numPairs + desiredCoreNum - 1) / desiredCoreNum);
        blockDim = (numPairs + outputPairsPerCore - 1) /
            outputPairsPerCore;

        const uint64_t inputBytes =
            2ULL * AlignUpInt32Block(tileLength) * sizeof(int32_t);
        const uint64_t outputBytes =
            static_cast<uint64_t>(outputPairsPerCore) * sizeof(int32_t);
        constexpr uint64_t ubSafetyBytes = 1024;
        if (inputBytes + outputBytes + ubSafetyBytes > resources.ubBytes) {
            tileLength = 0;
            outputPairsPerCore = numPairs;
            blockDim = 1;
        }
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* expandedPtr = expandedRowIdx.data_ptr();
    void* topkPtr = topkIds.data_ptr();
    void* tokenLoraPtr = tokenLoraIndices.data_ptr();
    void* adapterPtr = adapterEnabled.data_ptr();
    void* outputPtr = output.data_ptr();

    at_npu::native::OpCommand cmd;
    cmd.Name("moe_lora_build_combined_idx");
    cmd.SetCustomHandler([
        stream,
        expandedPtr,
        topkPtr,
        tokenLoraPtr,
        adapterPtr,
        outputPtr,
        numPairs,
        topK,
        numExperts32,
        tileLength,
        outputPairsPerCore,
        blockDim
    ]() -> int {
        moe_lora_build_combined_idx_impl(
            stream,
            expandedPtr,
            topkPtr,
            tokenLoraPtr,
            adapterPtr,
            outputPtr,
            numPairs,
            topK,
            numExperts32,
            tileLength,
            outputPairsPerCore,
            blockDim);
        return 0;
    });
    cmd.Run();
    return output;
}

}  // namespace vllm_ascend
