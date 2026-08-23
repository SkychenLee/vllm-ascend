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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "../../../ops.h"

namespace vllm_ascend {

namespace {

constexpr uint32_t kRank = 16;
constexpr uint32_t kGroupRows = 8;
constexpr uint32_t kOutputTileElements = 512;
constexpr uint32_t kWeightTileElements = kOutputTileElements * kRank;
constexpr uint32_t kMaxHiddenDim = 2048;
constexpr uint32_t kFp32ReuseMinRows = 2048;

struct BgmvFusedDeviceResources {
    uint32_t vectorCoreNum;
    uint64_t ubBytes;
};

const BgmvFusedDeviceResources& GetBgmvFusedDeviceResources()
{
    static const BgmvFusedDeviceResources resources = [] {
        int32_t deviceId = 0;
        int64_t vectorCoreNum = 0;
        int64_t ubBytes = 0;
        TORCH_CHECK(
            aclrtGetDevice(&deviceId) == ACL_SUCCESS,
            "moe_lora_bgmv_fused: failed to query current device");
        TORCH_CHECK(
            aclrtGetDeviceInfo(
                static_cast<uint32_t>(deviceId),
                ACL_DEV_ATTR_VECTOR_CORE_NUM,
                &vectorCoreNum) == ACL_SUCCESS,
            "moe_lora_bgmv_fused: failed to query Vector Core count");
        TORCH_CHECK(
            aclrtGetDeviceInfo(
                static_cast<uint32_t>(deviceId),
                ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE,
                &ubBytes) == ACL_SUCCESS,
            "moe_lora_bgmv_fused: failed to query UB capacity");
        TORCH_CHECK(
            vectorCoreNum > 0,
            "moe_lora_bgmv_fused: invalid Vector Core count");
        return BgmvFusedDeviceResources{
            static_cast<uint32_t>(vectorCoreNum),
            ubBytes > 0 ? static_cast<uint64_t>(ubBytes) : 0};
    }();
    return resources;
}

uint64_t AlignUp32Bytes(uint64_t bytes)
{
    constexpr uint64_t blockBytes = 32;
    return (bytes + blockBytes - 1) / blockBytes * blockBytes;
}

uint64_t CalculateUbBytes(
    uint32_t inputHiddenDim,
    uint32_t indexElementBytes,
    uint32_t dataElementBytes,
    bool reuseFp32Weight)
{
    const uint32_t weightBufferElements = std::max(
        inputHiddenDim, kWeightTileElements);
    uint64_t bytes = 0;
    bytes += AlignUp32Bytes(kGroupRows * indexElementBytes);
    bytes += AlignUp32Bytes(
        static_cast<uint64_t>(kGroupRows) * inputHiddenDim *
        dataElementBytes);
    bytes += AlignUp32Bytes(
        static_cast<uint64_t>(weightBufferElements) * dataElementBytes);
    bytes += 2 * AlignUp32Bytes(
        static_cast<uint64_t>(kOutputTileElements) * dataElementBytes);
    bytes += AlignUp32Bytes(
        static_cast<uint64_t>(kGroupRows) * inputHiddenDim *
        sizeof(float));
    if (reuseFp32Weight) {
        bytes += AlignUp32Bytes(
            static_cast<uint64_t>(weightBufferElements) * sizeof(float));
    }
    bytes += AlignUp32Bytes(
        static_cast<uint64_t>(weightBufferElements) * sizeof(float));
    bytes += AlignUp32Bytes(kGroupRows * kRank * sizeof(float));
    bytes += AlignUp32Bytes(256);
    bytes += 2 * AlignUp32Bytes(
        kOutputTileElements * sizeof(float));
    return bytes;
}

AscendType GetBgmvFusedAscendType(at::ScalarType scalarType)
{
    return scalarType == at::kBFloat16 ?
        AscendType::BF16 : AscendType::FP16;
}

}  // namespace

at::Tensor moe_lora_bgmv_fused(
    const at::Tensor& x,
    const at::Tensor& loraA,
    const at::Tensor& loraB,
    const at::Tensor& indices,
    at::Tensor& y,
    int64_t sliceOffset,
    int64_t sliceSize,
    double scale)
{
    const at::ScalarType scalarType = x.scalar_type();
    TORCH_CHECK(
        scalarType == at::kHalf || scalarType == at::kBFloat16,
        "moe_lora_bgmv_fused: x must be float16 or bfloat16");
    TORCH_CHECK(
        loraA.scalar_type() == scalarType &&
            loraB.scalar_type() == scalarType &&
            y.scalar_type() == scalarType,
        "moe_lora_bgmv_fused: x, lora_a, lora_b and y must have the same dtype");
    TORCH_CHECK(
        indices.scalar_type() == at::kInt ||
            indices.scalar_type() == at::kLong,
        "moe_lora_bgmv_fused: indices must be int32 or int64");
    TORCH_CHECK(
        x.dim() == 2 && y.dim() == 2,
        "moe_lora_bgmv_fused: x and y must be 2D");
    TORCH_CHECK(
        loraA.dim() == 3 && loraB.dim() == 3,
        "moe_lora_bgmv_fused: lora_a and lora_b must be 3D");
    TORCH_CHECK(
        indices.dim() == 1,
        "moe_lora_bgmv_fused: indices must be 1D");
    TORCH_CHECK(
        x.size(0) == y.size(0) && x.size(0) == indices.size(0),
        "moe_lora_bgmv_fused: row counts of x, y and indices must match");
    TORCH_CHECK(
        loraA.size(0) == loraB.size(0),
        "moe_lora_bgmv_fused: lora_a and lora_b weight counts must match");
    TORCH_CHECK(
        loraA.size(1) == kRank && loraB.size(2) == kRank,
        "moe_lora_bgmv_fused: only rank 16 is supported");
    TORCH_CHECK(
        loraA.size(2) == x.size(1),
        "moe_lora_bgmv_fused: lora_a input dimension must match x");
    TORCH_CHECK(
        sliceSize == loraB.size(1),
        "moe_lora_bgmv_fused: slice_size must match lora_b output dimension");
    TORCH_CHECK(
        sliceOffset >= 0 && sliceSize > 0 &&
            sliceOffset <= y.size(1) - sliceSize,
        "moe_lora_bgmv_fused: output slice is out of range");
    TORCH_CHECK(
        x.size(1) > 0 && x.size(1) <= kMaxHiddenDim,
        "moe_lora_bgmv_fused: input hidden dimension must be in [1, 2048]");
    TORCH_CHECK(
        sliceSize <= kMaxHiddenDim,
        "moe_lora_bgmv_fused: output hidden dimension must be <= 2048");
    TORCH_CHECK(
        std::isfinite(scale),
        "moe_lora_bgmv_fused: scale must be finite");
    TORCH_CHECK(
        x.is_contiguous() && loraA.is_contiguous() &&
            loraB.is_contiguous() && indices.is_contiguous() &&
            y.is_contiguous(),
        "moe_lora_bgmv_fused: all tensors must be contiguous");
    TORCH_CHECK(
        x.device() == loraA.device() && x.device() == loraB.device() &&
            x.device() == indices.device() && x.device() == y.device(),
        "moe_lora_bgmv_fused: all tensors must be on the same device");

    const int64_t numRows64 = x.size(0);
    if (numRows64 == 0) {
        return y;
    }
    TORCH_CHECK(
        numRows64 <= std::numeric_limits<uint32_t>::max() &&
            x.size(1) <= std::numeric_limits<uint32_t>::max() &&
            sliceSize <= std::numeric_limits<uint32_t>::max() &&
            y.size(1) <= std::numeric_limits<uint32_t>::max() &&
            sliceOffset <= std::numeric_limits<uint32_t>::max(),
        "moe_lora_bgmv_fused: shape exceeds uint32 launch limits");

    const BgmvFusedDeviceResources& resources =
        GetBgmvFusedDeviceResources();
    const uint32_t indexElementBytes =
        indices.scalar_type() == at::kInt ? sizeof(int32_t) : sizeof(int64_t);
    const uint64_t requiredUbBytes = CalculateUbBytes(
        static_cast<uint32_t>(x.size(1)),
        indexElementBytes,
        static_cast<uint32_t>(x.element_size()),
        numRows64 >= kFp32ReuseMinRows ||
            x.size(1) < kMaxHiddenDim);
    if (resources.ubBytes != 0) {
        TORCH_CHECK(
            requiredUbBytes <= resources.ubBytes,
            "moe_lora_bgmv_fused: requires ", requiredUbBytes,
            " UB bytes, but device reports ", resources.ubBytes);
    }

    const uint32_t numRows = static_cast<uint32_t>(numRows64);
    uint32_t desiredCoreNum = (numRows + kGroupRows - 1) / kGroupRows;
    if (desiredCoreNum > resources.vectorCoreNum) {
        desiredCoreNum = resources.vectorCoreNum;
    }
    uint32_t rowsPerCore =
        (numRows + desiredCoreNum - 1) / desiredCoreNum;
    rowsPerCore =
        (rowsPerCore + kGroupRows - 1) / kGroupRows * kGroupRows;

    void* xPtr = x.data_ptr();
    void* aPtr = loraA.data_ptr();
    void* bPtr = loraB.data_ptr();
    void* indicesPtr = indices.data_ptr();
    void* yPtr = y.data_ptr();
    const uint32_t inputHiddenDim = static_cast<uint32_t>(x.size(1));
    const uint32_t outputHiddenDim = static_cast<uint32_t>(sliceSize);
    const uint32_t outputFullDim = static_cast<uint32_t>(y.size(1));
    const uint32_t sliceOffset32 = static_cast<uint32_t>(sliceOffset);
    const float scaleFloat = static_cast<float>(scale);
    const bool indicesIsInt32 = indices.scalar_type() == at::kInt;
    const AscendType ascendType = GetBgmvFusedAscendType(scalarType);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    at_npu::native::OpCommand cmd;
    cmd.Name("moe_lora_bgmv_fused");
    cmd.SetCustomHandler([
        ascendType,
        stream,
        xPtr,
        aPtr,
        bPtr,
        indicesPtr,
        yPtr,
        numRows,
        inputHiddenDim,
        outputHiddenDim,
        outputFullDim,
        sliceOffset32,
        rowsPerCore,
        scaleFloat,
        indicesIsInt32
    ]() -> int {
        moe_lora_bgmv_fused_impl(
            ascendType,
            stream,
            xPtr,
            aPtr,
            bPtr,
            indicesPtr,
            yPtr,
            numRows,
            inputHiddenDim,
            outputHiddenDim,
            outputFullDim,
            sliceOffset32,
            rowsPerCore,
            scaleFloat,
            indicesIsInt32);
        return 0;
    });
    cmd.Run();
    return y;
}

}  // namespace vllm_ascend
