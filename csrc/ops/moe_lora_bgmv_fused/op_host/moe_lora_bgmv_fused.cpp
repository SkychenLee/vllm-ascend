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

constexpr uint32_t kFastRank = 16;
constexpr uint32_t kDefaultGroupRows = 8;
constexpr uint32_t kWideInputGroupRows = 4;
constexpr uint32_t kOutputTileElements = 512;
constexpr uint32_t kWeightTileElements =
    kOutputTileElements * kFastRank;
constexpr uint32_t kGenericTileElements = 8192;
constexpr uint32_t kGenericReduceTmpBytes = 512;
constexpr uint32_t kWideInputThreshold = 2048;
constexpr uint32_t kFastMaxHiddenDim = 4096;
constexpr uint32_t kMaxHiddenDim = 16384;
constexpr uint32_t kFp32ReuseMinRows = 2048;
constexpr uint32_t kBalancedCoreMaxRows = 1024;

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
            vectorCoreNum > 0 &&
                static_cast<uint64_t>(vectorCoreNum) <=
                    std::numeric_limits<uint32_t>::max(),
            "moe_lora_bgmv_fused: invalid Vector Core count");
        return BgmvFusedDeviceResources{
            static_cast<uint32_t>(vectorCoreNum),
            ubBytes > 0 ? static_cast<uint64_t>(ubBytes) : 0};
    }();
    return resources;
}

uint64_t CheckedAdd(
    uint64_t lhs,
    uint64_t rhs,
    const char* description)
{
    TORCH_CHECK(
        rhs <= std::numeric_limits<uint64_t>::max() - lhs,
        "moe_lora_bgmv_fused: ", description, " overflows uint64");
    return lhs + rhs;
}

uint64_t CheckedMultiply(
    uint64_t lhs,
    uint64_t rhs,
    const char* description)
{
    TORCH_CHECK(
        lhs == 0 || rhs <= std::numeric_limits<uint64_t>::max() / lhs,
        "moe_lora_bgmv_fused: ", description, " overflows uint64");
    return lhs * rhs;
}

uint64_t CeilDivide(uint64_t value, uint64_t divisor)
{
    TORCH_CHECK(
        divisor != 0,
        "moe_lora_bgmv_fused: ceil-divisor must be non-zero");
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

uint64_t AlignUp32Bytes(uint64_t bytes)
{
    constexpr uint64_t blockBytes = 32;
    return CheckedMultiply(
        CeilDivide(bytes, blockBytes), blockBytes, "aligned buffer size");
}

void AddAlignedBufferBytes(
    uint64_t& totalBytes,
    uint64_t elementCount,
    uint64_t elementBytes)
{
    const uint64_t bufferBytes = AlignUp32Bytes(CheckedMultiply(
        elementCount, elementBytes, "UB buffer size"));
    totalBytes = CheckedAdd(totalBytes, bufferBytes, "total UB size");
}

uint32_t CheckedUint32(uint64_t value, const char* description)
{
    TORCH_CHECK(
        value <= std::numeric_limits<uint32_t>::max(),
        "moe_lora_bgmv_fused: ", description,
        " exceeds uint32 launch limits");
    return static_cast<uint32_t>(value);
}

bool IsSupportedRank(int64_t rank)
{
    return rank == 8 || rank == 16 || rank == 32 || rank == 64;
}

uint64_t CalculateFastUbBytes(
    uint32_t inputHiddenDim,
    uint32_t indexElementBytes,
    uint32_t dataElementBytes,
    bool reuseFp32Weight,
    uint32_t groupRows)
{
    const uint64_t weightBufferElements = std::max<uint64_t>(
        inputHiddenDim, kWeightTileElements);
    uint64_t bytes = 0;
    AddAlignedBufferBytes(bytes, groupRows, indexElementBytes);
    AddAlignedBufferBytes(
        bytes,
        CheckedMultiply(groupRows, inputHiddenDim, "grouped input size"),
        dataElementBytes);
    AddAlignedBufferBytes(bytes, weightBufferElements, dataElementBytes);
    AddAlignedBufferBytes(bytes, kOutputTileElements, dataElementBytes);
    AddAlignedBufferBytes(bytes, kOutputTileElements, dataElementBytes);
    AddAlignedBufferBytes(
        bytes,
        CheckedMultiply(groupRows, inputHiddenDim, "grouped FP32 input size"),
        sizeof(float));
    if (reuseFp32Weight) {
        AddAlignedBufferBytes(bytes, weightBufferElements, sizeof(float));
    }
    AddAlignedBufferBytes(bytes, weightBufferElements, sizeof(float));
    AddAlignedBufferBytes(
        bytes,
        CheckedMultiply(groupRows, kFastRank, "grouped rank size"),
        sizeof(float));
    AddAlignedBufferBytes(bytes, 256, 1);
    AddAlignedBufferBytes(bytes, kOutputTileElements, sizeof(float));
    AddAlignedBufferBytes(bytes, kOutputTileElements, sizeof(float));
    return bytes;
}

uint64_t CalculateGenericUbBytes(
    uint32_t rank,
    uint32_t outputHiddenDim,
    uint32_t indexElementBytes,
    uint32_t dataElementBytes)
{
    const uint64_t outputTileElements = kGenericTileElements / rank;
    uint64_t bytes = 0;
    AddAlignedBufferBytes(bytes, 1, indexElementBytes);
    AddAlignedBufferBytes(bytes, kGenericTileElements, dataElementBytes);
    AddAlignedBufferBytes(bytes, kGenericTileElements, dataElementBytes);
    AddAlignedBufferBytes(bytes, outputHiddenDim, dataElementBytes);
    AddAlignedBufferBytes(bytes, kGenericTileElements, sizeof(float));
    AddAlignedBufferBytes(bytes, kGenericTileElements, sizeof(float));
    AddAlignedBufferBytes(
        bytes,
        CheckedMultiply(2, rank, "Generic rank partial size"),
        sizeof(float));
    AddAlignedBufferBytes(bytes, kGenericReduceTmpBytes, 1);
    AddAlignedBufferBytes(bytes, outputTileElements, sizeof(float));
    AddAlignedBufferBytes(bytes, outputTileElements, sizeof(float));
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
    const int64_t rank64 = loraA.size(1);
    TORCH_CHECK(
        loraB.size(2) == rank64,
        "moe_lora_bgmv_fused: lora_b rank must match lora_a rank");
    TORCH_CHECK(
        IsSupportedRank(rank64),
        "moe_lora_bgmv_fused: rank must be one of 8, 16, 32 or 64");
    TORCH_CHECK(
        loraA.size(2) == x.size(1),
        "moe_lora_bgmv_fused: lora_a input dimension must match x");
    TORCH_CHECK(
        sliceSize == loraB.size(1),
        "moe_lora_bgmv_fused: slice_size must match lora_b output dimension");
    TORCH_CHECK(
        sliceOffset >= 0 && sliceSize > 0 && sliceSize <= y.size(1) &&
            sliceOffset <= y.size(1) - sliceSize,
        "moe_lora_bgmv_fused: output slice is out of range");
    TORCH_CHECK(
        x.size(1) > 0 && x.size(1) <= kMaxHiddenDim,
        "moe_lora_bgmv_fused: input hidden dimension must be in [1, 16384]");
    TORCH_CHECK(
        sliceSize <= kMaxHiddenDim,
        "moe_lora_bgmv_fused: output hidden dimension must be in [1, 16384]");
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
    const uint32_t numRows = CheckedUint32(
        static_cast<uint64_t>(numRows64), "row count");
    const uint32_t inputHiddenDim = CheckedUint32(
        static_cast<uint64_t>(x.size(1)), "input hidden dimension");
    const uint32_t outputHiddenDim = CheckedUint32(
        static_cast<uint64_t>(sliceSize), "output hidden dimension");
    const uint32_t outputFullDim = CheckedUint32(
        static_cast<uint64_t>(y.size(1)), "full output dimension");
    const uint32_t sliceOffset32 = CheckedUint32(
        static_cast<uint64_t>(sliceOffset), "slice offset");
    const uint32_t rank = CheckedUint32(
        static_cast<uint64_t>(rank64), "rank");

    const BgmvFusedDeviceResources& resources =
        GetBgmvFusedDeviceResources();
    const uint32_t indexElementBytes =
        indices.scalar_type() == at::kInt ? sizeof(int32_t) : sizeof(int64_t);
    const uint32_t dataElementBytes = CheckedUint32(
        static_cast<uint64_t>(x.element_size()), "data element size");
    const bool useFastKernel =
        rank == kFastRank &&
        inputHiddenDim <= kFastMaxHiddenDim &&
        outputHiddenDim <= kFastMaxHiddenDim;
    const bool reuseFp32Weight =
        numRows >= kFp32ReuseMinRows ||
        inputHiddenDim != kWideInputThreshold ||
        outputHiddenDim > kWideInputThreshold;
    const uint32_t groupRows = useFastKernel ?
        (inputHiddenDim > kWideInputThreshold ?
            kWideInputGroupRows : kDefaultGroupRows) :
        1;
    const uint64_t requiredUbBytes = useFastKernel ?
        CalculateFastUbBytes(
            inputHiddenDim,
            indexElementBytes,
            dataElementBytes,
            reuseFp32Weight,
            groupRows) :
        CalculateGenericUbBytes(
            rank,
            outputHiddenDim,
            indexElementBytes,
            dataElementBytes);
    if (resources.ubBytes != 0) {
        TORCH_CHECK(
            requiredUbBytes <= resources.ubBytes,
            "moe_lora_bgmv_fused: requires ", requiredUbBytes,
            " UB bytes, but device reports ", resources.ubBytes);
    }

    const bool useBalancedRows =
        !useFastKernel || numRows <= kBalancedCoreMaxRows;
    uint64_t desiredCoreNum64 = 0;
    if (useBalancedRows) {
        desiredCoreNum64 = std::min<uint64_t>(
            numRows, resources.vectorCoreNum);
    } else {
        desiredCoreNum64 = std::min<uint64_t>(
            CeilDivide(numRows, groupRows), resources.vectorCoreNum);
    }
    uint64_t rowsPerCore64 = CeilDivide(numRows, desiredCoreNum64);
    if (!useBalancedRows) {
        rowsPerCore64 = CheckedMultiply(
            CeilDivide(rowsPerCore64, groupRows),
            groupRows,
            "aligned rows per core");
    }
    const uint64_t coreNum64 = useBalancedRows ?
        desiredCoreNum64 : CeilDivide(numRows, rowsPerCore64);
    const uint32_t rowsPerCore = CheckedUint32(
        rowsPerCore64, "rows per core");
    const uint32_t coreNum = CheckedUint32(coreNum64, "core count");

    void* xPtr = x.data_ptr();
    void* aPtr = loraA.data_ptr();
    void* bPtr = loraB.data_ptr();
    void* indicesPtr = indices.data_ptr();
    void* yPtr = y.data_ptr();
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
        coreNum,
        rank,
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
            coreNum,
            rank,
            scaleFloat,
            indicesIsInt32);
        return 0;
    });
    cmd.Run();
    return y;
}

}  // namespace vllm_ascend
