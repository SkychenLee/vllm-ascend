#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <limits>

namespace vllm_ascend {

// Shared by the device binding and Meta: metadata only, no pointer/value reads.
inline void check_bgmv_shrink_pair_metadata(
    const at::Tensor& x, const at::Tensor& weight0, const at::Tensor& weight1,
    const at::Tensor& indices, const at::Tensor& y_pair)
{
    TORCH_CHECK(x.dim() == 2, "bgmv_shrink_pair: x must be [M, K]");
    TORCH_CHECK(weight0.dim() == 3 && weight1.dim() == 3,
                "bgmv_shrink_pair: weights must be [slots, R, K]");
    TORCH_CHECK(indices.dim() == 1, "bgmv_shrink_pair: indices must be [M]");
    TORCH_CHECK(y_pair.dim() == 3 && y_pair.size(0) == 2,
                "bgmv_shrink_pair: y_pair must be [2, M, R]");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "bgmv_shrink_pair: x must be FP16 or BF16");
    TORCH_CHECK(weight0.scalar_type() == x.scalar_type() && weight1.scalar_type() == x.scalar_type(),
                "bgmv_shrink_pair: weights must have the input dtype");
    TORCH_CHECK(y_pair.scalar_type() == at::kFloat && indices.scalar_type() == at::kLong,
                "bgmv_shrink_pair: output must be FP32 and indices int64");
    TORCH_CHECK(x.device() == weight0.device() && x.device() == weight1.device() &&
                x.device() == indices.device() && x.device() == y_pair.device(),
                "bgmv_shrink_pair: all tensors must be on the same device");
    TORCH_CHECK(x.is_contiguous() && weight0.is_contiguous() && weight1.is_contiguous() &&
                indices.is_contiguous() && y_pair.is_contiguous(),
                "bgmv_shrink_pair: all tensors must be contiguous");
    TORCH_CHECK(weight0.sizes() == weight1.sizes() && weight0.size(0) > 0,
                "bgmv_shrink_pair: weights must have the same nonempty slot layout");
    TORCH_CHECK(x.size(0) == indices.size(0) && x.size(0) == y_pair.size(1),
                "bgmv_shrink_pair: row counts must agree");
    TORCH_CHECK(weight0.size(1) == y_pair.size(2) && weight0.size(2) == x.size(1),
                "bgmv_shrink_pair: weight/output dimensions must agree");
    TORCH_CHECK(y_pair.size(2) > 0 && x.size(1) > y_pair.size(2),
                "bgmv_shrink_pair: existing shrink domain requires K > R > 0");
    // The existing kernel uses signed row/column loops and uint32 byte lengths.
    constexpr int64_t signed_limit = std::numeric_limits<int32_t>::max();
    constexpr int64_t rank_byte_limit = std::numeric_limits<uint32_t>::max() / sizeof(float);
    TORCH_CHECK(x.size(0) <= signed_limit && x.size(1) <= signed_limit &&
                y_pair.size(2) <= rank_byte_limit,
                "bgmv_shrink_pair: shape exceeds existing kernel integer limits");
}

}  // namespace vllm_ascend
