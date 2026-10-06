// SPDX-License-Identifier: Apache-2.0
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <algorithm>
#include <cmath>
#include "kernels/moe_lora_activation.h"

namespace vllm_ascend {
namespace {
void check_activation(const at::Tensor& input, double limit, int64_t coreLimit) {
    TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
                "moe_lora_activation_quant: contiguous 2D gate/up required");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "moe_lora_activation_quant: FP16/BF16 required");
    TORCH_CHECK(input.size(1) == 2 * MOE_LORA_ACTIVATION_WIDTH &&
                input.size(0) <= MOE_LORA_ACTIVATION_MAX_ROWS,
                "moe_lora_activation_quant: supported shape is [0..384,4096]");
    TORCH_CHECK(std::isfinite(limit) && limit >= 0 && limit <= 65504,
                "moe_lora_activation_quant: finite limit in [0,65504] required");
    TORCH_CHECK(coreLimit >= 0 && coreLimit <= 64,
                "moe_lora_activation_quant: static core_limit must be in [0,64]");
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_lora_activation_quant_meta(
    const at::Tensor& input, double limit, int64_t coreLimit) {
    check_activation(input, limit, coreLimit);
    return {at::empty({input.size(0), MOE_LORA_ACTIVATION_WIDTH}, input.options()),
            at::empty({input.size(0), MOE_LORA_ACTIVATION_WIDTH}, input.options().dtype(at::kChar)),
            at::empty({input.size(0)}, input.options().dtype(at::kFloat))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_lora_activation_quant(
    const at::Tensor& input, double limit, int64_t coreLimit) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                "moe_lora_activation_quant: NPU input required");
    const c10_npu::NPUGuard guard(input.device());
    // Logical contiguity also holds for FRACTAL_NZ tensors. This kernel
    // addresses raw storage as ordinary row-major gate/up; inspect the NPU
    // descriptor on the runtime dispatch only, keeping Meta device-free.
    const auto storageFormat = at_npu::native::get_npu_format(input);
    TORCH_CHECK(storageFormat == ACL_FORMAT_ND || storageFormat == ACL_FORMAT_NCHW,
                "moe_lora_activation_quant: ND/base storage (format 0 or 2) required; "
                "blocked NPU storage is unsupported");
    auto output = moe_lora_activation_quant_meta(input, limit, coreLimit);
    if (input.size(0) == 0) return output;
    int32_t device = -1;
    int64_t available = 0;
    TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS &&
                aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &available) == ACL_SUCCESS && available > 0,
                "moe_lora_activation_quant: cannot query vector core count");
    uint32_t rows = input.size(0);
    uint32_t cores = static_cast<uint32_t>(std::min<int64_t>(available, rows));
    if (coreLimit > 0) cores = std::min<uint32_t>(cores, static_cast<uint32_t>(coreLimit));
    else if (rows <= 48) cores = std::min<uint32_t>(cores, 24);
    auto type = input.scalar_type() == at::kHalf ? AscendType::FP16 : AscendType::BF16;
    auto x = input.data_ptr();
    auto f = std::get<0>(output).data_ptr();
    auto q = std::get<1>(output).data_ptr();
    auto s = std::get<2>(output).data_ptr();
    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    float floatLimit = static_cast<float>(limit);
    at_npu::native::OpCommand command;
    command.Name("moe_lora_activation_quant");
    command.SetCustomHandler([=]() -> int {
        moe_lora_activation_impl(type, stream, x, f, q, s, rows, cores, floatLimit);
        return 0;
    });
    command.Run();
    return output;
}
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("moe_lora_activation_quant(Tensor input, float limit=0., int core_limit=0) -> (Tensor, Tensor, Tensor)");
    ops.impl("moe_lora_activation_quant", c10::DispatchKey::PrivateUse1,
             &vllm_ascend::moe_lora_activation_quant);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
    ops.impl("moe_lora_activation_quant", &vllm_ascend::moe_lora_activation_quant_meta);
}
#endif
