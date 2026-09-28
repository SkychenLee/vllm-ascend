// SPDX-License-Identifier: Apache-2.0
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <algorithm>
#include "kernels/moe_lora_combined_indices.h"

extern "C" uint64_t vllm_ascend_recover_vector_cores();

namespace vllm_ascend {
namespace {
void check_combined(const at::Tensor& experts, const at::Tensor& slots,
                    const at::Tensor& enabled, int64_t num_experts) {
    TORCH_CHECK(experts.dim() == 1 && slots.dim() == 1 && enabled.dim() == 1,
                "moe_lora_combined_indices: inputs must be 1D");
    TORCH_CHECK(experts.numel() == slots.numel(), "moe_lora_combined_indices: row count mismatch");
    TORCH_CHECK(experts.scalar_type() == at::kInt || experts.scalar_type() == at::kLong,
                "moe_lora_combined_indices: experts must be int32 or int64");
    TORCH_CHECK(slots.scalar_type() == at::kLong && enabled.scalar_type() == at::kInt,
                "moe_lora_combined_indices: slots must be int64 and enabled int32");
    TORCH_CHECK(experts.is_contiguous() && slots.is_contiguous() && enabled.is_contiguous(),
                "moe_lora_combined_indices: contiguous inputs required");
    TORCH_CHECK(experts.device() == slots.device() && experts.device() == enabled.device(),
                "moe_lora_combined_indices: input devices must match");
    TORCH_CHECK(num_experts > 0, "moe_lora_combined_indices: num_experts must be positive");
    TORCH_CHECK(enabled.numel() > 0 && enabled.numel() <= MOE_LORA_COMBINED_MAX_ADAPTERS,
                "moe_lora_combined_indices: enabled length must be 1..64");
}
}

at::Tensor moe_lora_combined_indices_meta(const at::Tensor& experts, const at::Tensor& slots,
                                         const at::Tensor& enabled, int64_t num_experts) {
    check_combined(experts, slots, enabled, num_experts);
    return at::empty(experts.sizes(), experts.options().dtype(at::kLong));
}

at::Tensor moe_lora_combined_indices(const at::Tensor& experts, const at::Tensor& slots,
                                    const at::Tensor& enabled, int64_t num_experts) {
    check_combined(experts, slots, enabled, num_experts);
    TORCH_CHECK(experts.device().type() == c10::DeviceType::PrivateUse1,
                "moe_lora_combined_indices: expected NPU inputs");
    const c10_npu::NPUGuard guard(experts.device());
    auto output = at::empty(experts.sizes(), experts.options().dtype(at::kLong));
    const uint64_t rows = experts.numel();
    if (rows == 0) return output;
    uint64_t available = vllm_ascend_recover_vector_cores();
    TORCH_CHECK(available > 0, "moe_lora_combined_indices: cannot query vector cores");
    const uint32_t cores = std::min(available, (rows + 7) / 8);
    const uint32_t adapters = enabled.numel();
    const uint32_t expert_bytes = experts.element_size();
    auto e = experts.data_ptr(), s = slots.data_ptr(), m = enabled.data_ptr(), o = output.data_ptr();
    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    at_npu::native::OpCommand command;
    command.Name("moe_lora_combined_indices");
    command.SetCustomHandler([=]() -> int {
        moe_lora_combined_indices_impl(stream, e, s, m, o, rows, num_experts, adapters, expert_bytes, cores);
        return 0;
    });
    command.Run();
    return output;
}
}

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("moe_lora_combined_indices(Tensor experts, Tensor slots, Tensor enabled, int num_experts) -> Tensor");
    ops.impl("moe_lora_combined_indices", c10::DispatchKey::PrivateUse1, &vllm_ascend::moe_lora_combined_indices);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
    ops.impl("moe_lora_combined_indices", &vllm_ascend::moe_lora_combined_indices_meta);
}
#endif
