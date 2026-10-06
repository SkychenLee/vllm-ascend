/* SPDX-License-Identifier: Apache-2.0 */
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "kernels/moe_lora_ep_routing.h"

namespace vllm_ascend {
namespace {
bool integer(const at::Tensor& t) { return t.scalar_type() == at::kInt || t.scalar_type() == at::kLong; }
void check_routing(const at::Tensor& expanded, const at::Tensor& topk,
    const at::Tensor& slots, const at::Tensor& map, int64_t capacity)
{
    TORCH_CHECK(expanded.dim() == 1 && topk.dim() == 2 && slots.dim() == 1 && map.dim() == 1,
                "moe_lora_ep_routing: expected 1D expanded/slots/map and 2D topk");
    TORCH_CHECK(expanded.is_contiguous() && topk.is_contiguous() && slots.is_contiguous() && map.is_contiguous(),
                "moe_lora_ep_routing: contiguous metadata required");
    TORCH_CHECK(integer(expanded) && integer(topk) && integer(map) && slots.scalar_type() == at::kLong,
                "moe_lora_ep_routing: int32/int64 indices/map and int64 slots required");
    TORCH_CHECK(expanded.device() == topk.device() && expanded.device() == slots.device() && expanded.device() == map.device(),
                "moe_lora_ep_routing: metadata must share one device");
    TORCH_CHECK(expanded.numel() == topk.numel() && slots.numel() == topk.size(0),
                "moe_lora_ep_routing: one expanded index per pair and one slot per token required");
    TORCH_CHECK(topk.size(1) > 0 && expanded.numel() <= MOE_LORA_EP_ROUTING_MAX_PAIRS &&
                map.numel() > 0 && map.numel() <= MOE_LORA_EP_ROUTING_MAX_EXPERTS &&
                capacity >= 0 && capacity <= expanded.numel(),
                "moe_lora_ep_routing: native bounds are pairs<=384, experts in [1,512], capacity in [0,pairs]");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> moe_lora_ep_routing_meta(const at::Tensor& expanded,
    const at::Tensor& topk, const at::Tensor& slots, const at::Tensor& map, int64_t capacity)
{
    check_routing(expanded, topk, slots, map, capacity);
    return {at::empty({capacity}, slots.options()), at::empty({capacity}, slots.options())};
}

std::tuple<at::Tensor, at::Tensor> moe_lora_ep_routing(const at::Tensor& expanded,
    const at::Tensor& topk, const at::Tensor& slots, const at::Tensor& map, int64_t capacity)
{
    TORCH_CHECK(expanded.device().type() == c10::DeviceType::PrivateUse1,
                "moe_lora_ep_routing: NPU metadata required");
    const c10_npu::NPUGuard guard(expanded.device());
    auto output = moe_lora_ep_routing_meta(expanded, topk, slots, map, capacity);
    // Logical contiguity does not imply base storage for integer metadata.
    // Only runtime NPU dispatch inspects descriptors; Meta remains device-free.
    for (const auto* metadata : {&expanded, &topk, &slots, &map}) {
        const auto format = at_npu::native::get_npu_format(*metadata);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "moe_lora_ep_routing: ND/base storage (format 0 or 2) required; "
                    "blocked NPU storage is unsupported");
    }
    if (capacity == 0) return output;
    auto x = expanded.data_ptr(), k = topk.data_ptr(), s = slots.data_ptr(), m = map.data_ptr();
    auto e = std::get<0>(output).data_ptr(), a = std::get<1>(output).data_ptr();
    const uint32_t pairs = expanded.numel(), rows = capacity, topK = topk.size(1), experts = map.numel();
    const uint32_t xb = expanded.element_size(), kb = topk.element_size(), mb = map.element_size();
    void* stream = c10_npu::getCurrentNPUStream().stream(true);
    at_npu::native::OpCommand command;
    command.Name("moe_lora_ep_routing");
    command.SetCustomHandler([=]() -> int {
        moe_lora_ep_routing_impl(stream, x, k, s, m, e, a, pairs, rows, topK, experts, xb, kb, mb);
        return 0;
    });
    command.Run();
    return output;
}
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("moe_lora_ep_routing(Tensor expanded_row_idx, Tensor topk_ids, Tensor prepared_lora_indices, Tensor expert_map, int num_dispatched_rows) -> (Tensor, Tensor)");
    ops.impl("moe_lora_ep_routing", c10::DispatchKey::PrivateUse1, &vllm_ascend::moe_lora_ep_routing);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
    ops.impl("moe_lora_ep_routing", &vllm_ascend::moe_lora_ep_routing_meta);
}
#endif
