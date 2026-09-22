// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
  #include <torch/library.h>
  #include <ATen/ATen.h>
  #include <ATen/MemoryOverlap.h>
  #include <acl/acl_rt.h>
  #include <torch_npu/csrc/core/npu/NPUGuard.h>
  #include <torch_npu/csrc/core/npu/NPUStream.h>
  #include <torch_npu/csrc/framework/OpCommand.h>
  #include <cmath>
  #include <limits>
  #include "kernels/moe_lora_int8.h"

namespace vllm_ascend {
namespace {
constexpr int64_t MAX_RANK = 512;
constexpr int64_t MAX_WIDTH = 8192;
void tensor_check(const at::Tensor& t, const at::Tensor& reference, at::ScalarType dtype, int64_t ndim) {
  TORCH_CHECK(t.device() == reference.device() && t.scalar_type() == dtype && t.dim() == ndim && t.is_contiguous(),
              "MoE INT8 LoRA: invalid tensor dtype, device, dimensions or non-contiguous layout");
}
uint32_t core_count() {
  int32_t device = -1;
  int64_t cores = 0;
  TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS &&
                  aclGetDeviceCapability(device, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cores) == ACL_SUCCESS && cores > 0,
              "MoE INT8 LoRA: cannot query vector core count");
  return static_cast<uint32_t>(cores);
}
void shrink_check(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ids, const at::Tensor& scale,
                  const at::Tensor& y) {
  tensor_check(x, x, at::kChar, 2);
  tensor_check(ids, x, at::kLong, 1);
  tensor_check(scale, x, at::kFloat, 1);
  tensor_check(y, x, at::kFloat, 2);
  TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kBFloat16, "LoRA weights must be FP16/BF16");
  tensor_check(w, x, w.scalar_type(), 3);
  TORCH_CHECK(x.size(0) == ids.size(0) && x.size(0) == scale.size(0) && x.size(0) == y.size(0),
              "LoRA row/scale mismatch");
  TORCH_CHECK(x.size(1) == w.size(2) && w.size(1) == y.size(1) && w.size(0) > 0 && x.size(1) > 0 && y.size(1) > 0 &&
                  y.size(1) <= MAX_RANK,
              "Invalid LoRA weight/rank shape");
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int32_t>::max() && x.size(1) <= std::numeric_limits<int32_t>::max() &&
                  w.size(0) <= std::numeric_limits<int32_t>::max(),
              "LoRA shape exceeds kernel index range");
}
void expand_check(const at::Tensor& base, const at::Tensor& g, const at::Tensor& u, const at::Tensor& bg,
                  const at::Tensor& bu, const at::Tensor& ids, const c10::optional<at::Tensor>& topk, double limit) {
  TORCH_CHECK(base.scalar_type() == at::kHalf || base.scalar_type() == at::kBFloat16,
              "MoE base output must be FP16/BF16");
  tensor_check(base, base, base.scalar_type(), 2);
  tensor_check(g, base, at::kFloat, 2);
  tensor_check(u, base, at::kFloat, 2);
  tensor_check(bg, base, base.scalar_type(), 3);
  tensor_check(bu, base, base.scalar_type(), 3);
  tensor_check(ids, base, at::kLong, 1);
  auto m = base.size(0), h = base.size(1) / 2, r = g.size(1);
  TORCH_CHECK(base.size(1) % 2 == 0 && h > 0 && h <= MAX_WIDTH && r > 0 && r <= MAX_RANK,
              "Unsupported MoE width or LoRA rank");
  TORCH_CHECK(g.size(0) == m && u.sizes() == g.sizes() && ids.size(0) == m && bg.sizes() == bu.sizes() &&
                  bg.size(0) > 0 && bg.size(1) == h && bg.size(2) == r,
              "MoE LoRA gate/up shape mismatch");
  TORCH_CHECK(m <= std::numeric_limits<int32_t>::max() && bg.size(0) <= std::numeric_limits<int32_t>::max(),
              "MoE shape exceeds kernel index range");
  TORCH_CHECK(std::isfinite(limit) && limit >= 0, "SwiGLU limit must be finite and nonnegative");
  if (topk) {
    tensor_check(*topk, base, at::kFloat, 1);
    TORCH_CHECK(topk->numel() == m, "topk scale row mismatch");
  }
}
void pair_shrink_check(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ids, const at::Tensor& scale,
                       const at::Tensor& y) {
  tensor_check(x, x, at::kChar, 2);
  tensor_check(ids, x, at::kLong, 1);
  tensor_check(scale, x, at::kFloat, 1);
  tensor_check(y, x, at::kFloat, 2);
  TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kBFloat16, "LoRA weights must be FP16/BF16");
  tensor_check(w, x, w.scalar_type(), 4);
  TORCH_CHECK(w.size(0) == 2 && w.size(1) > 0 && w.size(2) > 0 && w.size(2) <= MAX_RANK && w.size(3) == x.size(1) &&
                  x.size(1) > 0 && y.size(1) == 2 * w.size(2),
              "Invalid paired LoRA weight/rank shape");
  TORCH_CHECK(x.size(0) == ids.size(0) && x.size(0) == scale.size(0) && x.size(0) == y.size(0),
              "LoRA row/scale mismatch");
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int32_t>::max() && x.size(1) <= std::numeric_limits<int32_t>::max() &&
                  w.size(1) <= std::numeric_limits<int32_t>::max(),
              "LoRA shape exceeds kernel index range");
}
void pair_expand_check(const at::Tensor& base, const at::Tensor& a, const at::Tensor& bg, const at::Tensor& bu,
                       const at::Tensor& ids, const c10::optional<at::Tensor>& topk, double limit) {
  TORCH_CHECK(base.scalar_type() == at::kHalf || base.scalar_type() == at::kBFloat16,
              "MoE base output must be FP16/BF16");
  tensor_check(base, base, base.scalar_type(), 2);
  tensor_check(a, base, at::kFloat, 3);
  tensor_check(bg, base, base.scalar_type(), 3);
  tensor_check(bu, base, base.scalar_type(), 3);
  tensor_check(ids, base, at::kLong, 1);
  auto m = base.size(0), h = base.size(1) / 2, r = bg.size(2);
  TORCH_CHECK(base.size(1) % 2 == 0 && h > 0 && h <= MAX_WIDTH && r > 0 && r <= MAX_RANK,
              "Unsupported MoE width or LoRA rank");
  TORCH_CHECK(a.size(0) > 0 && a.size(0) <= r && a.size(1) == m && a.size(2) > 0 && a.size(2) % 2 == 0 &&
                  a.size(0) * (a.size(2) / 2) == r && ids.size(0) == m && bg.sizes() == bu.sizes() && bg.size(0) > 0 &&
                  bg.size(1) == h,
              "MoE LoRA paired/TP rank shape mismatch");
  TORCH_CHECK(m <= std::numeric_limits<int32_t>::max() && bg.size(0) <= std::numeric_limits<int32_t>::max() &&
                  (m == 0 || (2 * m - 1) * (a.size(2) / 2) * 4 <= std::numeric_limits<uint32_t>::max()),
              "MoE shape exceeds kernel index/stride range");
  TORCH_CHECK(std::isfinite(limit) && limit >= 0, "SwiGLU limit must be finite and nonnegative");
  if (topk) {
    tensor_check(*topk, base, at::kFloat, 1);
    TORCH_CHECK(topk->numel() == m, "topk scale row mismatch");
  }
}
}  // namespace

void bgmv_shrink_int8(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ids, const at::Tensor& scale,
                      at::Tensor& y) {
  shrink_check(x, w, ids, scale, y);
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "Expected NPU tensors");
  for (const auto& input : {x, w, ids, scale}) at::assert_no_overlap(y, input);
  if (x.size(0) == 0) return;
  const c10_npu::NPUGuard guard(x.device());
  auto cores = core_count();
  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto type = w.scalar_type() == at::kHalf ? AscendType::FP16 : AscendType::BF16;
  auto xp = x.data_ptr(), wp = w.data_ptr(), ip = ids.data_ptr(), sp = scale.data_ptr(), yp = y.data_ptr();
  uint32_t m = x.size(0), k = x.size(1), r = w.size(1), groups = w.size(0);
  at_npu::native::OpCommand cmd;
  cmd.Name("bgmv_shrink_int8");
  cmd.SetCustomHandler([=]() -> int {
    bgmv_shrink_int8_impl(type, stream, xp, wp, ip, sp, yp, m, k, r, groups, cores);
    return 0;
  });
  cmd.Run();
}
void bgmv_shrink_int8_meta(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ids, const at::Tensor& scale,
                           at::Tensor& y) {
  shrink_check(x, w, ids, scale, y);
}
std::tuple<at::Tensor, at::Tensor> expand_meta(const at::Tensor& base, const at::Tensor& g, const at::Tensor& u,
                                               const at::Tensor& bg, const at::Tensor& bu, const at::Tensor& ids,
                                               const c10::optional<at::Tensor>& topk, double limit) {
  expand_check(base, g, u, bg, bu, ids, topk, limit);
  return {at::empty({base.size(0), base.size(1) / 2}, base.options().dtype(at::kChar)),
          at::empty({base.size(0)}, base.options().dtype(at::kFloat))};
}
std::tuple<at::Tensor, at::Tensor> expand(const at::Tensor& base, const at::Tensor& g, const at::Tensor& u,
                                          const at::Tensor& bg, const at::Tensor& bu, const at::Tensor& ids,
                                          const c10::optional<at::Tensor>& topk, double limit) {
  auto result = expand_meta(base, g, u, bg, bu, ids, topk, limit);
  TORCH_CHECK(base.device().type() == c10::DeviceType::PrivateUse1, "Expected NPU tensors");
  if (base.size(0) == 0) return result;
  const c10_npu::NPUGuard guard(base.device());
  auto cores = core_count();
  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto type = base.scalar_type() == at::kHalf ? AscendType::FP16 : AscendType::BF16;
  auto bp = base.data_ptr(), gp = g.data_ptr(), up = u.data_ptr(), bgp = bg.data_ptr(), bup = bu.data_ptr(),
       ip = ids.data_ptr();
  void* tp = topk ? topk->data_ptr() : nullptr;
  auto yp = std::get<0>(result).data_ptr(), sp = std::get<1>(result).data_ptr();
  uint32_t m = base.size(0), h = base.size(1) / 2, r = g.size(1), groups = bg.size(0);
  float lim = limit;
  at_npu::native::OpCommand cmd;
  cmd.Name("moe_lora_expand_swiglu_quant");
  cmd.SetCustomHandler([=]() -> int {
    moe_lora_expand_swiglu_quant_impl(type, stream, bp, gp, up, bgp, bup, ip, tp, yp, sp, m, h, r, groups, cores, lim);
    return 0;
  });
  cmd.Run();
  return result;
}
void bgmv_shrink_int8_pair_meta(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ids,
                                const at::Tensor& scale, at::Tensor& y) {
  pair_shrink_check(x, w, ids, scale, y);
}
void bgmv_shrink_int8_pair(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ids, const at::Tensor& scale,
                           at::Tensor& y) {
  pair_shrink_check(x, w, ids, scale, y);
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "Expected NPU tensors");
  for (const auto& input : {x, w, ids, scale}) at::assert_no_overlap(y, input);
  if (x.size(0) == 0) return;
  const c10_npu::NPUGuard guard(x.device());
  auto cores = core_count();
  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto type = w.scalar_type() == at::kHalf ? AscendType::FP16 : AscendType::BF16;
  auto xp = x.data_ptr(), wp = w.data_ptr(), ip = ids.data_ptr(), sp = scale.data_ptr(), yp = y.data_ptr();
  uint32_t m = x.size(0), k = x.size(1), r = w.size(2), groups = w.size(1);
  at_npu::native::OpCommand cmd;
  cmd.Name("bgmv_shrink_int8_pair");
  cmd.SetCustomHandler([=]() -> int {
    bgmv_shrink_int8_pair_impl(type, stream, xp, wp, ip, sp, yp, m, k, r, groups, cores);
    return 0;
  });
  cmd.Run();
}
std::tuple<at::Tensor, at::Tensor> expand_pair_meta(const at::Tensor& base, const at::Tensor& a, const at::Tensor& bg,
                                                    const at::Tensor& bu, const at::Tensor& ids,
                                                    const c10::optional<at::Tensor>& topk, double limit) {
  pair_expand_check(base, a, bg, bu, ids, topk, limit);
  return {at::empty({base.size(0), base.size(1) / 2}, base.options().dtype(at::kChar)),
          at::empty({base.size(0)}, base.options().dtype(at::kFloat))};
}
std::tuple<at::Tensor, at::Tensor> expand_pair(const at::Tensor& base, const at::Tensor& a, const at::Tensor& bg,
                                               const at::Tensor& bu, const at::Tensor& ids,
                                               const c10::optional<at::Tensor>& topk, double limit) {
  auto result = expand_pair_meta(base, a, bg, bu, ids, topk, limit);
  TORCH_CHECK(base.device().type() == c10::DeviceType::PrivateUse1, "Expected NPU tensors");
  if (base.size(0) == 0) return result;
  const c10_npu::NPUGuard guard(base.device());
  auto cores = core_count();
  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto type = base.scalar_type() == at::kHalf ? AscendType::FP16 : AscendType::BF16;
  auto bp = base.data_ptr(), ap = a.data_ptr(), bgp = bg.data_ptr(), bup = bu.data_ptr(), ip = ids.data_ptr();
  void* tp = topk ? topk->data_ptr() : nullptr;
  auto yp = std::get<0>(result).data_ptr(), sp = std::get<1>(result).data_ptr();
  uint32_t m = base.size(0), h = base.size(1) / 2, r = bg.size(2), groups = bg.size(0), shards = a.size(0);
  float lim = limit;
  at_npu::native::OpCommand cmd;
  cmd.Name("moe_lora_expand_swiglu_quant_pair");
  cmd.SetCustomHandler([=]() -> int {
    moe_lora_expand_swiglu_quant_pair_impl(type, stream, bp, ap, bgp, bup, ip, tp, yp, sp, m, h, r, groups, cores, lim,
                                           shards);
    return 0;
  });
  cmd.Run();
  return result;
}
}  // namespace vllm_ascend
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
  ops.def("bgmv_shrink_int8_pair(Tensor x, Tensor weight, Tensor indices, Tensor scale, Tensor(a!) y) -> ()");
  ops.impl("bgmv_shrink_int8_pair", c10::DispatchKey::PrivateUse1, &vllm_ascend::bgmv_shrink_int8_pair);
  ops.def(
      "moe_lora_expand_swiglu_quant_pair(Tensor base, Tensor paired, Tensor bg, Tensor bu, Tensor indices, Tensor? "
      "topk, float limit) -> (Tensor, Tensor)");
  ops.impl("moe_lora_expand_swiglu_quant_pair", c10::DispatchKey::PrivateUse1, &vllm_ascend::expand_pair);
  ops.def("bgmv_shrink_int8(Tensor x, Tensor weight, Tensor indices, Tensor scale, Tensor(a!) y) -> ()");
  ops.impl("bgmv_shrink_int8", c10::DispatchKey::PrivateUse1, &vllm_ascend::bgmv_shrink_int8);
  ops.def(
      "moe_lora_expand_swiglu_quant(Tensor base, Tensor gate, Tensor up, Tensor bg, Tensor bu, Tensor indices, Tensor? "
      "topk, float limit) -> (Tensor, Tensor)");
  ops.impl("moe_lora_expand_swiglu_quant", c10::DispatchKey::PrivateUse1, &vllm_ascend::expand);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
  ops.impl("bgmv_shrink_int8_pair", &vllm_ascend::bgmv_shrink_int8_pair_meta);
  ops.impl("moe_lora_expand_swiglu_quant_pair", &vllm_ascend::expand_pair_meta);
  ops.impl("bgmv_shrink_int8", &vllm_ascend::bgmv_shrink_int8_meta);
  ops.impl("moe_lora_expand_swiglu_quant", &vllm_ascend::expand_meta);
}
#endif
