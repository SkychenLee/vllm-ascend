#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections.abc import Callable
from typing import Any
import os
import time

import torch

from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.mxfp_compat import (
    ensure_mxfp4_linear_available,
)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme
from vllm.logger import logger
from vllm_ascend.quantization.utils import quant_dequant_hif4, unpack_dynamic_hif4_tensor, unpack_hif4_scale_from_fp32_graph

@register_scheme("W4A4_HIFP4", "linear")
class AscendW4A4HiFPDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A4_HIFP4 (Microscaling FP4) quantization.
    """

    model_dtype = None

    def __init__(self):
        ensure_mxfp4_linear_available("W4A4_HIFP4 linear quantization")
        vllm_config = get_current_vllm_config()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 64)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.float32)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # reshape x for Qwen VL models
        original_shape = x.shape
        output_dtype = x.dtype
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
            
        # 激活、权重scale的伪量化
        quantized_x = quant_dequant_hif4(x)
        # 解析weight_scale
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        # 注意w的转置
        output = torch.mm(quantized_x, layer.weight)

        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output.to(output_dtype)

    def process_weights_after_loading(self, layer):
        """Process weights after loading for HiFP4 inference.

        This method transforms weights for NPU HiFP4 computation:
        - weight_scale: (n_dim, k_dim) -> (k_dim//2, n_dim, 2)
        """        

        weight_ori = unpack_dynamic_hif4_tensor(layer.weight)
        layer.weight.data = unpack_hif4_scale_from_fp32_graph(weight_ori, layer.weight_scale)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        if hasattr(layer, "weight_scale"):
            del layer.weight_scale

 

@register_scheme("W4A4_HIFP4", "moe")
class AscendW4A4HiFPDynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W4A4_HIFP4."""

    model_dtype = None
    quant_type: QuantType = QuantType.NONE

    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 64)
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}

        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // 2, dtype=torch.uint8
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // 2, dtype=torch.uint8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size , dtype=torch.float32
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.float32
        )
        return param_dict
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts
        )
        if layer.vllm_config.model_config is not None and layer.vllm_config.model_config.enable_return_routed_experts:
            capturer = getattr(layer, "_ascend_routed_experts_capturer", None)
            if capturer is not None:
                capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)

        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # NOTE: In the MoECommType.FUSED_MC2 branch, we wrap weights (w1, w2) into lists
        # and provide dummy scales (w1_scale, w2_scale). This is required because:
        # The underlying Ascend fused operator (e.g., dispatch_ffn_combine) expects
        # inputs in a list format.
        # TODO: Passing an empty tensor as scale for float (BF16) cases is semantically
        # incorrect. The ideal solution is to pass None. However, if the underlying
        # dispatch_ffn_combine C++ operator does not support None for the scale argument
        # (due to signature constraints), we are forced to use a placeholder empty tensor.
        # This TODO tracks the requirement to update the C++ operator to accept Optional[Tensor]
        # or None for scales in non-quantized scenarios.
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            w1 = [layer.w13_weight]
            w2 = [layer.w2_weight]
        else:
            w1 = layer.w13_weight
            w2 = layer.w2_weight
        x_q = quant_dequant_hif4(x)
        
        return  moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x_q,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
            )
        )

            

    def process_weights_after_loading(self, layer):
        # msit load weight
        # w13_weight_scale  w2_weight_scale 反量化 ——> layer.w13_weight layer.w2_weight
        # w13: up-gate w2: down

        layer.w13_weight.data = unpack_dynamic_hif4_tensor(layer.w13_weight)
        layer.w13_weight.data =  unpack_hif4_scale_from_fp32_graph(layer.w13_weight, layer.w13_weight_scale)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        
        layer.w2_weight.data = unpack_dynamic_hif4_tensor(layer.w2_weight) 
        layer.w2_weight.data =  unpack_hif4_scale_from_fp32_graph(layer.w2_weight, layer.w2_weight_scale)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
