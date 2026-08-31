#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch


def moe_lora_prepare_bgmv_indices(
    routed_lora_slots: torch.Tensor,
    group_list: torch.Tensor,
    adapter_enabled: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(routed_lora_slots, dtype=torch.int64)
    torch.ops._C_ascend.moe_lora_prepare_bgmv_indices(
        routed_lora_slots,
        group_list,
        adapter_enabled,
        output,
    )
    return output


def moe_lora_prepare_composite_gmm_routing(
    routed_lora_slots: torch.Tensor,
    group_list: torch.Tensor,
    adapter_enabled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_groups = group_list.numel() * adapter_enabled.numel()
    group_ids = torch.empty_like(routed_lora_slots, dtype=torch.int32)
    composite_group_list = torch.empty(
        num_groups,
        dtype=torch.int64,
        device=group_list.device,
    )
    enabled = torch.empty_like(routed_lora_slots, dtype=torch.bool)
    torch.ops._C_ascend.moe_lora_prepare_composite_gmm_routing(
        routed_lora_slots,
        group_list,
        adapter_enabled,
        group_ids,
        composite_group_list,
        enabled,
    )
    return group_ids, composite_group_list, enabled


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    return torch.ops._C_ascend.bgmv_shrink(
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    slice_offset = 0
    slice_size = output_tensor.size(1)
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    return torch.ops._C_ascend.sgmv_shrink(
        inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, output_tensor, scaling
    )


def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    slice_offset = 0
    slice_size = output_tensor.size(1)
    return torch.ops._C_ascend.sgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        seq_len_tensor,
        output_tensor,
        slice_offset,
        slice_size,
    )


def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    return torch.ops._C_ascend.sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )
