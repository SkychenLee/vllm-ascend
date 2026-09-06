# SPDX-License-Identifier: Apache-2.0
"""DSV4 routed experts must retain the checkpoint's activation clamp."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models import deepseek_v4


@pytest.mark.parametrize("limit", [None, 0.0, 10.0])
def test_routed_experts_receive_swiglu_limit(limit):
    config = SimpleNamespace(
        swiglu_limit=limit,
        n_routed_experts=4,
        n_shared_experts=None,
        hidden_act="silu",
        hidden_size=8,
        moe_intermediate_size=4,
        num_hash_layers=0,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=False,
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    group = SimpleNamespace(rank_in_group=0, device_group=SimpleNamespace(size=lambda: 1))
    with (
        patch.object(deepseek_v4, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(deepseek_v4, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(deepseek_v4, "get_ep_group", return_value=group),
        patch.object(deepseek_v4, "get_ascend_config", return_value=SimpleNamespace(mix_placement=False)),
        patch.object(deepseek_v4, "ReplicatedLinear", return_value=torch.nn.Module()),
        patch.object(deepseek_v4, "FusedMoE", return_value=torch.nn.Module()) as fused,
        patch.object(deepseek_v4, "rocm_aiter_ops", MagicMock()),
    ):
        deepseek_v4.DeepseekV4MoE(config, parallel, prefix="model.layers.3.mlp")
    assert fused.call_args.kwargs["swiglu_limit"] == limit
