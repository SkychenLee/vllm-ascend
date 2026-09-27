# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.fused_moe import supports_tp_fully_sharded_lora
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEQuantParams,
    MoERoutingParams,
    MoETokenDispatchInput,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("sharded,ep,expected", [(True, False, True), (False, False, False), (True, True, False)])
def test_scope_guard(sharded, ep, expected):
    assert supports_tp_fully_sharded_lora(SimpleNamespace(fully_sharded=sharded, use_ep=ep)) is expected
    assert not supports_tp_fully_sharded_lora(None)
    assert not supports_tp_fully_sharded_lora(SimpleNamespace(fully_sharded=True))


@pytest.mark.parametrize("sharded,ep,expected_mode", [(True, False, 1), (False, False, -1), (False, True, -1)])
def test_allgather_quantization_is_limited_to_tp_fully_sharded(sharded, ep, expected_mode):
    dispatcher = object.__new__(TokenDispatcherWithAllGather)
    dispatcher.top_k = 1
    dispatcher.num_experts_local = 2
    dispatcher.lora_context = SimpleNamespace(
        fully_sharded=sharded, use_ep=ep, punica_wrapper=SimpleNamespace(no_lora=False)
    )
    hidden = torch.ones(3, 8, dtype=torch.bfloat16)
    payload = MoETokenDispatchInput(
        hidden_states=hidden,
        topk_weights=torch.ones(3, 1),
        topk_ids=torch.zeros(3, 1, dtype=torch.int32),
        routing=MoERoutingParams(
            expert_map=None, global_redundant_expert_num=0, mc2_mask=None, apply_router_weight_on_input=False
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
    )
    scale = torch.ones(3)
    with patch.object(
        DeviceOperator, "npu_moe_init_routing", return_value=(hidden, torch.arange(3), torch.tensor([3, 0]), scale)
    ) as route:
        result = dispatcher.token_dispatch(payload)
    assert route.call_args.kwargs["quant_mode"] == expected_mode
    assert result.dynamic_scale is (scale if expected_mode == 1 else None)


@pytest.mark.parametrize("router_on_input", [False, True])
def test_prequantized_allgather_keeps_int8_input_and_adjusts_only_scale(router_on_input):
    dispatcher = object.__new__(TokenDispatcherWithAllGather)
    dispatcher.top_k = 1
    dispatcher.num_experts_local = 2
    dispatcher.lora_context = SimpleNamespace(
        fully_sharded=True, use_ep=False, punica_wrapper=SimpleNamespace(no_lora=False)
    )
    hidden = torch.arange(24, dtype=torch.int8).reshape(3, 8)
    scale = torch.tensor([[0.1], [0.2], [0.3]])
    topk_weights = torch.tensor([[0.5], [0.25], [0.75]])
    payload = MoETokenDispatchInput(
        hidden_states=hidden,
        topk_weights=topk_weights,
        topk_ids=torch.zeros(3, 1, dtype=torch.int32),
        routing=MoERoutingParams(
            expert_map=None,
            global_redundant_expert_num=0,
            mc2_mask=None,
            pertoken_scale=scale,
            apply_router_weight_on_input=router_on_input,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
    )
    with patch.object(
        DeviceOperator, "npu_moe_init_routing", return_value=(hidden, torch.arange(3), torch.tensor([3, 0]), scale)
    ) as route:
        dispatcher.token_dispatch(payload)
    assert route.call_args.args[0] is hidden
    assert route.call_args.kwargs["quant_mode"] == -1
    expected = scale.flatten() * topk_weights.flatten() if router_on_input else scale.flatten()
    torch.testing.assert_close(route.call_args.kwargs["scale"], expected)
