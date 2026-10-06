# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.ut.lora.test_punica_npu import _make_wrapper
from tests.ut.lora.test_quant_moe import QUANT_MOE, _make_input
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoELoRARouting


@pytest.mark.parametrize(
    "change,allowed",
    [
        ({}, True),
        ({"rows": 48}, True),
        ({"rows": 49}, True),
        ({"rows": 96}, True),
        ({"rows": 192}, True),
        ({"rows": 288}, True),
        ({"rows": 289}, False),
        ({"rows": 384}, False),
        ({"rows": 385}, False),
        ({"rows": 49, "limit": 10.0}, True),
        ({"rows": 96, "limit": 10.0}, True),
        ({"rows": 192, "limit": 10.0}, True),
        ({"rows": 288, "limit": 10.0}, True),
        ({"rows": 384, "limit": 10.0}, True),
        ({"rows": 385, "limit": 10.0}, False),
        ({"rows": 288, "limit": 10.0, "dtype": torch.float16}, True),
        ({"rows": 288, "limit": 0.0}, True),
        ({"rows": 288, "limit": float("inf")}, False),
        ({"rows": 288, "limit": 65505}, False),
        ({"rows": 288, "limit": 10.0, "activation": "swigluoai_uninterleave"}, False),
        ({"rows": 288, "limit": 10.0, "topk_scales": torch.ones(1)}, False),
        ({"rows": 288, "limit": 10.0, "format": 29}, False),
        ({"rows": 0}, False),
        ({"width": 2048}, False),
        ({"dtype": torch.float32}, False),
        ({"device": "cpu"}, False),
        ({"activation": "swigluoai_uninterleave"}, False),
        ({"activation": "gelu"}, False),
        ({"fully_sharded": True}, False),
        ({"use_ep": False}, False),
        ({"limit": float("nan")}, False),
        ({"limit": -1}, False),
        ({"topk_scales": torch.ones(1)}, False),
        ({"contiguous": False}, False),
        ({"format": 29}, False),
    ],
)
def test_activation_fusion_only_accepts_supported_contract(change, allowed):
    from vllm_ascend.lora import activation

    options = dict(rows=48, width=4096, dtype=torch.bfloat16, device="npu", contiguous=True)
    options.update(change)
    gate = SimpleNamespace(
        device=SimpleNamespace(type=options["device"]),
        dtype=options["dtype"],
        ndim=2,
        shape=(options["rows"], options["width"]),
        is_contiguous=lambda: options["contiguous"],
    )
    payload = _make_input(
        activation=options.get("activation", "silu"),
        swiglu_limit=options.get("limit", 0.0),
        topk_scales=options.get("topk_scales"),
        lora_context=SimpleNamespace(
            use_ep=options.get("use_ep", True), fully_sharded=options.get("fully_sharded", False)
        ),
    )
    with (
        patch.object(activation.envs, "VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION", True),
        patch("torch.ops._C_ascend.moe_lora_activation_quant", create=True),
        patch.object(activation.torch_npu, "get_npu_format", return_value=options.get("format", 2)),
    ):
        assert activation.can_fuse_moe_lora_activation(gate, payload, MoECommType.ALLGATHER) is allowed
        assert not activation.can_fuse_moe_lora_activation(gate, payload, MoECommType.ALLTOALL)


def test_fused_activation_keeps_float_w2_lora_and_int8_base_w2():
    payload = _make_input(lora_context=SimpleNamespace(use_ep=True), swiglu_limit=10.0)
    floating = torch.randn(2, 3, dtype=torch.bfloat16)
    quantized = torch.ones(2, 3, dtype=torch.int8)
    scale = torch.ones(2)
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant", return_value=(torch.ones(2, 4), torch.ones(2))) as quant,
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[torch.zeros(2, 6)]),
        patch(f"{QUANT_MOE}._apply_moe_activation") as unfused,
        patch(f"{QUANT_MOE}.can_fuse_moe_lora_activation", return_value=True),
        patch("torch.ops._C_ascend.moe_lora_activation_quant", return_value=(floating, quantized, scale), create=True),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_grouped_matmul_gmm2", return_value=torch.zeros(2, 4)) as gmm2,
        patch(
            f"{QUANT_MOE}._recover_moe_lora_routing_allgather",
            return_value=(torch.tensor([0, 1]), torch.tensor([0, 1])),
        ),
        patch(f"{QUANT_MOE}.moe_lora_apply_w13"),
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as lora2,
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=Mock()),
    ):
        from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora

        quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
    assert quant.call_count == 1
    unfused.assert_not_called()
    assert gmm2.call_args.kwargs["hidden_states"] is quantized
    assert gmm2.call_args.kwargs["per_token_scale"] is scale
    assert lora2.call_args.kwargs["silu_out"] is floating


def test_ep_w13_w2_share_indices_and_refresh_for_each_forward():
    context = SimpleNamespace(
        use_ep=True,
        fully_sharded=False,
        w13_lora_a_stacked=[torch.empty(2, 2, 16, 4)] * 2,
        w2_lora_a_stacked=[torch.empty(2, 2, 16, 3)],
        adapter_enabled=torch.tensor([1, 0], dtype=torch.int32),
    )
    routing = MoELoRARouting(torch.tensor([0, 1]), torch.tensor([0, 1]))
    payload = _make_input(lora_context=context, lora_routing=routing)
    expected = [torch.tensor([0, -1]), torch.tensor([-1, 1])]
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant", return_value=(torch.ones(2, 4), torch.ones(2))),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[torch.zeros(2, 6)]),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=torch.zeros(2, 3)),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_grouped_matmul_gmm2", return_value=torch.zeros(2, 4)),
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as w2,
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=Mock()),
    ):
        from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora

        for index in range(2):
            quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
            first = w13.call_args.kwargs["combined_indices"]
            second = w2.call_args.kwargs["combined_indices"]
            assert first is second
            torch.testing.assert_close(first, expected[index], rtol=0, atol=0)
            routing.adapter_slots.copy_(torch.tensor([1, 0]))
        assert w13.call_args_list[0].kwargs["combined_indices"] is not first


def test_float_punica_consumes_shared_indices_without_rebuilding():
    wrapper = _make_wrapper()
    combined = torch.tensor([0, -1, 3])
    with patch("vllm_ascend.lora.punica_npu.combined_moe_lora_indices") as build_indices:
        wrapper.add_lora_fused_moe(
            y=torch.zeros(3, 16, dtype=torch.bfloat16),
            x=torch.ones(3, 8, dtype=torch.bfloat16),
            lora_a_stacked=[torch.ones(2, 2, 16, 8, dtype=torch.bfloat16)],
            lora_b_stacked=[torch.ones(2, 2, 16, 16, dtype=torch.bfloat16)],
            expert_ids=torch.tensor([0, 1, 1]),
            adapter_enabled=torch.ones(2, dtype=torch.int32),
            token_lora_mapping=torch.tensor([0, -1, 1]),
            combined_indices=combined,
        )
    build_indices.assert_not_called()
    assert wrapper.bgmv_shrink.call_args.args[3] is combined
    assert wrapper.bgmv_expand_slice.call_args.args[3] is combined
