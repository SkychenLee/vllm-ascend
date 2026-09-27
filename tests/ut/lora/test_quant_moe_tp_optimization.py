# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.ut.lora.test_quant_moe import QUANT_MOE
from tests.ut.lora.test_quant_moe import _make_input as _legacy_make_input
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.quant_moe import (
    _can_fuse_int8_lora_swiglu,
    quant_apply_mlp_with_moe_lora,
    validate_quant_moe_lora_activation_input,
)
from vllm_ascend.quantization.quant_type import QuantType


def _make_input(**kwargs):
    kwargs.setdefault("lora_context", SimpleNamespace(use_ep=False, fully_sharded=True))
    return _legacy_make_input(**kwargs)


def test_dynamic_int8_allgather_lora_handles_empty_int8_input() -> None:
    context = SimpleNamespace(use_ep=False, fully_sharded=True, split_lora_indices=torch.empty(0, dtype=torch.long))
    payload = _make_input(
        hidden_states=torch.empty(0, 4, dtype=torch.int8),
        dynamic_scale=torch.empty(0),
        output_dtype=torch.bfloat16,
        lora_context=context,
    )
    with patch(f"{QUANT_MOE}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)):
        output, event = quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
    assert output.shape == (0, 4) and output.dtype == torch.bfloat16
    assert event is None and not hasattr(context, "split_lora_indices")


@pytest.mark.parametrize("scale", [None, torch.ones(3), torch.ones(2, dtype=torch.float16)])
def test_allgather_rejects_invalid_int8_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        validate_quant_moe_lora_activation_input(
            quant_type=QuantType.W8A8,
            hidden_states=torch.ones(2, 4, dtype=torch.int8),
            dynamic_scale=scale,
            comm_type=MoECommType.ALLGATHER,
            lora_context=SimpleNamespace(use_ep=False, fully_sharded=True),
        )


def test_allgather_int8_fuses_w13_and_reuses_quantized_activations():
    x = torch.ones(2, 4, dtype=torch.int8)
    scale = torch.tensor([0.1, 0.2])
    activated = torch.ones(2, 32, dtype=torch.int8)
    activated_scale = torch.tensor([0.3, 0.4])
    combined_indices = torch.tensor([0, 3])
    mlp_input = _make_input(
        hidden_states=x,
        dynamic_scale=scale,
        output_dtype=torch.bfloat16,
        fusion=True,
        lora_context=SimpleNamespace(
            use_ep=False,
            fully_sharded=True,
            w13_lora_a_stacked=[None, None],
            w13_lora_b_stacked=[torch.empty(1, 1, 32, 16)] * 2,
        ),
    )
    routing = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    event = object()
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as context,
        patch.object(DeviceOperator, "npu_dynamic_quant") as quant,
        patch(
            f"{QUANT_MOE}.torch_npu.npu_grouped_matmul",
            return_value=[torch.zeros(2, 64, dtype=torch.bfloat16)],
            create=True,
        ) as gmm1,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather", return_value=routing),
        patch(
            f"{QUANT_MOE}.moe_lora_apply_w13_swiglu_quant",
            return_value=(activated, activated_scale, combined_indices),
        ) as fused,
        patch(f"{QUANT_MOE}._apply_moe_activation") as activation,
        patch.object(
            DeviceOperator, "npu_grouped_matmul_gmm2", return_value=torch.zeros(2, 4, dtype=torch.bfloat16)
        ) as gmm2,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as lora2,
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=Mock(record_event=Mock(return_value=event))),
    ):
        context.moe_comm_type = MoECommType.ALLGATHER
        _, returned_event = quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)
    quant.assert_not_called()
    activation.assert_not_called()
    assert gmm1.call_args.kwargs["x"][0] is x
    assert fused.call_args.kwargs["hidden_states"] is x
    assert fused.call_args.kwargs["input_scale"].data_ptr() == scale.data_ptr()
    assert gmm2.call_args.kwargs["hidden_states"] is activated
    assert lora2.call_args.kwargs["silu_out"] is activated
    assert lora2.call_args.kwargs["input_scale"] is activated_scale
    assert lora2.call_args.kwargs["combined_indices"] is combined_indices
    assert returned_event is event


@pytest.mark.parametrize(
    "width,rank,expected",
    [
        (256, 16, True),
        (256, 32, True),
        (128, 64, True),
        (512, 16, False),
        (1024, 64, False),
        (256, 128, False),
        (255, 16, False),
        (256, 3, False),
        (8193, 16, False),
    ],
)
def test_fused_dispatch_keeps_measured_regressions_on_separate_path(width, rank, expected):
    context = SimpleNamespace(
        w13_lora_a_stacked=[None, None],
        w13_lora_b_stacked=[torch.empty(1, 1, width, rank, device="meta")] * 2,
    )
    assert _can_fuse_int8_lora_swiglu(context, width) is expected


def test_allgather_int8_requires_float_output_dtype():
    mlp_input = _make_input(hidden_states=torch.ones(2, 4, dtype=torch.int8), dynamic_scale=torch.ones(2))
    with patch(f"{QUANT_MOE}._EXTRA_CTX") as context:
        context.moe_comm_type = MoECommType.ALLGATHER
        with pytest.raises(ValueError, match="output_dtype"):
            quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)


@pytest.mark.parametrize("fully_sharded,use_ep", [(False, False), (False, True), (True, True)])
def test_other_parallel_modes_retain_float_activation_contract(fully_sharded, use_ep):
    context = SimpleNamespace(fully_sharded=fully_sharded, use_ep=use_ep)
    with pytest.raises(NotImplementedError, match="unquantized"):
        validate_quant_moe_lora_activation_input(
            quant_type=QuantType.W8A8,
            hidden_states=torch.ones(2, 4, dtype=torch.int8),
            dynamic_scale=torch.ones(2),
            comm_type=MoECommType.ALLGATHER,
            lora_context=context,
        )


@pytest.mark.parametrize("scale", [torch.ones(2, 1), torch.ones(2)])
def test_tp_allgather_accepts_each_per_token_scale_layout(scale):
    validate_quant_moe_lora_activation_input(
        quant_type=QuantType.W8A8,
        hidden_states=torch.ones(2, 4, dtype=torch.int8),
        dynamic_scale=scale,
        comm_type=MoECommType.ALLGATHER,
        lora_context=SimpleNamespace(fully_sharded=True, use_ep=False),
    )


def test_tp_allgather_rejects_scale_on_different_device():
    with pytest.raises(ValueError, match="device"):
        validate_quant_moe_lora_activation_input(
            quant_type=QuantType.W8A8,
            hidden_states=torch.ones(2, 4, dtype=torch.int8),
            dynamic_scale=torch.ones(2, device="meta"),
            comm_type=MoECommType.ALLGATHER,
            lora_context=SimpleNamespace(fully_sharded=True, use_ep=False),
        )
