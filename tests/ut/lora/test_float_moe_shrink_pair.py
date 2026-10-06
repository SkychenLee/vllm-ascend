# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.ut.lora.test_punica_npu import _make_wrapper
from vllm_ascend.lora import punica_npu


@pytest.mark.parametrize(
    "change,allowed",
    [
        ({}, True),
        ({"rows": 384}, True),
        ({"rows": 385}, False),
        ({"rows": 0}, False),
        ({"hidden": 2048}, False),
        ({"rank": 8}, False),
        ({"dtype": torch.float16}, True),
        ({"dtype": torch.float32}, False),
        ({"weight_dtype": torch.float16}, False),
        ({"device": "cpu"}, False),
        ({"input_contiguous": False}, False),
        ({"weight_contiguous": False}, False),
        ({"index_contiguous": False}, False),
        ({"index_dtype": torch.int32}, False),
        ({"projections": 1}, False),
        ({"weight_mismatch": True}, False),
        ({"slots": 0}, False),
        ({"enabled": False}, False),
        ({"hardware": punica_npu.AscendDeviceType._310P}, False),
        ({"format": 29}, False),
        ({"format": 0}, True),
        ({"compiling": True, "format": 29}, True),
    ],
)
def test_float_pair_guard_requires_native_layout(change, allowed):
    options = dict(
        rows=48, hidden=4096, rank=16, slots=2, dtype=torch.bfloat16, device="npu", projections=2, enabled=True
    )
    options.update(change)
    device = SimpleNamespace(type=options["device"])

    def tensor(shape, dtype, contiguous=True):
        return SimpleNamespace(
            shape=shape, ndim=len(shape), dtype=dtype, device=device, is_contiguous=lambda: contiguous
        )

    x = tensor((options["rows"], options["hidden"]), options["dtype"], options.get("input_contiguous", True))
    shape = (options["slots"], 32, options["rank"], options["hidden"])
    weights = [
        tensor(shape, options.get("weight_dtype", options["dtype"]), options.get("weight_contiguous", True))
        for _ in range(options["projections"])
    ]
    if options.get("weight_mismatch"):
        weights[1].shape = (options["slots"], 31, options["rank"], options["hidden"])
    indices = tensor((options["rows"],), options.get("index_dtype", torch.int64), options.get("index_contiguous", True))
    with (
        patch.object(punica_npu.envs_ascend, "VLLM_ASCEND_MOE_LORA_FUSED_SHRINK_PAIR", options["enabled"]),
        patch.object(
            punica_npu, "get_ascend_device_type", return_value=options.get("hardware", punica_npu.AscendDeviceType.A3)
        ),
        patch("torch.ops._C_ascend.bgmv_shrink_pair", create=True),
        patch.object(punica_npu.torch_npu, "get_npu_format", return_value=options.get("format", 2), create=True) as fmt,
        patch("torch.compiler.is_compiling", return_value=options.get("compiling", False)),
    ):
        assert punica_npu._can_fuse_float_moe_shrink_pair(x, weights, indices) is allowed
        if options.get("compiling"):
            fmt.assert_not_called()


def test_float_pair_projections_keep_zero_inactive_rows_and_existing_b_loop():
    wrapper = _make_wrapper()
    x = torch.ones(3, 4096, dtype=torch.bfloat16)
    a = tuple(torch.ones(2, 2, 16, 4096, dtype=torch.bfloat16) for _ in range(2))
    b = tuple(torch.ones(2, 2, 32, 16, dtype=torch.bfloat16) for _ in range(2))
    combined = torch.tensor([0, -1, 3])

    def pair(inputs, a0, a1, indices, output, scale):
        assert inputs.data_ptr() == x.data_ptr() and indices is combined and scale == 1.0
        assert a0.shape == a1.shape == (4, 16, 4096)
        assert output.shape == (2, 3, 16) and output.dtype == torch.float32
        assert torch.count_nonzero(output) == 0
        output[0, indices >= 0, :] = 2.0
        output[1, indices >= 0, :] = 3.0

    with (
        patch.object(punica_npu, "_can_fuse_float_moe_shrink_pair", return_value=True),
        patch("torch.ops._C_ascend.bgmv_shrink_pair", side_effect=pair, create=True) as paired,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(3, 64, dtype=torch.bfloat16),
            x=x,
            lora_a_stacked=a,
            lora_b_stacked=b,
            expert_ids=torch.tensor([0, -1, 1]),
            adapter_enabled=torch.ones(2, dtype=torch.bool),
            token_lora_mapping=torch.tensor([0, -1, 1]),
            combined_indices=combined,
        )
    paired.assert_called_once()
    wrapper.bgmv_shrink.assert_not_called()
    assert wrapper.bgmv_expand_slice.call_count == 2
    gate, up = wrapper.bgmv_expand_slice.call_args_list
    torch.testing.assert_close(gate.args[0], torch.tensor([[2.0] * 16, [0.0] * 16, [2.0] * 16]))
    torch.testing.assert_close(up.args[0], torch.tensor([[3.0] * 16, [0.0] * 16, [3.0] * 16]))
    assert gate.args[3] is combined and up.args[3] is combined
    assert gate.args[4:6] == (0, 32) and up.args[4:6] == (32, 32)


@pytest.mark.parametrize("fully_sharded,input_scale", [(True, None), (False, torch.ones(3))])
def test_float_pair_does_not_replace_sharded_or_quantized_a(fully_sharded, input_scale):
    wrapper = _make_wrapper()
    a = tuple(torch.ones(2, 2, 16, 4096, dtype=torch.bfloat16) for _ in range(2))
    b = tuple(torch.ones(2, 2, 32, 16, dtype=torch.bfloat16) for _ in range(2))
    with (
        patch.object(punica_npu, "_can_fuse_float_moe_shrink_pair", return_value=True) as guard,
        patch("torch.ops._C_ascend.bgmv_shrink_pair", create=True) as paired,
        patch("torch.ops._C_ascend.bgmv_shrink_int8", create=True) as int8_shrink,
        patch.object(punica_npu, "tensor_model_parallel_all_reduce", side_effect=lambda t: t),
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(3, 64, dtype=torch.bfloat16),
            x=torch.ones(3, 4096, dtype=torch.bfloat16),
            lora_a_stacked=a,
            lora_b_stacked=b,
            expert_ids=torch.tensor([0, 0, 1]),
            adapter_enabled=torch.ones(2, dtype=torch.bool),
            combined_indices=torch.tensor([0, -1, 3]),
            fully_sharded=fully_sharded,
            input_scale=input_scale,
        )
    guard.assert_not_called()
    paired.assert_not_called()
    assert wrapper.bgmv_shrink.call_count == (2 if fully_sharded else 0)
    assert int8_shrink.call_count == (0 if fully_sharded else 2)


def test_existing_cube_pair_keeps_priority():
    wrapper = _make_wrapper()
    a = tuple(torch.ones(2, 2, 16, 4096, dtype=torch.bfloat16) for _ in range(2))
    b = tuple(torch.ones(2, 2, 32, 16, dtype=torch.bfloat16) for _ in range(2))
    cube = Mock()
    cube.shrink_pair.return_value = (torch.ones(3, 16), torch.ones(3, 16))
    with (
        patch.object(punica_npu, "_can_fuse_float_moe_shrink_pair", return_value=True) as guard,
        patch("torch.ops._C_ascend.bgmv_shrink_pair", create=True) as paired,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(3, 64, dtype=torch.bfloat16),
            x=torch.ones(3, 4096, dtype=torch.bfloat16),
            lora_a_stacked=a,
            lora_b_stacked=b,
            expert_ids=torch.tensor([0, 0, 1]),
            adapter_enabled=torch.ones(2, dtype=torch.bool),
            combined_indices=torch.tensor([0, -1, 3]),
            cube_routing=object(),
            cube_state=cube,
        )
    cube.shrink_pair.assert_called_once()
    guard.assert_not_called()
    paired.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
