from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoEWithLoRA,
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    has_lora,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def test_ascend_fused_moe_lora_initializes_skipped_upstream_fields() -> None:
    parallel_config = SimpleNamespace(tp_size=8, tp_rank=3, ep_rank=0, use_ep=False)
    shared_experts = torch.nn.Module()
    base_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            hidden_dim=4096,
            num_local_experts=256,
            num_experts=256,
            intermediate_size_per_partition=256,
            experts_per_token=8,
            moe_parallel_config=parallel_config,
            is_act_and_mul=True,
        ),
        _shared_experts=shared_experts,
    )

    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._lora_stream is None
    assert wrapper._events is None
    assert wrapper.enable_moe_shared_loras is False
    assert wrapper._shared_experts is shared_experts
    assert wrapper.n_slices == 256 * 3


def test_moe_lora_apply_uses_adapter_enabled() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="all_enabled",
        fully_sharded=False,
        tp_rank=0,
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["adapter_enabled"] == "all_enabled"
    assert calls[1].kwargs["adapter_enabled"] == "all_enabled"
    assert calls[0].kwargs["fully_sharded"] is False
    assert calls[1].kwargs["fully_sharded"] is False
    assert calls[1].kwargs["offset"] == 0


def test_moe_lora_apply_propagates_fully_sharded_metadata() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked=(torch.empty(1, 1, 16, 8),),
        adapter_enabled="all_enabled",
        fully_sharded=True,
        tp_rank=3,
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["fully_sharded"] is True
    assert calls[1].kwargs["fully_sharded"] is True
    assert calls[1].kwargs["offset"] == 48


def test_punica_fully_sharded_moe_gathers_rank_shards() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)

    def shrink(_, __, output, ___, ____):
        output.copy_(torch.arange(16, dtype=torch.float32).view(2, 8))

    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    lora_a = (torch.zeros(2, 2, 8, 32, dtype=torch.float16),)
    lora_b = (torch.zeros(2, 2, 16, 16, dtype=torch.float16),)

    with (
        patch(
            "vllm_ascend.lora.punica_npu.get_tensor_model_parallel_world_size",
            return_value=2,
        ),
        patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather",
            side_effect=lambda value: torch.cat((value, value + 10), dim=-1),
        ) as all_gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce") as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 16, dtype=torch.float16),
            x=torch.zeros(2, 32, dtype=torch.float16),
            lora_a_stacked=lora_a,
            lora_b_stacked=lora_b,
            expert_ids=torch.tensor([0, 1]),
            adapter_enabled=torch.tensor([1, 1]),
            fully_sharded=True,
            token_lora_mapping=torch.tensor([0, 1]),
        )

    all_gather.assert_called_once()
    all_reduce.assert_not_called()
    wrapper.moe_lora_bgmv_fused.assert_not_called()
    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(
        expand_args[0],
        torch.cat(
            (
                torch.arange(16, dtype=torch.float32).view(2, 8),
                torch.arange(16, dtype=torch.float32).view(2, 8) + 10,
            ),
            dim=-1,
        ),
    )
    assert expand_args[1].shape == (4, 16, 16)
    assert torch.equal(expand_args[3], torch.tensor([0, 3]))


def test_punica_fully_sharded_moe_reduces_partial_rank() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)

    def shrink(_, __, output, ___, ____):
        output.copy_(torch.arange(16, dtype=torch.float32).view(2, 8))

    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    lora_a = (torch.zeros(2, 2, 8, 16, dtype=torch.float16),)
    lora_b = (torch.zeros(2, 2, 8, 8, dtype=torch.float16),)

    with (
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather") as all_gather,
        patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value + 10,
        ) as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 13, dtype=torch.float16),
            x=torch.zeros(2, 16, dtype=torch.float16),
            lora_a_stacked=lora_a,
            lora_b_stacked=lora_b,
            expert_ids=torch.tensor([0, 1]),
            adapter_enabled=torch.tensor([1, 1]),
            fully_sharded=True,
            offset=5,
            token_lora_mapping=torch.tensor([0, 1]),
        )

    all_gather.assert_not_called()
    all_reduce.assert_called_once()
    wrapper.moe_lora_bgmv_fused.assert_not_called()
    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(
        expand_args[0],
        torch.arange(16, dtype=torch.float32).view(2, 8) + 10,
    )
    assert expand_args[1].shape == (4, 8, 8)
    assert expand_args[4] == 5


@pytest.mark.parametrize(
    ("rows", "input_hidden", "output_hidden", "rank", "dtype"),
    [
        (6, 4096, 256, 16, torch.float16),
        (6, 256, 4096, 16, torch.bfloat16),
        (1, 4097, 256, 8, torch.float16),
        (2, 256, 4097, 32, torch.bfloat16),
        (3, 16384, 1, 64, torch.float16),
        (2, 8, 7, 8, torch.float16),
    ],
)
def test_punica_uses_fused_bgmv_for_supported_contract(
    rows: int,
    input_hidden: int,
    output_hidden: int,
    rank: int,
    dtype: torch.dtype,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()
    wrapper.add_lora_fused_moe(
        y=torch.zeros(rows, output_hidden, dtype=dtype),
        x=torch.zeros(rows, input_hidden, dtype=dtype),
        lora_a_stacked=(
            torch.zeros(1, 1, rank, input_hidden, dtype=dtype),
        ),
        lora_b_stacked=(
            torch.zeros(1, 1, output_hidden, rank, dtype=dtype),
        ),
        adapter_enabled=torch.tensor([1]),
        combined_indices=torch.zeros(rows, dtype=torch.int32),
    )

    wrapper.moe_lora_bgmv_fused.assert_called_once()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("input_hidden", "output_hidden", "rank", "dtype", "fused_available"),
    [
        (16385, 256, 16, torch.float16, True),
        (256, 16385, 16, torch.bfloat16, True),
        (256, 256, 16, torch.float16, False),
    ],
)
def test_punica_falls_back_outside_fused_bgmv_contract(
    input_hidden: int,
    output_hidden: int,
    rank: int,
    dtype: torch.dtype,
    fused_available: bool,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    fused_op = Mock() if fused_available else None
    wrapper.moe_lora_bgmv_fused = fused_op
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    wrapper.add_lora_fused_moe(
        y=torch.zeros(1, output_hidden, dtype=dtype),
        x=torch.zeros(1, input_hidden, dtype=dtype),
        lora_a_stacked=(
            torch.zeros(1, 1, rank, input_hidden, dtype=dtype),
        ),
        lora_b_stacked=(
            torch.zeros(1, 1, output_hidden, rank, dtype=dtype),
        ),
        adapter_enabled=torch.tensor([1]),
        combined_indices=torch.zeros(1, dtype=torch.int32),
    )

    if fused_op is not None:
        fused_op.assert_not_called()
    wrapper.bgmv_shrink.assert_called_once()
    wrapper.bgmv_expand_slice.assert_called_once()


@pytest.mark.parametrize(
    ("rank", "dtype", "error_type", "error_match"),
    [
        (4, torch.float16, ValueError, "B rank must be one of"),
        (128, torch.float16, ValueError, "B rank must be one of"),
        (16, torch.float32, TypeError, "same float16 or bfloat16 dtype"),
    ],
)
def test_punica_native_rejects_unsupported_direct_bgmv_contract(
    rank: int,
    dtype: torch.dtype,
    error_type: type[Exception],
    error_match: str,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(error_type, match=error_match):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 256, dtype=dtype),
            x=torch.zeros(2, 256, dtype=dtype),
            lora_a_stacked=(torch.zeros(1, 1, rank, 256, dtype=dtype),),
            lora_b_stacked=(torch.zeros(1, 1, 256, rank, dtype=dtype),),
            adapter_enabled=torch.tensor([1]),
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


def test_punica_native_rejects_mismatched_direct_bgmv_dtypes_before_launch() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(TypeError, match="must have the same float16 or bfloat16 dtype"):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 5, dtype=torch.float16),
            x=torch.zeros(2, 3, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 3, dtype=torch.bfloat16),),
            lora_b_stacked=(torch.zeros(1, 1, 5, 8, dtype=torch.float16),),
            adapter_enabled=torch.tensor([1]),
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("rank", "dtype"),
    [
        (4, torch.float16),
        (128, torch.float16),
        (16, torch.float32),
    ],
)
def test_punica_torch_bgmv_keeps_generic_rank_and_dtype_contract(
    rank: int,
    dtype: torch.dtype,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._bgmv_uses_torch_ops = True
    wrapper.moe_lora_bgmv_fused = None
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()
    hidden = max(rank + 1, 256)
    output = max(rank, 256)

    wrapper.add_lora_fused_moe(
        y=torch.zeros(2, output, dtype=dtype),
        x=torch.zeros(2, hidden, dtype=dtype),
        lora_a_stacked=(torch.zeros(1, 1, rank, hidden, dtype=dtype),),
        lora_b_stacked=(torch.zeros(1, 1, output, rank, dtype=dtype),),
        adapter_enabled=torch.tensor([1]),
        combined_indices=torch.zeros(2, dtype=torch.int32),
    )

    wrapper.bgmv_shrink.assert_called_once()
    wrapper.bgmv_expand_slice.assert_called_once()


def test_punica_torch_bgmv_allows_weight_dtype_conversion() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._bgmv_uses_torch_ops = True
    wrapper.moe_lora_bgmv_fused = None
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    wrapper.add_lora_fused_moe(
        y=torch.zeros(2, 5, dtype=torch.float32),
        x=torch.zeros(2, 3, dtype=torch.float32),
        lora_a_stacked=(torch.zeros(1, 1, 8, 3, dtype=torch.bfloat16),),
        lora_b_stacked=(torch.zeros(1, 1, 5, 8, dtype=torch.float16),),
        adapter_enabled=torch.tensor([1]),
        combined_indices=torch.zeros(2, dtype=torch.int64),
    )

    wrapper.bgmv_shrink.assert_called_once()
    wrapper.bgmv_expand_slice.assert_called_once()


def test_punica_torch_bgmv_masks_disabled_indices() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._bgmv_uses_torch_ops = True
    wrapper.moe_lora_bgmv_fused = None

    def shrink(_, __, output, indices, ___):
        assert torch.equal(indices, torch.tensor([0, 0, 0], dtype=torch.int32))
        output.fill_(3.0)

    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()

    wrapper.add_lora_fused_moe(
        y=torch.zeros(3, 4, dtype=torch.float32),
        x=torch.zeros(3, 16, dtype=torch.float32),
        lora_a_stacked=(torch.zeros(1, 1, 4, 16, dtype=torch.float16),),
        lora_b_stacked=(torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16),),
        adapter_enabled=torch.tensor([1]),
        combined_indices=torch.tensor([-1, 7, 0], dtype=torch.int32),
    )

    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(expand_args[0][0], torch.zeros(4, dtype=torch.float32))
    assert torch.equal(expand_args[0][1], torch.zeros(4, dtype=torch.float32))
    assert torch.equal(expand_args[0][2], torch.full((4,), 3.0, dtype=torch.float32))
    assert torch.equal(expand_args[3], torch.tensor([0, 0, 0], dtype=torch.int32))


@pytest.mark.parametrize(
    ("expert_ids", "token_lora_mapping", "error_match"),
    [
        (torch.tensor([0.0, 1.0]), torch.tensor([0, 0]), "expert_ids must be int32 or int64"),
        (torch.tensor([0, 1]), torch.tensor([0.0, 0.0]), "token_lora_mapping must be int32 or int64"),
    ],
)
def test_punica_rejects_non_integer_implicit_routing_before_launch(
    expert_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    error_match: str,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(TypeError, match=error_match):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 8, dtype=torch.float16),
            x=torch.zeros(2, 16, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 2, 8, 16, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 2, 8, 8, dtype=torch.float16),),
            expert_ids=expert_ids,
            adapter_enabled=torch.tensor([True]),
            token_lora_mapping=token_lora_mapping,
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


def test_punica_implicit_routing_disables_out_of_range_metadata() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    wrapper.add_lora_fused_moe(
        y=torch.zeros(2, 8, dtype=torch.float16),
        x=torch.zeros(2, 16, dtype=torch.float16),
        lora_a_stacked=(torch.zeros(1, 2, 8, 16, dtype=torch.float16),),
        lora_b_stacked=(torch.zeros(1, 2, 8, 8, dtype=torch.float16),),
        expert_ids=torch.tensor([0, 2]),
        adapter_enabled=torch.tensor([True]),
        token_lora_mapping=torch.tensor([0, 7]),
    )

    fused_indices = wrapper.moe_lora_bgmv_fused.call_args.args[3]
    assert torch.equal(fused_indices, torch.tensor([0, -1]))
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("fully_sharded", "mul_routed_weight", "fused_available"),
    [
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_punica_zero_rows_never_launches_moe_lora_kernels(
    fully_sharded: bool,
    mul_routed_weight: bool,
    fused_available: bool,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    fused_op = Mock() if fused_available else None
    wrapper.moe_lora_bgmv_fused = fused_op
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with (
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather") as all_gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce") as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(0, 16, dtype=torch.float16),
            x=torch.zeros(0, 16, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 16, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 16, 8, dtype=torch.float16),),
            topk_weights=torch.empty(0) if mul_routed_weight else None,
            adapter_enabled=torch.tensor([1]),
            fully_sharded=fully_sharded,
            mul_routed_weight=mul_routed_weight,
            combined_indices=torch.zeros(0, dtype=torch.int32),
        )

    if fused_op is not None:
        fused_op.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()
    all_gather.assert_not_called()
    all_reduce.assert_not_called()


def test_punica_zero_rows_with_implicit_routing_never_launches_moe_lora_kernels() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with (
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather") as all_gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce") as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(0, 16, dtype=torch.float16),
            x=torch.zeros(0, 16, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 16, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 16, 8, dtype=torch.float16),),
            expert_ids=torch.empty(0, dtype=torch.int64),
            adapter_enabled=torch.tensor([1]),
            token_lora_mapping=torch.empty(0, dtype=torch.int64),
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()
    all_gather.assert_not_called()
    all_reduce.assert_not_called()


def test_punica_mul_routed_weight_uses_split_bgmv() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)

    def shrink(_, __, output, ___, ____):
        output.copy_(torch.arange(16, dtype=torch.float32).view(2, 8))

    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    topk_weights = torch.tensor([0.5, 99.0, -2.0, 99.0], dtype=torch.float64)[::2]

    wrapper.add_lora_fused_moe(
        y=torch.zeros(2, 8, dtype=torch.float16),
        x=torch.zeros(2, 16, dtype=torch.float16),
        lora_a_stacked=(torch.zeros(1, 1, 8, 16, dtype=torch.float16),),
        lora_b_stacked=(torch.zeros(1, 1, 8, 8, dtype=torch.float16),),
        topk_weights=topk_weights,
        adapter_enabled=torch.tensor([1]),
        mul_routed_weight=True,
        combined_indices=torch.zeros(2, dtype=torch.int32),
    )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_called_once()
    wrapper.bgmv_expand_slice.assert_called_once()
    delta = wrapper.bgmv_expand_slice.call_args.args[0]
    expected_delta = torch.arange(16, dtype=torch.float32).view(2, 8) * topk_weights.float().view(-1, 1)
    assert delta.dtype == torch.float32
    assert delta.is_contiguous()
    assert torch.equal(delta, expected_delta)


def test_punica_mul_routed_weight_requires_topk_weights_before_launch() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(ValueError, match="topk_weights are required"):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 5, dtype=torch.float16),
            x=torch.zeros(2, 16, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 16, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 5, 8, dtype=torch.float16),),
            adapter_enabled=torch.tensor([1]),
            mul_routed_weight=True,
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("topk_weights", "error_match"),
    [
        (torch.ones(1), "topk_weights size must match"),
        (torch.ones(2, device="meta"), "topk_weights must be on the same device"),
    ],
)
def test_punica_mul_routed_weight_rejects_invalid_metadata_before_launch(
    topk_weights: torch.Tensor,
    error_match: str,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(ValueError, match=error_match):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 8, dtype=torch.float16),
            x=torch.zeros(2, 16, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 16, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 8, 8, dtype=torch.float16),),
            topk_weights=topk_weights,
            adapter_enabled=torch.tensor([1]),
            mul_routed_weight=True,
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool, torch.complex64])
def test_punica_mul_routed_weight_rejects_non_floating_dtype_before_launch(
    dtype: torch.dtype,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(TypeError, match="topk_weights must have a floating-point dtype"):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 8, dtype=torch.float16),
            x=torch.zeros(2, 16, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 16, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 8, 8, dtype=torch.float16),),
            topk_weights=torch.ones(2, dtype=dtype),
            adapter_enabled=torch.tensor([1]),
            mul_routed_weight=True,
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


def test_punica_zero_rows_still_validates_static_metadata() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(ValueError, match="x and y row counts must match"):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(1, 8, dtype=torch.float16),
            x=torch.zeros(0, 16, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 16, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 8, 8, dtype=torch.float16),),
            adapter_enabled=torch.tensor([1]),
            combined_indices=torch.zeros(0, dtype=torch.int32),
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("input_hidden", "output_hidden", "error_match"),
    [
        (8, 16, "input dimension must be greater"),
        (16, 7, "output dimension must be at least"),
    ],
)
def test_punica_native_split_rejects_unsupported_dimension_relation_before_launch(
    input_hidden: int,
    output_hidden: int,
    error_match: str,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = None
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(ValueError, match=error_match):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, output_hidden, dtype=torch.float16),
            x=torch.zeros(2, input_hidden, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, input_hidden, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, output_hidden, 8, dtype=torch.float16),),
            adapter_enabled=torch.tensor([1]),
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("local_rank", "full_rank"),
    [
        (32, 16),
        (6, 16),
    ],
)
def test_punica_fully_sharded_rejects_impossible_rank_relation_before_launch(
    local_rank: int,
    full_rank: int,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with (
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather") as all_gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce") as all_reduce,
        pytest.raises(ValueError, match="must not exceed and must divide"),
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 32, dtype=torch.float16),
            x=torch.zeros(2, 64, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, local_rank, 64, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 32, full_rank, dtype=torch.float16),),
            adapter_enabled=torch.tensor([1]),
            fully_sharded=True,
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()
    all_gather.assert_not_called()
    all_reduce.assert_not_called()


def test_punica_fully_sharded_rejects_tp_world_size_mismatch_before_launch() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with (
        patch(
            "vllm_ascend.lora.punica_npu.get_tensor_model_parallel_world_size",
            return_value=2,
        ),
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather") as all_gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce") as all_reduce,
        pytest.raises(ValueError, match="does not reconstruct B rank 32"),
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 32, dtype=torch.float16),
            x=torch.zeros(2, 64, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 64, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 32, 32, dtype=torch.float16),),
            adapter_enabled=torch.tensor([1]),
            fully_sharded=True,
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()
    all_gather.assert_not_called()
    all_reduce.assert_not_called()


def test_punica_rank_mismatch_preserves_split_error() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(ValueError, match="A projection has rank 8, but LoRA B expects rank 16"):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 5, dtype=torch.float16),
            x=torch.zeros(2, 3, dtype=torch.float16),
            lora_a_stacked=(torch.zeros(1, 1, 8, 3, dtype=torch.float16),),
            lora_b_stacked=(torch.zeros(1, 1, 5, 16, dtype=torch.float16),),
            adapter_enabled=torch.tensor([1]),
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


def test_punica_prevalidates_all_slices_before_launch() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()

    with pytest.raises(TypeError, match="slice 1 does not"):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 16, dtype=torch.float16),
            x=torch.zeros(2, 16, dtype=torch.float16),
            lora_a_stacked=(
                torch.zeros(1, 1, 8, 16, dtype=torch.float16),
                torch.zeros(1, 1, 8, 16, dtype=torch.bfloat16),
            ),
            lora_b_stacked=(
                torch.zeros(1, 1, 8, 8, dtype=torch.float16),
                torch.zeros(1, 1, 8, 8, dtype=torch.bfloat16),
            ),
            adapter_enabled=torch.tensor([1]),
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


def test_punica_prevalidates_native_layout_for_all_slices_before_launch() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.moe_lora_bgmv_fused = Mock()
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()
    noncontiguous_a = torch.zeros(1, 1, 8, 32, dtype=torch.float16)[..., ::2]
    assert not noncontiguous_a.is_contiguous()

    with pytest.raises(ValueError, match="slice 1 must be contiguous"):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 16, dtype=torch.float16),
            x=torch.zeros(2, 16, dtype=torch.float16),
            lora_a_stacked=(
                torch.zeros(1, 1, 8, 16, dtype=torch.float16),
                noncontiguous_a,
            ),
            lora_b_stacked=(
                torch.zeros(1, 1, 8, 8, dtype=torch.float16),
                torch.zeros(1, 1, 8, 8, dtype=torch.float16),
            ),
            adapter_enabled=torch.tensor([1]),
            combined_indices=torch.zeros(2, dtype=torch.int32),
        )

    wrapper.moe_lora_bgmv_fused.assert_not_called()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


def test_allgather_routing_preserves_multi_adapter_and_base_mapping() -> None:
    context = SimpleNamespace(
        top_k=2,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0, -1, 1])),
        adapter_enabled=torch.tensor([True, True]),
        local_num_experts=2,
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]])
    # Original flat rows [0..5] land at these expert-sorted positions.
    expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])

    expected = torch.tensor([0, -1, 1, -1, 3, 3], dtype=torch.int32)
    with patch(
        "vllm_ascend.lora.fused_moe.moe_lora_build_combined_idx",
        return_value=expected,
    ) as build_combined:
        combined = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    assert combined is expected
    build_combined.assert_called_once_with(
        expanded_row_idx,
        topk_ids,
        context.punica_wrapper.token_lora_indices,
        context.adapter_enabled,
        2,
    )


def test_moe_lora_apply_reuses_fused_combined_indices() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="already_applied_by_fused_index_builder",
        fully_sharded=False,
        tp_rank=0,
    )
    combined = torch.tensor([0, -1, 3], dtype=torch.int32)

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=combined,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=combined,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["combined_indices"] is combined
    assert calls[1].kwargs["combined_indices"] is combined
    assert calls[0].kwargs["expert_ids"] is None
    assert calls[1].kwargs["expert_ids"] is None


def test_all2all_routing_uses_local_experts_and_exchanged_adapters() -> None:
    context = SimpleNamespace(
        local_num_experts=3,
        exchanged_lora_indices=torch.tensor([1, -1, 0, 2]),
    )

    expert_ids, lora_slots = _recover_moe_lora_routing_all2all(
        context,
        group_list=torch.tensor([2, 0, 2]),
    )

    assert torch.equal(expert_ids, torch.tensor([0, 0, 2, 2]))
    assert torch.equal(lora_slots, torch.tensor([1, -1, 0, 2]))


def test_has_lora_follows_batch_metadata() -> None:
    assert not has_lora(None)
    assert not has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True)))
    assert has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False)))


@pytest.mark.parametrize(
    ("index_mapping", "expected_no_lora"),
    [((0, 0), True), ((0, 1), False), ((2, 0), False)],
)
def test_decode_metadata_refreshes_no_lora(index_mapping, expected_no_lora) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    mapping = SimpleNamespace(index_mapping=index_mapping)
    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [], 2, 100)
    assert wrapper.no_lora is expected_no_lora
