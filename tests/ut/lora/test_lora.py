import inspect
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
    preprocess_lora_indices,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.quantization.quant_type import QuantType


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
    assert wrapper._single_lora_packed_weights is None
    assert wrapper._single_lora_cache_slot is None
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
        output.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    lora_a = (torch.zeros(2, 2, 2, 3),)
    lora_b = (torch.zeros(2, 2, 5, 4),)

    with (
        patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather",
            side_effect=lambda value: torch.cat((value, value + 10), dim=-1),
        ) as all_gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce") as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 5),
            x=torch.zeros(2, 3),
            lora_a_stacked=lora_a,
            lora_b_stacked=lora_b,
            expert_ids=torch.tensor([0, 1]),
            adapter_enabled=torch.tensor([1, 1]),
            fully_sharded=True,
            token_lora_mapping=torch.tensor([0, 1]),
        )

    all_gather.assert_called_once()
    all_reduce.assert_not_called()
    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(
        expand_args[0],
        torch.tensor([[1.0, 2.0, 11.0, 12.0], [3.0, 4.0, 13.0, 14.0]]),
    )
    assert expand_args[1].shape == (4, 5, 4)
    assert torch.equal(expand_args[3], torch.tensor([0, 3]))


def test_punica_fully_sharded_moe_reduces_partial_rank() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)

    def shrink(_, __, output, ___, ____):
        output.copy_(torch.arange(8, dtype=torch.float32).view(2, 4))

    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    lora_a = (torch.zeros(2, 2, 4, 3),)
    lora_b = (torch.zeros(2, 2, 5, 4),)

    with (
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather") as all_gather,
        patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value + 10,
        ) as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 10),
            x=torch.zeros(2, 3),
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
    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(
        expand_args[0],
        torch.arange(8, dtype=torch.float32).view(2, 4) + 10,
    )
    assert expand_args[1].shape == (4, 5, 4)
    assert expand_args[4] == 5


def test_allgather_routing_preserves_multi_adapter_and_base_mapping() -> None:
    context = SimpleNamespace(
        top_k=2,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0, -1, 1])),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]])
    # Original flat rows [0..5] land at these expert-sorted positions.
    expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])

    expert_ids, lora_slots = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    assert torch.equal(expert_ids, torch.tensor([0, 0, 1, 1, 1, 1]))
    assert torch.equal(lora_slots, torch.tensor([0, -1, 0, -1, 1, 1]))


def test_preprocess_lora_indices_broadcasts_before_permutation() -> None:
    context = SimpleNamespace(split_lora_indices=torch.tensor([7, -1]))
    topk_ids = torch.tensor([[0, 1], [1, 0]])
    reversed_permutation_mapping = torch.tensor([[2, 0], [3, 1]])

    preprocess_lora_indices(
        context,
        topk_ids=topk_ids,
        reversed_permutation_mapping=reversed_permutation_mapping,
    )

    assert torch.equal(context.permuted_lora_indices, torch.tensor([7, -1, 7, -1]))


def test_allgather_ep_routing_uses_gathered_adapters_and_local_experts() -> None:
    context = SimpleNamespace(
        top_k=2,
        allgather_lora_indices=torch.tensor([1, 2]),
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([9, 9])),
    )
    topk_ids = torch.tensor([[0, 2], [3, 1]])
    # Only the two local pairs have valid destinations (0 and 1). Non-local
    # pairs use repeated invalid entries, matching active_expert_range output.
    expanded_row_idx = torch.tensor([-1, 0, 1, -1])
    expert_map = torch.tensor([-1, -1, 0, 1])

    expert_ids, lora_slots = _recover_moe_lora_routing_allgather(
        context,
        expanded_row_idx,
        topk_ids,
        expert_map=expert_map,
    )

    assert torch.equal(expert_ids, torch.tensor([0, 1, 0, 0]))
    assert torch.equal(lora_slots, torch.tensor([1, 2, -1, -1]))


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


def test_all2all_routing_handles_leading_and_consecutive_empty_experts() -> None:
    context = SimpleNamespace(
        local_num_experts=4,
        exchanged_lora_indices=torch.tensor([5, -1, 2]),
    )

    expert_ids, lora_slots = _recover_moe_lora_routing_all2all(
        context,
        group_list=torch.tensor([0, 2, 0, 1]),
    )

    assert torch.equal(expert_ids, torch.tensor([1, 1, 3]))
    assert torch.equal(lora_slots, torch.tensor([5, -1, 2]))


@pytest.mark.parametrize(
    "routing_fn",
    [
        preprocess_lora_indices,
        _recover_moe_lora_routing_allgather,
        _recover_moe_lora_routing_all2all,
    ],
)
def test_moe_lora_routing_does_not_use_repeat_interleave(routing_fn) -> None:
    assert ".repeat_interleave(" not in inspect.getsource(routing_fn)


def test_has_lora_follows_batch_metadata() -> None:
    assert not has_lora(None)
    assert not has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True)))
    assert has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False)))
    assert has_lora(
        SimpleNamespace(
            allgather_lora_indices=torch.tensor([-1]),
            punica_wrapper=SimpleNamespace(no_lora=True),
        )
    )


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


@pytest.mark.parametrize(
    ("is_prefill", "index_mapping", "expected_count", "expected_slot"),
    [
        (True, (42, 42, 42), 1, 1),
        (True, (42, 7, 42), 2, None),
        (True, (42, 0, 42), 1, 1),
        (False, (42, 42, 42), 1, 1),
    ],
)
def test_metadata_tracks_active_moe_lora_slot(
    is_prefill,
    index_mapping,
    expected_count,
    expected_slot,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    mapping = SimpleNamespace(is_prefill=is_prefill, index_mapping=index_mapping)
    layer = Mock()
    wrapper._single_lora_moe_layers = (layer,)

    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [7, 42, None], 3, 100)

    assert wrapper.num_active_moe_loras == expected_count
    assert wrapper.active_moe_lora_slot == expected_slot
    if expected_slot is None:
        layer.refresh_single_lora_cache.assert_not_called()
    else:
        layer.refresh_single_lora_cache.assert_called_once_with(expected_slot)


def test_metadata_does_not_refresh_an_unchanged_single_lora_slot() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.active_moe_lora_slot = 1
    layer = Mock()
    wrapper._single_lora_moe_layers = (layer,)
    mapping = SimpleNamespace(is_prefill=False, index_mapping=(42, 42))

    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [7, 42, None], 3, 100)

    assert wrapper.active_moe_lora_slot == 1
    layer.refresh_single_lora_cache.assert_not_called()


def test_single_lora_cache_copies_only_when_active_slot_changes() -> None:
    layer = object.__new__(AscendFusedMoEWithLoRA)
    torch.nn.Module.__init__(layer)
    source_weights = tuple(
        (torch.stack(tuple(torch.full((2, 3), slot, dtype=torch.bfloat16) for slot in range(3))),) for _ in range(4)
    )
    layer.max_loras = 3
    layer._single_lora_packed_weights = tuple((torch.empty(2, 3, dtype=torch.bfloat16),) for _ in range(4))
    layer._single_lora_slot_views = tuple(tuple(weight.unbind(0) for weight in stacked) for stacked in source_weights)
    layer._single_lora_cache_slot = None
    layer._ascend_lora_context = SimpleNamespace(single_lora_cache_slot=None)

    layer.refresh_single_lora_cache(2)

    assert layer._single_lora_cache_slot == 2
    assert layer._ascend_lora_context.single_lora_cache_slot == 2
    assert all(
        torch.equal(stack[0], torch.full((2, 3), 2, dtype=torch.bfloat16))
        for stack in layer._single_lora_packed_weights
    )

    for stack in source_weights:
        stack[0][2].fill_(7)
    layer.refresh_single_lora_cache(2)
    assert all(
        torch.equal(stack[0], torch.full((2, 3), 2, dtype=torch.bfloat16))
        for stack in layer._single_lora_packed_weights
    )

    layer.refresh_single_lora_cache(None)
    layer.refresh_single_lora_cache(2)
    assert all(
        torch.equal(stack[0], torch.full((2, 3), 7, dtype=torch.bfloat16))
        for stack in layer._single_lora_packed_weights
    )


def test_create_lora_weights_allocates_dynamic_fixed_address_cache_for_w8a8() -> None:
    parallel_config = SimpleNamespace(tp_size=1, tp_rank=0, ep_rank=0, use_ep=False)
    base_layer = SimpleNamespace(
        quant_type=QuantType.W8A8,
        moe_config=SimpleNamespace(
            hidden_dim=4,
            num_local_experts=2,
            num_experts=2,
            intermediate_size_per_partition=3,
            experts_per_token=4,
            moe_parallel_config=parallel_config,
            is_act_and_mul=True,
        ),
    )
    lora_config = SimpleNamespace(
        max_loras=2,
        max_lora_rank=8,
        lora_dtype=torch.bfloat16,
        fully_sharded_loras=False,
        enable_moe_shared_loras=False,
    )
    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        layer = AscendFusedMoEWithLoRA(base_layer)
        layer.create_lora_weights(2, lora_config)

    assert layer._single_lora_packed_weights is not None
    assert layer._single_lora_slot_views is not None
    source_stacks = (
        layer.w13_lora_a_stacked,
        layer.w13_lora_b_stacked,
        layer.w2_lora_a_stacked,
        layer.w2_lora_b_stacked,
    )
    for packed_stack, source_stack in zip(
        layer._single_lora_packed_weights,
        source_stacks,
        strict=True,
    ):
        for packed, source in zip(packed_stack, source_stack, strict=True):
            assert packed.shape == source.shape[1:]
            assert packed.data_ptr() != source.data_ptr()

    layer._ascend_lora_context = SimpleNamespace(single_lora_cache_slot=None)
    layer.refresh_single_lora_cache(1)
    layer.set_lora(
        1,
        [
            torch.full((2, 8, 4), 1, dtype=torch.bfloat16),
            torch.full((2, 8, 3), 2, dtype=torch.bfloat16),
            torch.full((2, 8, 4), 3, dtype=torch.bfloat16),
        ],
        [
            torch.full((2, 3, 8), 4, dtype=torch.bfloat16),
            torch.full((2, 4, 8), 5, dtype=torch.bfloat16),
            torch.full((2, 3, 8), 6, dtype=torch.bfloat16),
        ],
    )
    for packed_stack, source_stack in zip(
        layer._single_lora_packed_weights,
        source_stacks,
        strict=True,
    ):
        for packed, source in zip(packed_stack, source_stack, strict=True):
            assert torch.equal(packed, source[1])

    layer.reset_lora(1)
    assert layer._single_lora_cache_slot == 1
    assert all(
        torch.count_nonzero(packed) == 0
        for packed_stack in layer._single_lora_packed_weights
        for packed in packed_stack
    )
