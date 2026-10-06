# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.lora.fused_moe import scoped_moe_lora_context
from vllm_ascend.lora.routing import recover_allgather_ep_lora_routing
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEQuantParams, MoERoutingParams, MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType

PREPARE = "vllm_ascend.ops.fused_moe.prepare_finalize"
DISPATCH = "vllm_ascend.ops.fused_moe.token_dispatcher"


@pytest.mark.parametrize("count", [0, 1, 7, 32])
@pytest.mark.parametrize("local_begin", [0, 2])
@pytest.mark.parametrize("no_local_hit", [False, True])
def test_local_routing_matches_independent_pair_enumeration(count, local_begin, no_local_hit):
    topk = torch.arange(count * 2, dtype=torch.int32).reshape(count, 2) % 4
    if no_local_hit:
        topk.fill_((local_begin + 2) % 4)
    # Include repeated expert selections for a token.
    if count and not no_local_hit:
        topk[0, 1] = topk[0, 0]
    slots = (torch.arange(count, dtype=torch.int64) % 4) - 1
    expert_map = torch.full((4,), -1, dtype=torch.int32)
    expert_map[local_begin : local_begin + 2] = torch.arange(2, dtype=torch.int32)
    pairs = sorted(
        [(e, t, k) for t in range(count) for k, e in enumerate(topk[t].tolist()) if local_begin <= e < local_begin + 2]
    )
    expanded = torch.full((count * 2,), -1, dtype=torch.int32)
    expected_experts = torch.full((count * 2,), -1, dtype=torch.int64)
    expected_slots = torch.full_like(expected_experts, -1)
    for row, (expert, token, k) in enumerate(pairs):
        expanded[token * 2 + k] = row
        expected_experts[row] = expert - local_begin
        expected_slots[row] = slots[token]
    result = recover_allgather_ep_lora_routing(
        expanded_row_idx=expanded,
        topk_ids=topk,
        prepared_lora_indices=slots,
        expert_map=expert_map,
        num_dispatched_rows=count * 2,
    )
    torch.testing.assert_close(result.expert_ids, expected_experts, rtol=0, atol=0)
    torch.testing.assert_close(result.adapter_slots, expected_slots, rtol=0, atol=0)


def test_default_backend_and_invalid_config():
    assert AscendConfig._parse_moe_lora_ep_backend({}) == "allgather"
    for backend in ("allgather", "alltoall"):
        assert AscendConfig._parse_moe_lora_ep_backend({"moe_lora_ep_backend": backend}) == backend
    for backend in ("ALLGATHER", "mc2", None, 0, []):
        with pytest.raises(ValueError):
            AscendConfig._parse_moe_lora_ep_backend({"moe_lora_ep_backend": backend})


def test_fp32_sort_key_bound_checked_without_allocating_large_buffer():
    capacity = (1 << 24) + 1
    with pytest.raises(ValueError, match="exact FP32"):
        recover_allgather_ep_lora_routing(
            expanded_row_idx=torch.empty(capacity, device="meta", dtype=torch.int32),
            topk_ids=torch.empty((1, capacity), device="meta", dtype=torch.int32),
            prepared_lora_indices=torch.empty(1, device="meta", dtype=torch.int64),
            expert_map=torch.empty(2, device="meta", dtype=torch.int32),
            num_dispatched_rows=capacity,
        )


def test_fully_sharded_ep_remains_rejected_by_inherited_guard():
    from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

    from vllm_ascend.lora.fused_moe import AscendFusedMoEWithLoRA

    assert AscendFusedMoEWithLoRA._verify_ep_fs is FusedMoEWithLoRA._verify_ep_fs
    with pytest.raises(AssertionError, match="fully_sharded_loras=True"):
        AscendFusedMoEWithLoRA._verify_ep_fs(SimpleNamespace(use_ep=True), SimpleNamespace(fully_sharded_loras=True))


def _engine_config(**overrides):
    parallel = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        expert_placement_strategy="linear",
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    for name, value in overrides.items():
        setattr(parallel, name, value)
    return SimpleNamespace(
        lora_config=object(),
        model_config=SimpleNamespace(get_num_experts=lambda: 256),
        parallel_config=parallel,
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("dynamic_eplb", True),
        ("expert_map_path", "custom.json"),
        ("expert_map_record_path", "record.json"),
        ("num_redundant_experts", 1),
    ],
)
def test_allgather_rejects_nonstatic_expert_maps(name, value):
    additional = {"eplb_config": {name: value}}
    with pytest.raises(ValueError, match="static linear"):
        AscendConfig._validate_moe_lora_ep_backend(_engine_config(), additional, "allgather")
    AscendConfig._validate_moe_lora_ep_backend(_engine_config(), additional, "alltoall")


@pytest.mark.parametrize(
    "overrides,additional",
    [
        ({"expert_placement_strategy": "round_robin"}, {}),
        ({"enable_eplb": True}, {}),
        ({"eplb_config": SimpleNamespace(num_redundant_experts=1)}, {}),
        ({}, {"mix_placement": True}),
        ({"data_parallel_size": 2}, {}),
        ({"prefill_context_parallel_size": 2}, {}),
        ({"decode_context_parallel_size": 2}, {}),
    ],
)
def test_allgather_rejects_unsupported_parallel_layout(overrides, additional):
    with pytest.raises(ValueError):
        AscendConfig._validate_moe_lora_ep_backend(_engine_config(**overrides), additional, "allgather")


@pytest.mark.parametrize("scope", ["ep1", "dense", "no_lora", "no_ep"])
def test_allgather_guards_do_not_change_unrelated_engines(scope):
    config = _engine_config(expert_placement_strategy="round_robin")
    if scope == "ep1":
        config.parallel_config.tensor_parallel_size = 1
    elif scope == "dense":
        config.model_config.get_num_experts = lambda: 0
    elif scope == "no_lora":
        config.lora_config = None
    else:
        config.parallel_config.enable_expert_parallel = False
    AscendConfig._validate_moe_lora_ep_backend(config, {"mix_placement": True}, "allgather")


@pytest.mark.parametrize("backend", ["allgather", "alltoall"])
@pytest.mark.parametrize("tokens", [0, 1, 8192])
def test_backend_is_engine_fixed_including_dummy(backend, tokens):
    from vllm_ascend import ascend_forward_context as afc

    config = SimpleNamespace(lora_config=object(), parallel_config=SimpleNamespace(enable_expert_parallel=True))
    with (
        patch.object(afc, "is_moe_model", return_value=True),
        patch.object(afc, "get_mc2_tokens_capacity", return_value=256),
        patch.object(afc, "get_ascend_device_type"),
        patch.object(afc, "get_ep_group", return_value=SimpleNamespace(world_size=8)),
        patch.object(afc, "get_ascend_config", return_value=SimpleNamespace(moe_lora_ep_backend=backend)),
    ):
        expected = MoECommType.ALLGATHER if backend == "allgather" else MoECommType.ALLTOALL
        assert afc.select_moe_comm_method(tokens, config) == expected


def _context(slots):
    return SimpleNamespace(use_ep=True, punica_wrapper=SimpleNamespace(token_lora_indices=slots, no_lora=False))


def test_non_sp_prepare_preserves_float_and_does_not_gather():
    context = _context(torch.tensor([0, -1, 2], dtype=torch.int64))
    prepare = PrepareAndFinalizeWithAllGather(SimpleNamespace(dp_size=1, pcp_size=1))
    prepare.set_lora_context(context)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    logits = torch.randn(3, 4)
    with (
        patch(f"{PREPARE}.enable_sp", return_value=False),
        patch(f"{PREPARE}.enable_sp_by_pass", return_value=False),
        patch(f"{PREPARE}._EXTRA_CTX", SimpleNamespace(flash_comm_v1_enabled=False)),
        patch(f"{PREPARE}.torch_npu.npu_dynamic_quant", side_effect=AssertionError("early quant")),
    ):
        output = prepare.prepare(hidden, logits, quant_type=QuantType.W8A8)
        assert output.hidden_states is hidden and output.router_logits is logits
        assert output.pertoken_scale is None
        assert context.prepared_lora_indices.tolist() == [0, -1, 2]
        context.punica_wrapper.token_lora_indices.copy_(torch.tensor([2, 1, 0]))
        prepare.prepare(hidden, logits)
        assert context.prepared_lora_indices.tolist() == [2, 1, 0]
        assert prepare.finalize(hidden, reduce_results=True) is hidden


def test_sp_prepare_slots_follow_hidden_split_gather_unpad():
    context = _context(torch.tensor([0, 1, 2], dtype=torch.int64))
    prepare = PrepareAndFinalizeWithAllGather(SimpleNamespace(dp_size=1, pcp_size=1))
    prepare.set_lora_context(context)
    hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    logits = torch.randn(2, 4)
    observed = []

    def gather(x, label, is_ep):
        observed.append(x.clone())
        if x.dtype == torch.int64:
            assert x.tolist() == [2, -1]
            return torch.tensor([0, 1, 2], dtype=torch.int64)
        return torch.cat((x, x), dim=0)[:3]

    with (
        patch(f"{PREPARE}.enable_sp", return_value=True),
        patch(f"{PREPARE}.enable_sp_by_pass", return_value=False),
        patch(f"{PREPARE}._EXTRA_CTX", SimpleNamespace(flash_comm_v1_enabled=True, pad_size=1)),
        patch(f"{PREPARE}.get_tensor_model_parallel_world_size", return_value=2),
        patch(f"{PREPARE}.get_tensor_model_parallel_rank", return_value=1),
        patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad", side_effect=gather, create=True),
        patch(f"{PREPARE}.torch_npu.npu_dynamic_quant", side_effect=AssertionError("early quant")),
    ):
        output = prepare.prepare(hidden, logits, quant_type=QuantType.W8A8)
    assert len(observed) == 3
    assert output.hidden_states.shape[0] == context.prepared_lora_indices.shape[0] == 3
    assert output.hidden_states.dtype == torch.bfloat16 and output.pertoken_scale is None
    assert context.punica_wrapper.token_lora_indices.tolist() == [0, 1, 2]


@pytest.mark.parametrize("raises", [False, True])
def test_scoped_context_restores_singletons_and_clears_transient(raises):
    previous = object()
    comm = SimpleNamespace(lora_context=previous)
    quant = SimpleNamespace(lora_context=previous)
    comm.set_lora_context = lambda value: setattr(comm, "lora_context", value)
    quant.set_lora_context = lambda value: setattr(quant, "lora_context", value)
    current = _context(torch.tensor([0]))
    with patch("vllm_ascend.lora.fused_moe._EXTRA_CTX", SimpleNamespace(moe_comm_method=comm)):
        try:
            with scoped_moe_lora_context(quant, current):
                assert comm.lora_context is current and quant.lora_context is current
                current.prepared_lora_indices = torch.tensor([1])
                if raises:
                    raise RuntimeError("injected prepare/apply failure")
        except RuntimeError:
            assert raises
        assert comm.lora_context is previous and quant.lora_context is previous
        assert not hasattr(current, "prepared_lora_indices")
        with scoped_moe_lora_context(quant, current):
            assert not hasattr(current, "prepared_lora_indices")
            current.prepared_lora_indices = torch.tensor([2])
    assert not hasattr(current, "prepared_lora_indices")


def test_dispatch_masks_poison_without_removing_base_only_rows():
    context = _context(torch.tensor([-1, 1], dtype=torch.int64))
    context.prepared_lora_indices = context.punica_wrapper.token_lora_indices
    dispatcher = TokenDispatcherWithAllGather(top_k=2, num_experts=4, num_local_experts=2)
    dispatcher.set_lora_context(context)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 2], [3, 0]], dtype=torch.int32)
    routed = torch.cat((hidden, torch.full_like(hidden, float("nan"))))
    request = MoETokenDispatchInput(
        hidden_states=hidden,
        topk_weights=torch.ones(2, 2),
        topk_ids=topk_ids,
        routing=MoERoutingParams(
            expert_map=torch.tensor([-1, -1, 0, 1]),
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
    )
    with (
        patch(f"{DISPATCH}.get_ep_group", return_value=SimpleNamespace(rank_in_group=1)),
        patch(
            f"{DISPATCH}.DeviceOperator.npu_moe_init_routing",
            return_value=(routed, torch.tensor([-1, 0, 1, -1]), torch.tensor([1, 1]), None),
        ) as route,
    ):
        output = dispatcher.token_dispatch(request)
    assert route.call_args.kwargs["quant_mode"] == -1
    assert output.lora_routing.expert_ids.tolist() == [0, 1, -1, -1]
    assert output.lora_routing.adapter_slots.tolist() == [-1, 1, -1, -1]
    torch.testing.assert_close(output.hidden_states[:2], hidden)
    assert output.hidden_states[2:].count_nonzero() == 0
    assert output.dynamic_scale is None
    combine = Mock(side_effect=lambda **kwargs: kwargs["permuted_tokens"])
    with patch(f"{DISPATCH}.DeviceOperator.npu_moe_token_unpermute", combine):
        result = dispatcher.token_combine(routed, output.combine_metadata)
    torch.testing.assert_close(result[:2], hidden)
    assert result[2:].count_nonzero() == 0 and result.isfinite().all()
