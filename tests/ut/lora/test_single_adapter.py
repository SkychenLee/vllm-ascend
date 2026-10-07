# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; native arithmetic and replay are covered on NPU."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

from tests.ut.lora.test_punica_npu import _make_wrapper
from tests.ut.lora.test_quant_moe import QUANT_MOE, _make_input
from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.lora import single_adapter
from vllm_ascend.lora.fused_moe import (
    moe_lora_apply_w2,
    moe_lora_apply_w13,
    moe_lora_apply_w13_swiglu_quant,
)


@dataclass
class TensorDescriptor:
    """Describe an NPU tensor without permitting any data-dependent read."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: object = field(default_factory=lambda: torch.device("npu:0"))
    contiguous_calls: int = 0

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def contiguous(self):
        self.contiguous_calls += 1
        return self

    def __getitem__(self, key):
        assert isinstance(key, slice) and self.ndim == 1
        length = len(range(self.shape[0])[key])
        if length == self.shape[0]:
            return self
        return TensorDescriptor((length,), self.dtype, self.device)

    def __getattr__(self, name):
        if name in ("item", "tolist", "cpu", "numpy", "unique", "all", "any"):
            raise AssertionError(f"hot-path metadata must not inspect tensor values: {name}")
        raise AttributeError(name)


def make_descriptors(tokens=6, top_k=6, slots=None):
    slots = tokens if slots is None else slots
    expanded = TensorDescriptor((tokens * top_k,), torch.int32)
    experts = TensorDescriptor((tokens, top_k), torch.int64)
    context = SimpleNamespace(
        fully_sharded=True,
        use_ep=False,
        top_k=top_k,
        punica_wrapper=SimpleNamespace(token_lora_indices=TensorDescriptor((slots,), torch.int64)),
        adapter_enabled=TensorDescriptor((3,), torch.int32),
        w13_lora_a_stacked=[TensorDescriptor((3, 256, 2, 4096), torch.bfloat16)] * 2,
        w2_lora_a_stacked=[TensorDescriptor((3, 256, 16, 256), torch.bfloat16)],
    )
    return context, expanded, experts


@pytest.fixture
def native_mock(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER", "1")
    native = Mock(return_value=object())
    with (
        patch.object(torch.ops, "_C_ascend", SimpleNamespace(moe_lora_single_adapter_indices=native)),
        patch.object(single_adapter.torch_npu, "get_npu_format", return_value=2),
        patch.object(torch.compiler, "is_compiling", return_value=False),
    ):
        yield native


@pytest.mark.parametrize(
    "tokens,top_k,slots,allowed",
    [
        (0, 6, 0, True),
        (1, 1, 1, True),
        (6, 6, 6, True),
        (8, 6, 8, True),
        (24, 6, 3, True),
        (48, 6, 48, True),
        (64, 6, 64, True),
        (146, 6, 146, True),
        (147, 6, 147, False),
        (1360, 6, 3, False),
        (8161, 1, 3, True),
        (1366, 6, 3, True),
        (16384, 6, 16384, True),
        (16385, 6, 16385, False),
        (1, 6, 0, False),
    ],
)
def test_static_dispatch_cost_and_empty_boundaries(native_mock, tokens, top_k, slots, allowed):
    context, expanded, experts = make_descriptors(tokens, top_k, slots)
    result = single_adapter.try_single_adapter_indices(context, expanded, experts)
    assert (result is native_mock.return_value) is allowed
    assert native_mock.call_count == int(allowed)
    if allowed:
        native_mock.assert_called_once_with(
            expanded,
            experts,
            context.punica_wrapper.token_lora_indices,
            context.adapter_enabled,
            top_k,
            256,
        )
        assert all(value.contiguous_calls == 1 for value in native_mock.call_args.args[:4])


@pytest.mark.parametrize(
    "invalid",
    [
        "context-none",
        "fully-false",
        "fully-int",
        "ep-true",
        "ep-int",
        "ep-unknown",
        "expanded-none",
        "experts-none",
        "top-k-zero",
        "top-k-float",
        "expanded-rank",
        "experts-rank",
        "slots-rank",
        "enabled-rank",
        "expert-columns",
        "row-count",
        "expanded-dtype",
        "experts-dtype",
        "slots-dtype",
        "enabled-dtype",
        "cpu",
        "expert-device",
        "slots-device",
        "enabled-device",
        "empty-enabled",
        "too-many-enabled",
        "weight-adapter-count",
        "weight-expert-count",
        "zero-experts",
        "blocked-format",
        "old-binary",
    ],
)
def test_unsupported_metadata_preserves_old_path(native_mock, invalid):
    context, expanded, experts = make_descriptors()
    slots = context.punica_wrapper.token_lora_indices
    enabled = context.adapter_enabled
    if invalid == "context-none":
        context = None
    elif invalid.startswith("fully-"):
        context.fully_sharded = False if invalid == "fully-false" else 1
    elif invalid.startswith("ep-"):
        context.use_ep = {"ep-true": True, "ep-int": 0, "ep-unknown": None}[invalid]
    elif invalid == "expanded-none":
        expanded = None
    elif invalid == "experts-none":
        experts = None
    elif invalid.startswith("top-k-"):
        context.top_k = 0 if invalid == "top-k-zero" else 6.0
    elif invalid.endswith("-rank"):
        target = {"expanded-rank": expanded, "experts-rank": experts, "slots-rank": slots, "enabled-rank": enabled}[
            invalid
        ]
        target.shape = (target.numel(),) if target is experts else (1, target.numel())
    elif invalid == "expert-columns":
        experts.shape = (12, 3)
    elif invalid == "row-count":
        expanded.shape = (35,)
    elif invalid.endswith("-dtype"):
        target = {"expanded-dtype": expanded, "experts-dtype": experts, "slots-dtype": slots, "enabled-dtype": enabled}[
            invalid
        ]
        target.dtype = torch.float32
    elif invalid == "cpu":
        expanded.device = torch.device("cpu")
    elif invalid.endswith("-device"):
        target = {"expert-device": experts, "slots-device": slots, "enabled-device": enabled}[invalid]
        target.device = torch.device("npu:1")
    elif invalid == "empty-enabled":
        enabled.shape = (0,)
    elif invalid == "too-many-enabled":
        enabled.shape = (65,)
    elif invalid == "weight-adapter-count":
        context.w13_lora_a_stacked[0].shape = (2, 256, 2, 4096)
    elif invalid == "weight-expert-count":
        context.w2_lora_a_stacked[0].shape = (3, 255, 16, 256)
    elif invalid == "zero-experts":
        context.w13_lora_a_stacked[0].shape = (3, 0, 2, 4096)
        context.w2_lora_a_stacked[0].shape = (3, 0, 16, 256)
    with (
        patch.object(single_adapter.torch_npu, "get_npu_format", return_value=29 if invalid == "blocked-format" else 2),
        patch.object(
            torch.ops,
            "_C_ascend",
            SimpleNamespace()
            if invalid == "old-binary"
            else SimpleNamespace(moe_lora_single_adapter_indices=native_mock),
        ),
    ):
        assert single_adapter.try_single_adapter_indices(context, expanded, experts) is None
    native_mock.assert_not_called()


@pytest.mark.parametrize("flag", [None, "0", "1", "-1", "2", "true", "", "invalid"])
def test_opt_in_flag_is_strict_binary(flag, monkeypatch):
    if flag is None:
        monkeypatch.delenv("VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER", raising=False)
    else:
        monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER", flag)
    if flag in (None, "0", "1"):
        assert envs.VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER is (flag == "1")
    else:
        with pytest.raises(ValueError, match="VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER"):
            _ = envs.VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER


def test_disabled_flag_does_not_touch_context_or_native(native_mock, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER", "0")
    assert single_adapter.try_single_adapter_indices(object(), object(), object()) is None
    native_mock.assert_not_called()


def test_real_cpu_tensors_never_dispatch_native(native_mock):
    context, _, _ = make_descriptors()
    context.punica_wrapper.token_lora_indices = torch.zeros(6, dtype=torch.int64)
    context.adapter_enabled = torch.ones(3, dtype=torch.int32)
    assert (
        single_adapter.try_single_adapter_indices(
            context, torch.arange(36, dtype=torch.int32), torch.zeros((6, 6), dtype=torch.int64)
        )
        is None
    )
    native_mock.assert_not_called()


def test_compilation_skips_physical_format_query(native_mock):
    context, expanded, experts = make_descriptors()
    with (
        patch.object(torch.compiler, "is_compiling", return_value=True),
        patch.object(
            single_adapter.torch_npu, "get_npu_format", side_effect=AssertionError("FakeTensor has no storage")
        ),
    ):
        assert single_adapter.try_single_adapter_indices(context, expanded, experts) is native_mock.return_value


def test_mapping_and_enable_buffers_are_read_again_each_forward(native_mock):
    context, expanded, experts = make_descriptors()
    first_slots = context.punica_wrapper.token_lora_indices
    first_mask = context.adapter_enabled
    single_adapter.try_single_adapter_indices(context, expanded, experts)
    context.punica_wrapper.token_lora_indices = TensorDescriptor((6,), torch.int64)
    context.adapter_enabled = TensorDescriptor((3,), torch.int32)
    single_adapter.try_single_adapter_indices(context, expanded, experts)
    assert native_mock.call_count == 2
    assert native_mock.call_args_list[0].args[2] is first_slots
    assert native_mock.call_args_list[0].args[3] is first_mask
    assert native_mock.call_args_list[1].args[2] is context.punica_wrapper.token_lora_indices
    assert native_mock.call_args_list[1].args[3] is context.adapter_enabled


@pytest.mark.parametrize("compiling", [False, True])
@pytest.mark.parametrize("storage_format", [0, 2])
def test_real_three_adapter_four_mask_contract(native_mock, compiling, storage_format):
    context, _, _ = make_descriptors()
    expanded = torch.arange(36, dtype=torch.int32)
    experts = torch.zeros((6, 6), dtype=torch.int64)
    context.punica_wrapper.token_lora_indices = torch.tensor([0, 1, 2, -1, 0, 2], dtype=torch.int64)
    # Upstream allocates max_loras+1 enable entries, but the weight stacks
    # contain max_loras adapters. The fourth mask entry owns no LoRA weight.
    context.adapter_enabled = torch.tensor([1, 0, -7, 1], dtype=torch.int32)
    full_mask = context.adapter_enabled
    snapshot = full_mask.clone()
    device = torch.device("npu:0")
    format_effect = AssertionError("FakeTensor has no physical format") if compiling else None
    with (
        patch.object(torch.Tensor, "device", property(lambda self: device)),
        patch.object(torch.compiler, "is_compiling", return_value=compiling),
        patch.object(
            single_adapter.torch_npu, "get_npu_format", return_value=storage_format, side_effect=format_effect
        ) as get_format,
    ):
        assert single_adapter.try_single_adapter_indices(context, expanded, experts) is native_mock.return_value
    forwarded_mask = native_mock.call_args.args[3]
    assert forwarded_mask.shape == (3,) and forwarded_mask.dtype == torch.int32
    assert torch.equal(forwarded_mask, snapshot[:3])
    assert torch._C._is_alias_of(forwarded_mask, full_mask)
    assert context.adapter_enabled is full_mask and torch.equal(full_mask, snapshot)
    assert native_mock.call_args.args[4:] == (6, 256)
    assert get_format.call_count == (0 if compiling else 4)


@pytest.mark.parametrize("compiling", [False, True])
@pytest.mark.parametrize("mask_count", [2, 5])
def test_unrelated_enable_lengths_fall_back(native_mock, compiling, mask_count):
    context, expanded, experts = make_descriptors()
    context.adapter_enabled = TensorDescriptor((mask_count,), torch.int32)
    with patch.object(torch.compiler, "is_compiling", return_value=compiling):
        assert single_adapter.try_single_adapter_indices(context, expanded, experts) is None
    native_mock.assert_not_called()


@pytest.mark.parametrize("adapter_count,allowed", [(0, False), (64, True), (65, False)])
def test_weight_adapter_limit_is_independent_of_padding_mask(native_mock, adapter_count, allowed):
    context, expanded, experts = make_descriptors()
    context.w13_lora_a_stacked[0].shape = (adapter_count, 256, 2, 4096)
    context.w2_lora_a_stacked[0].shape = (adapter_count, 256, 16, 256)
    context.adapter_enabled = TensorDescriptor((adapter_count + 1,), torch.int32)
    output = single_adapter.try_single_adapter_indices(context, expanded, experts)
    assert (output is native_mock.return_value) is allowed
    assert native_mock.call_count == int(allowed)
    if allowed:
        assert native_mock.call_args.args[3].numel() == adapter_count


@pytest.mark.parametrize("compiling", [False, True])
def test_actual_upstream_allocation_and_context_reach_fast_path(native_mock, compiling):
    # Exercise the installed upstream allocator and context constructor on
    # Meta tensors. This covers its max_loras+1 mask convention, rather than
    # constructing only a context that happens to match the helper guard.
    layer = SimpleNamespace(
        device=torch.device("meta"),
        use_ep=False,
        tp_size=8,
        tp_rank=0,
        local_num_experts=256,
        hidden_size=4096,
        intermediate_size_per_partition=256,
        _w13_slices=2,
        _enable_aux_cuda_stream=False,
        _lora_stream=None,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.zeros(6, dtype=torch.int64)),
        moe_config=SimpleNamespace(experts_per_token=6),
    )
    layer._verify_ep_fs = lambda config: FusedMoEWithLoRA._verify_ep_fs(layer, config)
    layer._create_lora_a_weights = lambda count, config: FusedMoEWithLoRA._create_lora_a_weights(layer, count, config)
    layer._create_lora_b_weights = lambda count, config: FusedMoEWithLoRA._create_lora_b_weights(layer, count, config)
    config = SimpleNamespace(max_loras=3, fully_sharded_loras=True, max_lora_rank=16, lora_dtype=torch.bfloat16)
    FusedMoEWithLoRA.create_lora_weights(layer, 3, config)
    context = FusedMoEWithLoRA._build_lora_context(layer)
    # Ascend._build_lora_context supplies this additional upstream property.
    context.use_ep = layer.use_ep
    assert context.adapter_enabled.shape == (4,) and context.adapter_enabled.dtype == torch.int32
    assert context.w13_lora_a_stacked[0].shape == (3, 256, 2, 4096)
    assert context.w2_lora_a_stacked[0].shape == (3, 256, 16, 256)
    assert context.fully_sharded is True and context.use_ep is False and context.top_k == 6
    device = torch.device("npu:0")
    with (
        patch.object(torch.Tensor, "device", property(lambda self: device)),
        patch.object(torch.compiler, "is_compiling", return_value=compiling),
    ):
        output = single_adapter.try_single_adapter_indices(
            context, torch.arange(36, dtype=torch.int32), torch.zeros((6, 6), dtype=torch.int32)
        )
    assert output is native_mock.return_value
    assert native_mock.call_args.args[3].shape == (3,)
    assert torch._C._is_alias_of(native_mock.call_args.args[3], context.adapter_enabled)


@pytest.mark.parametrize("fuse_activation", [False, True])
def test_quant_forward_reuses_one_combined_tensor_without_recovery(fuse_activation):
    from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora

    context = SimpleNamespace(use_ep=False, fully_sharded=True)
    payload = _make_input(
        hidden_states=torch.ones(2, 4, dtype=torch.int8),
        dynamic_scale=torch.ones(2),
        output_dtype=torch.bfloat16,
        lora_context=context,
        fusion=fuse_activation,
    )
    combined = torch.tensor([0, -1])
    activated = torch.ones(2, 3, dtype=torch.int8)
    scale = torch.ones(2)
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
        patch(f"{QUANT_MOE}.try_single_adapter_indices", return_value=combined) as single,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather") as recover,
        patch(f"{QUANT_MOE}.combined_moe_lora_indices") as build,
        patch(f"{QUANT_MOE}._can_fuse_int8_lora_swiglu", return_value=fuse_activation),
        patch(f"{QUANT_MOE}.can_use_grouped_moe_lora", return_value=False),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[torch.zeros(2, 6)]),
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13_swiglu_quant", return_value=(activated, scale, combined)) as fused,
        patch(f"{QUANT_MOE}.can_fuse_moe_lora_activation", return_value=False),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=torch.zeros(2, 3)),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant", return_value=(activated, scale)),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_grouped_matmul_gmm2", return_value=torch.zeros(2, 4)),
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as w2,
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=Mock()),
    ):
        quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
    single.assert_called_once_with(context, payload.expanded_row_idx, payload.topk_ids)
    recover.assert_not_called()
    build.assert_not_called()
    first = fused if fuse_activation else w13
    first.assert_called_once()
    w2.assert_called_once()
    assert first.call_args.kwargs["lora_routing"] is None
    assert w2.call_args.kwargs["lora_routing"] is None
    assert first.call_args.kwargs["combined_indices"] is combined
    assert w2.call_args.kwargs["combined_indices"] is combined


def test_none_result_preserves_original_recovery_and_routing():
    from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora

    payload = _make_input(lora_context=SimpleNamespace(use_ep=False, fully_sharded=True), fusion=False)
    routing = (torch.tensor([0, 1]), torch.tensor([0, -1]))
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
        patch(f"{QUANT_MOE}.try_single_adapter_indices", return_value=None),
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather", return_value=routing) as recover,
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[torch.zeros(2, 6)]),
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as w13,
        patch(f"{QUANT_MOE}.can_fuse_moe_lora_activation", return_value=False),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=torch.zeros(2, 3)),
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            return_value=(torch.zeros(2, 3, dtype=torch.int8), torch.ones(2)),
        ),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_grouped_matmul_gmm2", return_value=torch.zeros(2, 4)),
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as w2,
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=Mock()),
    ):
        quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
    recover.assert_called_once_with(payload.lora_context, payload.expanded_row_idx, payload.topk_ids)
    assert w13.call_args.kwargs["lora_routing"] is routing
    assert w2.call_args.kwargs["lora_routing"] is routing
    assert "combined_indices" not in w13.call_args.kwargs


@pytest.mark.parametrize("rows", [0, 3])
def test_wrappers_accept_combined_only_and_keep_tp_output_offset(rows):
    wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=wrapper,
        fully_sharded=True,
        tp_rank=5,
        adapter_enabled=torch.ones(3, dtype=torch.int32),
        w13_lora_a_stacked=[torch.empty(3, 256, 2, 4)] * 2,
        w13_lora_b_stacked=[torch.empty(3, 256, 3, 16)] * 2,
        w2_lora_a_stacked=[torch.empty(3, 256, 16, 3)],
        w2_lora_b_stacked=[torch.empty(3, 256, 512, 16)],
        split_lora_indices=object(),
    )
    combined = torch.zeros(rows, dtype=torch.int64)
    moe_lora_apply_w13(
        context,
        gate_up_out=torch.zeros(rows, 6),
        hidden_states=torch.zeros(rows, 4),
        lora_routing=None,
        combined_indices=combined,
    )
    moe_lora_apply_w2(
        context,
        down_out=torch.zeros(rows, 4096),
        silu_out=torch.zeros(rows, 3),
        lora_routing=None,
        combined_indices=combined,
    )
    assert wrapper.add_lora_fused_moe.call_count == (2 if rows else 0)
    if rows:
        first, second = wrapper.add_lora_fused_moe.call_args_list
        assert first.kwargs["expert_ids"] is first.kwargs["token_lora_mapping"] is None
        assert second.kwargs["expert_ids"] is second.kwargs["token_lora_mapping"] is None
        assert first.kwargs["combined_indices"] is second.kwargs["combined_indices"] is combined
        assert second.kwargs["offset"] == 512 * 5
        assert not hasattr(context, "split_lora_indices")


def test_fused_wrapper_accepts_combined_without_separate_metadata():
    context = SimpleNamespace(
        punica_wrapper=Mock(),
        fully_sharded=True,
        adapter_enabled=torch.ones(3, dtype=torch.int32),
        w13_lora_a_stacked=[],
        w13_lora_b_stacked=[],
        w13_lora_a_packed=object(),
    )
    combined = torch.tensor([0, -1])
    moe_lora_apply_w13_swiglu_quant(
        context,
        gate_up_out=torch.zeros(2, 6),
        hidden_states=torch.zeros(2, 4),
        input_scale=torch.ones(2),
        lora_routing=None,
        swiglu_limit=10.0,
        combined_indices=combined,
    )
    kwargs = context.punica_wrapper.add_lora_fused_moe.call_args.kwargs
    assert kwargs["combined_indices"] is combined
    assert kwargs["expert_ids"] is kwargs["token_lora_mapping"] is None


def test_punica_combined_only_never_rebuilds_routing():
    wrapper = _make_wrapper()
    combined = torch.tensor([0, -1, 3])
    with (
        patch("vllm_ascend.lora.punica_npu.combined_moe_lora_indices") as build,
        patch("torch.ops._C_ascend.bgmv_shrink_int8", create=True) as shrink,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce", side_effect=lambda value: value),
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(3, 16, dtype=torch.bfloat16),
            x=torch.ones(3, 8, dtype=torch.int8),
            lora_a_stacked=[torch.ones(2, 2, 16, 8, dtype=torch.bfloat16)],
            lora_b_stacked=[torch.ones(2, 2, 16, 16, dtype=torch.bfloat16)],
            expert_ids=None,
            adapter_enabled=torch.ones(2, dtype=torch.int32),
            token_lora_mapping=None,
            input_scale=torch.ones(3),
            combined_indices=combined,
            fully_sharded=True,
        )
    build.assert_not_called()
    assert shrink.call_args.args[2] is combined
