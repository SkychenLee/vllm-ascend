from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.quant_moe import (
    MOE_LORA_GMM_MIN_ROWS_PER_GROUP,
    _add_composite_lora_gmm,
    _add_single_lora_gmm,
    _build_composite_lora_gmm_routing,
    _build_single_lora_gmm_routing,
    _can_use_composite_lora_gmm,
    _can_use_ep_moe_lora_aux_stream,
    _can_use_single_lora_gmm,
    _CompositeLoraGMMRouting,
    _execute_moe_lora_in_parallel,
    _new_lora_delta_workspace,
    _SingleLoraGMMRouting,
    quant_apply_mlp_with_moe_lora,
    validate_quant_moe_lora_activation_input,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput, MoEQuantParams, MoEWeights
from vllm_ascend.quantization.quant_type import QuantType

QUANT_MOE = "vllm_ascend.lora.quant_moe"


@pytest.mark.parametrize("comm_type", [MoECommType.ALLGATHER, MoECommType.ALLTOALL])
def test_ep_moe_lora_aux_stream_eligibility(comm_type) -> None:
    context = SimpleNamespace(
        use_ep=True,
        fully_sharded=False,
        aux_stream=object(),
        events=tuple(object() for _ in range(4)),
    )

    assert _can_use_ep_moe_lora_aux_stream(context, comm_type, is_decode_only=True)
    assert not _can_use_ep_moe_lora_aux_stream(context, comm_type, is_decode_only=False)
    context.use_ep = False
    assert not _can_use_ep_moe_lora_aux_stream(context, comm_type, is_decode_only=True)
    context.use_ep = True
    context.fully_sharded = True
    assert not _can_use_ep_moe_lora_aux_stream(context, comm_type, is_decode_only=True)


def test_execute_moe_lora_in_parallel_forks_and_joins_streams() -> None:
    main_stream = MagicMock()
    aux_stream = MagicMock()
    start_event = MagicMock()
    done_event = MagicMock()
    base_result = torch.empty(1)
    calls = []

    with (
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=main_stream),
        patch(f"{QUANT_MOE}.npu_stream_switch", return_value=nullcontext()),
    ):
        result = _execute_moe_lora_in_parallel(
            lambda: calls.append("base") or base_result,
            lambda: calls.append("lora"),
            start_event,
            done_event,
            aux_stream,
        )

    assert result is base_result
    assert calls == ["base", "lora"]
    start_event.record.assert_called_once_with(main_stream)
    aux_stream.wait_event.assert_called_once_with(start_event)
    done_event.record.assert_called_once_with(aux_stream)
    main_stream.wait_event.assert_called_once_with(done_event)


def test_execute_moe_lora_in_parallel_enqueues_aux_prepare_before_base() -> None:
    main_stream = MagicMock()
    aux_stream = MagicMock()
    start_event = MagicMock()
    done_event = MagicMock()
    base_result = torch.empty(1)
    calls = []

    with (
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=main_stream),
        patch(f"{QUANT_MOE}.npu_stream_switch", side_effect=lambda *_args, **_kwargs: nullcontext()) as switch,
    ):
        result = _execute_moe_lora_in_parallel(
            lambda: calls.append("base") or base_result,
            lambda: calls.append("lora"),
            start_event,
            done_event,
            aux_stream,
            aux_prepare_fn=lambda: calls.append("prepare"),
        )

    assert result is base_result
    assert calls == ["prepare", "base", "lora"]
    start_event.record.assert_called_once_with(main_stream)
    aux_stream.wait_event.assert_called_once_with(start_event)
    assert switch.call_count == 2
    done_event.record.assert_called_once_with(aux_stream)
    main_stream.wait_event.assert_called_once_with(done_event)


@pytest.mark.skipif(torch.npu.is_available() is not True, reason="requires an Ascend NPU")
@pytest.mark.parametrize("with_aux_prepare", [False, True])
def test_execute_moe_lora_in_parallel_supports_aclgraph_replay(with_aux_prepare) -> None:
    static_input = torch.ones(16, device="npu", dtype=torch.float32)
    aux_stream = torch.npu.Stream()
    events = tuple(torch.npu.Event() for _ in range(2))

    def run_parallel() -> torch.Tensor:
        lora_delta = torch.empty_like(static_input)
        prepared_input = torch.empty_like(static_input)

        def base_fn() -> torch.Tensor:
            return static_input * 2

        def aux_prepare_fn() -> None:
            prepared_input.copy_(static_input + 1)

        def lora_fn() -> None:
            lora_input = prepared_input if with_aux_prepare else static_input
            lora_delta.copy_(lora_input * 3)

        base_output = _execute_moe_lora_in_parallel(
            base_fn,
            lora_fn,
            events[0],
            events[1],
            aux_stream,
            aux_prepare_fn=aux_prepare_fn if with_aux_prepare else None,
        )
        return base_output.add_(lora_delta)

    # Materialize lazy runtime resources before capture, like model warmup.
    run_parallel()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run_parallel()

    static_input.fill_(2)
    graph.replay()
    torch.npu.synchronize()

    expected = 13.0 if with_aux_prepare else 10.0
    torch.testing.assert_close(output.cpu(), torch.full((16,), expected))


def test_lora_delta_workspace_is_reused_for_w13_and_w2() -> None:
    inputs = torch.empty(5, 4, dtype=torch.bfloat16)
    w13_b = (torch.empty(2, 3, 16), torch.empty(2, 5, 16))
    w2_b = (torch.empty(2, 4, 16),)

    workspace, w13_output_size, w2_output_size = _new_lora_delta_workspace(inputs, w13_b, w2_b)

    assert workspace.shape == (5, 8)
    assert workspace.dtype == inputs.dtype
    assert w13_output_size == 8
    assert w2_output_size == 4


def _make_input(**overrides) -> MoEMlpComputeInput:
    values = dict(
        hidden_states=torch.randn(2, 4, dtype=torch.bfloat16),
        group_list=torch.tensor([1, 1], dtype=torch.int64),
        group_list_type=1,
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(
            w1=[torch.ones(1, 4, 6, dtype=torch.int8)],
            w2=[torch.ones(1, 3, 4, dtype=torch.int8)],
            w1_scale=[torch.ones(1, 6)],
            w2_scale=[torch.ones(1, 4, dtype=torch.bfloat16)],
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
        fusion=True,
        activation="silu",
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
        lora_context=SimpleNamespace(use_ep=False),
    )
    values.update(overrides)
    return MoEMlpComputeInput(**values)


@pytest.mark.parametrize(
    ("comm_type", "mlp_input"),
    [
        (MoECommType.ALLGATHER, _make_input()),
        (
            MoECommType.ALLTOALL,
            _make_input(
                expanded_row_idx=None,
                topk_ids=None,
                lora_context=SimpleNamespace(use_ep=True),
            ),
        ),
    ],
)
def test_dynamic_int8_lora_injects_at_float_boundaries(comm_type, mlp_input) -> None:
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.randn(2, 6, dtype=torch.bfloat16)
    activated = torch.randn(2, 3, dtype=torch.bfloat16)
    quantized_activated = torch.ones(2, 3, dtype=torch.int8)
    activated_scale = torch.ones(2)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    routing = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            side_effect=[(quantized_input, input_scale), (quantized_activated, activated_scale)],
        ) as dynamic_quant,
        patch(
            f"{QUANT_MOE}.torch_npu.npu_grouped_matmul",
            return_value=[gate_up_out],
            create=True,
        ) as gmm1,
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activated),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=down_out) as gmm2,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather", return_value=routing) as recover_allgather,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_all2all", return_value=routing) as recover_all2all,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as apply_w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as apply_w2,
    ):
        extra_ctx.moe_comm_type = comm_type
        output, output_event = quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert output is down_out
    assert output_event is None
    assert dynamic_quant.call_count == 2
    assert dynamic_quant.call_args_list[0].kwargs["hidden_states"] is mlp_input.hidden_states
    assert dynamic_quant.call_args_list[1].kwargs["hidden_states"] is activated
    assert gmm1.call_args.kwargs["x"][0] is quantized_input
    assert gmm2.call_args.kwargs["hidden_states"] is quantized_activated
    if comm_type == MoECommType.ALLGATHER:
        recover_allgather.assert_called_once_with(
            mlp_input.lora_context,
            mlp_input.expanded_row_idx,
            mlp_input.topk_ids,
            expert_map=mlp_input.expert_map,
        )
        recover_all2all.assert_not_called()
    else:
        recover_all2all.assert_called_once_with(
            mlp_input.lora_context,
            group_list=mlp_input.group_list,
        )
        recover_allgather.assert_not_called()
    apply_w13.assert_called_once_with(
        mlp_input.lora_context,
        gate_up_out=gate_up_out,
        hidden_states=mlp_input.hidden_states,
        lora_routing=routing,
    )
    apply_w2.assert_called_once_with(
        mlp_input.lora_context,
        down_out=down_out,
        silu_out=activated,
        lora_routing=routing,
    )


@pytest.mark.parametrize("comm_type", [MoECommType.ALLGATHER, MoECommType.ALLTOALL])
def test_dynamic_int8_ep_lora_uses_aux_stream_delta_workspace(comm_type) -> None:
    lora_context = _make_gmm_lora_context(use_ep=True)
    lora_context.aux_stream = object()
    lora_context.events = tuple(object() for _ in range(4))
    mlp_input = _make_input(
        lora_context=lora_context,
        expanded_row_idx=None if comm_type == MoECommType.ALLTOALL else torch.tensor([0, 1], dtype=torch.int32),
        topk_ids=None if comm_type == MoECommType.ALLTOALL else torch.tensor([[0], [1]], dtype=torch.int32),
    )
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.zeros(2, 6, dtype=torch.bfloat16)
    activated = torch.ones(2, 3, dtype=torch.bfloat16)
    quantized_activated = torch.ones(2, 3, dtype=torch.int8)
    activated_scale = torch.ones(2)
    down_out = torch.zeros(2, 4, dtype=torch.bfloat16)
    routing = (torch.tensor([0, 1]), torch.tensor([0, 0]))
    call_order = []

    quant_results = iter(
        [
            (quantized_input, input_scale),
            (quantized_activated, activated_scale),
        ]
    )

    def execute_parallel(base_fn, lora_fn, *_args, aux_prepare_fn=None):
        call_order.append("execute")
        if aux_prepare_fn is not None:
            aux_prepare_fn()
        base_output = base_fn()
        lora_fn()
        return base_output

    def dynamic_quant(*_args, **_kwargs):
        call_order.append("dynamic_quant")
        return next(quant_results)

    def base_w13_gmm(*_args, **_kwargs):
        call_order.append("base_w13")
        return [gate_up_out]

    def recover_routing(*_args, **_kwargs):
        call_order.append("routing")
        return routing

    def write_w13_delta(_context, *, gate_up_out, **_kwargs):
        gate_up_out.fill_(2)

    def write_w2_delta(_context, *, down_out, **_kwargs):
        down_out.fill_(3)

    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            side_effect=dynamic_quant,
        ),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", side_effect=base_w13_gmm, create=True),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activated),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=down_out),
        patch(f"{QUANT_MOE}._can_use_single_lora_gmm", return_value=False),
        patch(f"{QUANT_MOE}._can_use_composite_lora_gmm", return_value=False),
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather", side_effect=recover_routing) as recover_allgather,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_all2all", side_effect=recover_routing) as recover_all2all,
        patch(f"{QUANT_MOE}._can_use_ep_moe_lora_aux_stream", return_value=True),
        patch(f"{QUANT_MOE}._execute_moe_lora_in_parallel", side_effect=execute_parallel) as execute,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13", side_effect=write_w13_delta) as apply_w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2", side_effect=write_w2_delta) as apply_w2,
    ):
        extra_ctx.moe_comm_type = comm_type
        output, output_event = quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert output is down_out
    assert output_event is None
    assert execute.call_count == 2
    assert apply_w13.call_args.kwargs["gate_up_out"] is not gate_up_out
    assert apply_w2.call_args.kwargs["down_out"] is not down_out
    assert apply_w13.call_args.kwargs["gate_up_out"].data_ptr() == apply_w2.call_args.kwargs["down_out"].data_ptr()
    assert torch.equal(gate_up_out, torch.full_like(gate_up_out, 2))
    assert torch.equal(down_out, torch.full_like(down_out, 3))
    if comm_type == MoECommType.ALLGATHER:
        assert call_order == [
            "dynamic_quant",
            "execute",
            "routing",
            "base_w13",
            "execute",
            "dynamic_quant",
        ]
        recover_allgather.assert_called_once()
        recover_all2all.assert_not_called()
    else:
        assert call_order == [
            "routing",
            "execute",
            "dynamic_quant",
            "base_w13",
            "execute",
            "dynamic_quant",
        ]
        recover_all2all.assert_called_once()
        recover_allgather.assert_not_called()


def test_dynamic_int8_uses_single_lora_gmm_without_recovering_routing() -> None:
    lora_context = SimpleNamespace(
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        w13_lora_a_packed="w13_a_packed",
        w13_lora_b_packed="w13_b_packed",
        w2_lora_a_packed="w2_a_packed",
        w2_lora_b_packed="w2_b_packed",
        fully_sharded=False,
    )
    topk_scales = torch.tensor([[0.25], [0.5]], dtype=torch.bfloat16)
    routed_lora_slots = torch.tensor([0, -1], dtype=torch.long)
    mlp_input = _make_input(
        lora_context=lora_context,
        topk_scales=topk_scales,
        routed_lora_slots=routed_lora_slots,
    )
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.randn(2, 6, dtype=torch.bfloat16)
    activation_before_scale = torch.ones(2, 3, dtype=torch.bfloat16)
    quantized_activated = torch.ones(2, 3, dtype=torch.int8)
    activated_scale = torch.ones(2)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    single_routing = SimpleNamespace(enabled="enabled")
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            side_effect=[(quantized_input, input_scale), (quantized_activated, activated_scale)],
        ),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activation_before_scale),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=down_out),
        patch(f"{QUANT_MOE}._can_use_single_lora_gmm", return_value=True),
        patch(
            f"{QUANT_MOE}._build_single_lora_gmm_routing",
            return_value=single_routing,
        ) as build_routing,
        patch(f"{QUANT_MOE}._add_single_lora_gmm") as add_gmm,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather") as recover,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as apply_w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as apply_w2,
        patch(f"{QUANT_MOE}.reset_lora_indices") as reset_indices,
    ):
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert add_gmm.call_count == 2
    build_routing.assert_called_once_with(
        routed_lora_slots=routed_lora_slots,
    )
    assert add_gmm.call_args_list[0].args[0] is gate_up_out
    assert add_gmm.call_args_list[0].args[1] is mlp_input.hidden_states
    assert add_gmm.call_args_list[0].args[2] == "w13_a_packed"
    assert add_gmm.call_args_list[0].args[3] == "w13_b_packed"
    assert add_gmm.call_args_list[0].kwargs["routing"] is single_routing
    assert add_gmm.call_args_list[0].kwargs["group_list"] is mlp_input.group_list
    assert add_gmm.call_args_list[0].kwargs["fully_sharded"] is False
    assert torch.equal(add_gmm.call_args_list[1].args[1], topk_scales.expand(-1, 3))
    assert add_gmm.call_args_list[1].args[2] == "w2_a_packed"
    assert add_gmm.call_args_list[1].args[3] == "w2_b_packed"
    assert add_gmm.call_args_list[1].kwargs["routing"] is single_routing
    assert add_gmm.call_args_list[1].kwargs["group_list"] is mlp_input.group_list
    assert add_gmm.call_args_list[1].kwargs["fully_sharded"] is False
    assert add_gmm.call_args_list[1].kwargs["output_offset"] == 0
    recover.assert_not_called()
    apply_w13.assert_not_called()
    apply_w2.assert_not_called()
    reset_indices.assert_called_once_with(lora_context)


def test_dynamic_int8_allows_single_and_composite_lora_gmm_fast_paths_for_ep() -> None:
    lora_context = _make_gmm_lora_context(use_ep=True)
    expert_map = torch.tensor([-1, 0, 1, -1])
    group_list = torch.tensor([16, 16])
    hidden_states = torch.randn(32, 4, dtype=torch.bfloat16)
    routed_lora_slots = torch.zeros(32, dtype=torch.long)

    assert _can_use_single_lora_gmm(
        lora_context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
        expert_map=expert_map,
        routed_lora_slots=routed_lora_slots,
    )
    assert not _can_use_single_lora_gmm(
        lora_context,
        hidden_states=hidden_states[:-1],
        group_list=group_list,
        group_list_type=1,
        expert_map=expert_map,
        routed_lora_slots=routed_lora_slots[:-1],
    )
    assert not _can_use_single_lora_gmm(
        lora_context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
        routed_lora_slots=routed_lora_slots,
    )
    assert not _can_use_composite_lora_gmm(
        lora_context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
        expert_map=expert_map,
        routed_lora_slots=routed_lora_slots,
    )

    lora_context.fully_sharded = True
    assert not _can_use_single_lora_gmm(
        lora_context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
        expert_map=expert_map,
        routed_lora_slots=routed_lora_slots,
    )

    lora_context = _make_gmm_lora_context(use_ep=True)
    lora_context.punica_wrapper.num_active_moe_loras = 2
    composite_hidden_states = torch.randn(96, 4, dtype=torch.bfloat16)
    composite_lora_slots = torch.zeros(96, dtype=torch.long)
    assert _can_use_composite_lora_gmm(
        lora_context,
        hidden_states=composite_hidden_states,
        group_list=torch.tensor([48, 48]),
        group_list_type=1,
        expert_map=expert_map,
        routed_lora_slots=composite_lora_slots,
    )
    assert not _can_use_composite_lora_gmm(
        lora_context,
        hidden_states=composite_hidden_states,
        group_list=torch.tensor([48, 48]),
        group_list_type=1,
        routed_lora_slots=composite_lora_slots,
    )
    lora_context.fully_sharded = True
    assert not _can_use_composite_lora_gmm(
        lora_context,
        hidden_states=composite_hidden_states,
        group_list=torch.tensor([48, 48]),
        group_list_type=1,
        expert_map=expert_map,
        routed_lora_slots=composite_lora_slots,
    )


def test_dynamic_int8_uses_composite_gmm_without_recovering_routing() -> None:
    lora_context = SimpleNamespace(
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        fully_sharded=False,
    )
    routed_lora_slots = torch.tensor([0, 1], dtype=torch.long)
    mlp_input = _make_input(
        lora_context=lora_context,
        routed_lora_slots=routed_lora_slots,
    )
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.randn(2, 6, dtype=torch.bfloat16)
    activated = torch.ones(2, 3, dtype=torch.bfloat16)
    quantized_activated = torch.ones(2, 3, dtype=torch.int8)
    activated_scale = torch.ones(2)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    composite_routing = SimpleNamespace(group_ids="group_ids")
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            side_effect=[(quantized_input, input_scale), (quantized_activated, activated_scale)],
        ),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activated),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=down_out),
        patch(f"{QUANT_MOE}._can_use_single_lora_gmm", return_value=False),
        patch(f"{QUANT_MOE}._can_use_composite_lora_gmm", return_value=True),
        patch(
            f"{QUANT_MOE}._build_composite_lora_gmm_routing",
            return_value=composite_routing,
        ) as build_routing,
        patch(f"{QUANT_MOE}._add_composite_lora_gmm") as add_gmm,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather") as recover,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as apply_w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as apply_w2,
        patch(f"{QUANT_MOE}.reset_lora_indices") as reset_indices,
    ):
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    build_routing.assert_called_once_with(
        lora_context,
        routed_lora_slots=routed_lora_slots,
        group_list=mlp_input.group_list,
    )
    assert add_gmm.call_count == 2
    assert add_gmm.call_args_list[0].args[0] is gate_up_out
    assert add_gmm.call_args_list[0].args[1] is mlp_input.hidden_states
    assert add_gmm.call_args_list[0].kwargs["routing"] is composite_routing
    assert add_gmm.call_args_list[0].kwargs["fully_sharded"] is False
    assert add_gmm.call_args_list[1].args[0] is down_out
    assert add_gmm.call_args_list[1].args[1] is activated
    assert add_gmm.call_args_list[1].kwargs["routing"] is composite_routing
    assert add_gmm.call_args_list[1].kwargs["fully_sharded"] is False
    assert add_gmm.call_args_list[1].kwargs["output_offset"] == 0
    recover.assert_not_called()
    apply_w13.assert_not_called()
    apply_w2.assert_not_called()
    reset_indices.assert_called_once_with(lora_context)


@pytest.mark.parametrize(
    ("comm_type", "mlp_input", "message"),
    [
        (MoECommType.FUSED_MC2, _make_input(), "AllGather TP"),
        (MoECommType.ALLGATHER, _make_input(dynamic_eplb=True), "dynamic EPLB"),
    ],
)
def test_dynamic_int8_lora_rejects_unsupported_modes(comm_type, mlp_input, message) -> None:
    with patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx:
        extra_ctx.moe_comm_type = comm_type
        with pytest.raises(NotImplementedError, match=message):
            quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)


def test_dynamic_int8_all2all_lora_handles_empty_ep_rank() -> None:
    mlp_input = _make_input(
        hidden_states=torch.empty(0, 4, dtype=torch.bfloat16),
        group_list=torch.zeros(2, dtype=torch.int64),
        expanded_row_idx=None,
        topk_ids=None,
        lora_context=SimpleNamespace(use_ep=True),
    )

    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant") as dynamic_quant,
    ):
        extra_ctx.moe_comm_type = MoECommType.ALLTOALL
        output, output_event = quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert output is mlp_input.hidden_states
    assert output_event is None
    dynamic_quant.assert_not_called()


def test_registered_backend_requires_float_input() -> None:
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    validate_quant_moe_lora_activation_input(
        quant_type=QuantType.W8A8,
        hidden_states=hidden_states,
        dynamic_scale=None,
    )
    with pytest.raises(NotImplementedError, match="unquantized activations"):
        validate_quant_moe_lora_activation_input(
            quant_type=QuantType.W8A8,
            hidden_states=hidden_states.to(torch.int8),
            dynamic_scale=torch.ones(2),
        )


def test_unregistered_quantized_moe_lora_fails_fast() -> None:
    with pytest.raises(NotImplementedError, match="no implementation registered"):
        validate_quant_moe_lora_activation_input(
            quant_type=QuantType.W4A8,
            hidden_states=torch.randn(2, 4),
            dynamic_scale=None,
        )


def _make_gmm_lora_context(
    num_experts: int = 2,
    *,
    rank: int = 16,
    max_loras: int = 3,
    top_k: int = 6,
    fully_sharded: bool = False,
    tp_size: int = 1,
    tp_rank: int = 0,
    use_ep: bool = False,
):
    hidden_size = 4
    intermediate_size = 3
    w13_a_rank = rank // tp_size if fully_sharded else rank
    w2_b_output_size = hidden_size // tp_size if fully_sharded else hidden_size
    w13_a = (
        torch.zeros(max_loras, num_experts, w13_a_rank, hidden_size, dtype=torch.bfloat16),
        torch.zeros(max_loras, num_experts, w13_a_rank, hidden_size, dtype=torch.bfloat16),
    )
    w13_b = (
        torch.zeros(max_loras, num_experts, intermediate_size, rank, dtype=torch.bfloat16),
        torch.zeros(max_loras, num_experts, intermediate_size, rank, dtype=torch.bfloat16),
    )
    w2_a = (torch.zeros(max_loras, num_experts, rank, intermediate_size, dtype=torch.bfloat16),)
    w2_b = (torch.zeros(max_loras, num_experts, w2_b_output_size, rank, dtype=torch.bfloat16),)
    return SimpleNamespace(
        punica_wrapper=SimpleNamespace(
            is_prefill=True,
            num_active_moe_loras=1,
            active_moe_lora_slot=0,
            no_lora=False,
        ),
        adapter_enabled=torch.tensor([1] + [0] * max_loras, dtype=torch.int32),
        fully_sharded=fully_sharded,
        use_ep=use_ep,
        tp_size=tp_size,
        tp_rank=tp_rank,
        top_k=top_k,
        max_loras=max_loras,
        single_lora_cache_slot=0,
        w13_lora_a_stacked=w13_a,
        w13_lora_b_stacked=w13_b,
        w2_lora_a_stacked=w2_a,
        w2_lora_b_stacked=w2_b,
        w13_lora_a_packed=tuple(weight[0].clone() for weight in w13_a),
        w13_lora_b_packed=tuple(weight[0].clone() for weight in w13_b),
        w2_lora_a_packed=tuple(weight[0].clone() for weight in w2_a),
        w2_lora_b_packed=tuple(weight[0].clone() for weight in w2_b),
    )


def test_single_lora_gmm_checks_dynamic_fast_path_shape() -> None:
    context = _make_gmm_lora_context()
    # V1 sets LoRAMapping.is_prefill=True to select SGMV on non-CUDA
    # platforms, so it is not a real scheduler-phase signal. GMM eligibility
    # must remain valid when a caller supplies the actual decode value.
    context.punica_wrapper.is_prefill = False
    group_list = torch.tensor([8, 8], dtype=torch.int64)
    hidden_states = torch.zeros(16, 4, dtype=torch.bfloat16)
    routed_lora_slots = torch.zeros(16, dtype=torch.long)

    assert not _can_use_single_lora_gmm(
        context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
    )

    assert (
        _can_use_single_lora_gmm(
            context,
            hidden_states=hidden_states,
            group_list=group_list,
            group_list_type=1,
            routed_lora_slots=routed_lora_slots,
        )
        is True
    )
    context.single_lora_cache_slot = 1
    assert not _can_use_single_lora_gmm(
        context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
        routed_lora_slots=routed_lora_slots,
    )
    context.single_lora_cache_slot = 0
    assert (
        _can_use_single_lora_gmm(
            context,
            hidden_states=hidden_states[:15],
            group_list=group_list,
            group_list_type=1,
            routed_lora_slots=routed_lora_slots[:15],
        )
        is False
    )
    context = _make_gmm_lora_context(fully_sharded=True, tp_size=2, tp_rank=1)
    assert (
        _can_use_single_lora_gmm(
            context,
            hidden_states=hidden_states,
            group_list=group_list,
            group_list_type=1,
            routed_lora_slots=routed_lora_slots,
        )
        is True
    )


@pytest.mark.parametrize(
    ("top_k", "max_loras", "rank"),
    ((4, 2, 8), (8, 5, 32)),
)
def test_single_lora_gmm_accepts_dynamic_lora_configuration(
    top_k: int,
    max_loras: int,
    rank: int,
) -> None:
    context = _make_gmm_lora_context(
        top_k=top_k,
        max_loras=max_loras,
        rank=rank,
    )
    hidden_states = torch.zeros(16, 4, dtype=torch.bfloat16)
    routed_lora_slots = torch.zeros(16, dtype=torch.long)

    assert _can_use_single_lora_gmm(
        context,
        hidden_states=hidden_states,
        group_list=torch.tensor([8, 8], dtype=torch.int64),
        group_list_type=1,
        routed_lora_slots=routed_lora_slots,
    )


def test_composite_lora_gmm_checks_mixed_requests_and_minimum_rows() -> None:
    context = _make_gmm_lora_context()
    context.punica_wrapper.num_active_moe_loras = 2
    context.punica_wrapper.is_prefill = False
    group_list = torch.tensor([24, 24], dtype=torch.int64)
    hidden_states = torch.zeros(48, 4, dtype=torch.bfloat16)
    routed_lora_slots = torch.zeros(48, dtype=torch.long)

    assert _can_use_composite_lora_gmm(
        context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
        routed_lora_slots=routed_lora_slots,
    )
    assert not _can_use_composite_lora_gmm(
        context,
        hidden_states=hidden_states[:47],
        group_list=group_list,
        group_list_type=1,
        routed_lora_slots=routed_lora_slots[:47],
    )
    context.punica_wrapper.no_lora = True
    assert not _can_use_composite_lora_gmm(
        context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
        routed_lora_slots=routed_lora_slots,
    )

    context = _make_gmm_lora_context(fully_sharded=True, tp_size=2, tp_rank=1)
    context.punica_wrapper.num_active_moe_loras = 2
    assert _can_use_composite_lora_gmm(
        context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
        routed_lora_slots=routed_lora_slots,
    )


@pytest.mark.parametrize(
    ("top_k", "max_loras", "rank", "fully_sharded", "tp_size"),
    ((4, 2, 8, False, 1), (8, 5, 32, True, 2)),
)
def test_composite_lora_gmm_accepts_dynamic_lora_configuration(
    top_k: int,
    max_loras: int,
    rank: int,
    fully_sharded: bool,
    tp_size: int,
) -> None:
    context = _make_gmm_lora_context(
        top_k=top_k,
        max_loras=max_loras,
        rank=rank,
        fully_sharded=fully_sharded,
        tp_size=tp_size,
    )
    context.punica_wrapper.num_active_moe_loras = 2
    num_routed_rows = MOE_LORA_GMM_MIN_ROWS_PER_GROUP * max_loras * 2
    routed_lora_slots = torch.zeros(num_routed_rows, dtype=torch.long)

    assert _can_use_composite_lora_gmm(
        context,
        hidden_states=torch.zeros(num_routed_rows, 4, dtype=torch.bfloat16),
        group_list=torch.full((2,), num_routed_rows // 2, dtype=torch.int64),
        group_list_type=1,
        routed_lora_slots=routed_lora_slots,
    )


def test_build_single_lora_gmm_routing_handles_base_rows() -> None:
    routed_lora_slots = torch.tensor([2, -1, 2, -1], dtype=torch.long)

    routing = _build_single_lora_gmm_routing(
        routed_lora_slots=routed_lora_slots,
    )

    assert torch.equal(routing.enabled, torch.tensor([True, False, True, False]))


def test_build_single_lora_gmm_routing_ignores_nonlocal_ep_rows() -> None:
    # init-routing sideband has already marked the non-local tail as -1.
    routed_lora_slots = torch.tensor([1, -1, -1, -1], dtype=torch.long)

    routing = _build_single_lora_gmm_routing(
        routed_lora_slots=routed_lora_slots,
    )

    assert torch.equal(routing.enabled, torch.tensor([True, False, False, False]))


def test_build_composite_lora_gmm_routing_handles_multiple_slots_and_base() -> None:
    context = SimpleNamespace(
        adapter_enabled=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
        w13_lora_a_stacked=(torch.empty(3, 2, 16, 4),),
    )
    routed_lora_slots = torch.tensor([2, -1, 0, 2, -1, 0])
    group_list = torch.tensor([3, 3], dtype=torch.int64)

    routing = _build_composite_lora_gmm_routing(
        context,
        routed_lora_slots=routed_lora_slots,
        group_list=group_list,
    )

    assert torch.equal(routing.group_ids, torch.tensor([4, 6, 0, 5, 6, 1], dtype=torch.int32))
    assert torch.equal(routing.group_list, torch.tensor([1, 1, 0, 0, 1, 1]))
    assert torch.equal(routing.enabled, torch.tensor([True, False, True, True, False, True]))


def test_build_composite_lora_gmm_routing_moves_nonlocal_ep_tail_to_sentinel() -> None:
    context = SimpleNamespace(
        adapter_enabled=torch.tensor([1, 1, 0], dtype=torch.int32),
        w13_lora_a_stacked=(torch.empty(3, 1, 16, 4),),
    )

    routing = _build_composite_lora_gmm_routing(
        context,
        routed_lora_slots=torch.tensor([-1, 1, -1, -1, -1, -1]),
        group_list=torch.tensor([2], dtype=torch.int64),
    )

    assert torch.equal(routing.group_ids, torch.tensor([3, 1, 3, 3, 3, 3], dtype=torch.int32))
    assert torch.equal(routing.group_list, torch.tensor([0, 1, 0]))
    assert torch.equal(routing.enabled, torch.tensor([False, True, False, False, False, False]))


def test_composite_lora_gmm_reorders_masks_and_restores_rows() -> None:
    output = torch.zeros(3, 2)
    inputs = torch.tensor([[1.0], [2.0], [3.0]])
    lora_a = (torch.zeros(3, 2, 16, 1),)
    lora_b = (torch.zeros(3, 2, 2, 16),)
    routing = _CompositeLoraGMMRouting(
        group_ids=torch.tensor([6, 5, 0], dtype=torch.int32),
        group_list=torch.tensor([1, 0, 0, 0, 0, 1]),
        enabled=torch.tensor([False, True, True]),
    )
    grouped_inputs = torch.tensor([[3.0], [2.0], [1.0]])
    reverse_mapping = torch.tensor([1, 2, 0], dtype=torch.int32)
    shrink = torch.zeros(3, 16)
    delta = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    restored_delta = torch.tensor([[20.0, 21.0], [30.0, 31.0], [10.0, 11.0]])

    with (
        patch(
            f"{QUANT_MOE}.torch_npu.npu_moe_token_permute",
            return_value=(grouped_inputs, reverse_mapping),
            create=True,
        ) as token_permute,
        patch(
            f"{QUANT_MOE}.torch_npu.npu_moe_token_unpermute",
            return_value=restored_delta,
            create=True,
        ) as token_unpermute,
        patch(
            f"{QUANT_MOE}._grouped_lora_matmul",
            side_effect=[shrink, delta],
        ) as grouped_matmul,
    ):
        _add_composite_lora_gmm(
            output,
            inputs,
            lora_a,
            lora_b,
            routing=routing,
        )

    token_permute.assert_called_once()
    assert token_permute.call_args.kwargs["tokens"] is inputs
    assert token_permute.call_args.kwargs["indices"] is routing.group_ids
    assert token_permute.call_args.kwargs["num_out_tokens"] == 3
    token_unpermute.assert_called_once_with(
        permuted_tokens=delta,
        sorted_indices=reverse_mapping,
    )
    assert grouped_matmul.call_args_list[0].args[0] is grouped_inputs
    assert grouped_matmul.call_args_list[0].args[1].shape == (6, 16, 1)
    assert grouped_matmul.call_args_list[1].args[1].shape == (6, 2, 16)
    assert torch.equal(output, torch.tensor([[0.0, 0.0], [30.0, 31.0], [10.0, 11.0]]))


def test_single_lora_gmm_uses_packed_weights_and_adds_each_output_slice() -> None:
    output = torch.zeros(2, 5, dtype=torch.bfloat16)
    inputs = torch.ones(2, 4, dtype=torch.bfloat16)
    lora_a = tuple(torch.zeros(shape, dtype=torch.bfloat16) for shape in ((2, 16, 4), (2, 16, 4)))
    lora_b = tuple(torch.zeros(shape, dtype=torch.bfloat16) for shape in ((2, 2, 16), (2, 3, 16)))
    shrink = torch.zeros(2, 16, dtype=torch.bfloat16)
    group_list = torch.ones(2, dtype=torch.int64)
    routing = _SingleLoraGMMRouting(
        enabled=torch.tensor([True, False]),
    )

    with (
        patch(
            f"{QUANT_MOE}._grouped_lora_matmul",
            side_effect=[
                shrink,
                torch.tensor([[1, 1], [0, 0]], dtype=torch.bfloat16),
                shrink,
                torch.tensor([[2, 2, 2], [0, 0, 0]], dtype=torch.bfloat16),
            ],
        ) as grouped_matmul,
        patch(
            f"{QUANT_MOE}.torch.index_select",
            side_effect=AssertionError("single-LoRA GMM must not gather weights"),
        ) as index_select,
    ):
        _add_single_lora_gmm(
            output,
            inputs,
            lora_a,
            lora_b,
            routing=routing,
            group_list=group_list,
        )

    assert grouped_matmul.call_count == 4
    index_select.assert_not_called()
    assert torch.equal(
        grouped_matmul.call_args_list[0].args[0],
        torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bfloat16),
    )
    for call in grouped_matmul.call_args_list:
        assert call.args[1].shape[0] == 2
        assert call.args[2] is group_list
    assert torch.equal(output[0, :2], torch.ones(2, dtype=torch.bfloat16))
    assert torch.equal(output[0, 2:], torch.full((3,), 2, dtype=torch.bfloat16))
    assert torch.equal(output[1], torch.zeros(5, dtype=torch.bfloat16))


def test_single_lora_gmm_fully_sharded_gathers_rank_and_uses_output_offset() -> None:
    output = torch.zeros(2, 6, dtype=torch.bfloat16)
    inputs = torch.ones(2, 4, dtype=torch.bfloat16)
    lora_a = (torch.zeros(2, 8, 4, dtype=torch.bfloat16),)
    lora_b = (torch.zeros(2, 2, 16, dtype=torch.bfloat16),)
    group_list = torch.ones(2, dtype=torch.int64)
    routing = _SingleLoraGMMRouting(
        enabled=torch.tensor([True, True]),
    )
    local_shrink = torch.ones(2, 8, dtype=torch.bfloat16)
    gathered_shrink = torch.ones(2, 16, dtype=torch.bfloat16)
    delta = torch.full((2, 2), 3, dtype=torch.bfloat16)

    with (
        patch(
            f"{QUANT_MOE}._grouped_lora_matmul",
            side_effect=[local_shrink, delta],
        ) as grouped_matmul,
        patch(
            f"{QUANT_MOE}.tensor_model_parallel_all_gather",
            return_value=gathered_shrink,
        ) as all_gather,
        patch(f"{QUANT_MOE}.tensor_model_parallel_all_reduce") as all_reduce,
    ):
        _add_single_lora_gmm(
            output,
            inputs,
            lora_a,
            lora_b,
            routing=routing,
            group_list=group_list,
            fully_sharded=True,
            output_offset=3,
        )

    all_gather.assert_called_once_with(local_shrink)
    all_reduce.assert_not_called()
    assert grouped_matmul.call_args_list[1].args[0] is gathered_shrink
    assert torch.equal(output[:, :3], torch.zeros(2, 3, dtype=torch.bfloat16))
    assert torch.equal(output[:, 3:5], delta)
    assert torch.equal(output[:, 5], torch.zeros(2, dtype=torch.bfloat16))


def test_composite_lora_gmm_fully_sharded_reduces_partial_rank() -> None:
    output = torch.zeros(2, 4, dtype=torch.bfloat16)
    inputs = torch.ones(2, 3, dtype=torch.bfloat16)
    lora_a = (torch.zeros(3, 2, 16, 3, dtype=torch.bfloat16),)
    lora_b = (torch.zeros(3, 2, 2, 16, dtype=torch.bfloat16),)
    routing = _CompositeLoraGMMRouting(
        group_ids=torch.tensor([0, 1], dtype=torch.int32),
        group_list=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.int64),
        enabled=torch.tensor([True, True]),
    )
    local_shrink = torch.ones(2, 16, dtype=torch.bfloat16)
    reduced_shrink = torch.full((2, 16), 2, dtype=torch.bfloat16)
    delta = torch.full((2, 2), 5, dtype=torch.bfloat16)
    reverse_mapping = torch.tensor([0, 1], dtype=torch.int32)

    with (
        patch(
            f"{QUANT_MOE}.torch_npu.npu_moe_token_permute",
            return_value=(inputs, reverse_mapping),
            create=True,
        ),
        patch(
            f"{QUANT_MOE}.torch_npu.npu_moe_token_unpermute",
            return_value=delta,
            create=True,
        ),
        patch(
            f"{QUANT_MOE}._grouped_lora_matmul",
            side_effect=[local_shrink, delta],
        ) as grouped_matmul,
        patch(f"{QUANT_MOE}.tensor_model_parallel_all_gather") as all_gather,
        patch(
            f"{QUANT_MOE}.tensor_model_parallel_all_reduce",
            return_value=reduced_shrink,
        ) as all_reduce,
    ):
        _add_composite_lora_gmm(
            output,
            inputs,
            lora_a,
            lora_b,
            routing=routing,
            fully_sharded=True,
            output_offset=2,
        )

    all_reduce.assert_called_once_with(local_shrink)
    all_gather.assert_not_called()
    assert grouped_matmul.call_args_list[1].args[0] is reduced_shrink
    assert torch.equal(output[:, :2], torch.zeros(2, 2, dtype=torch.bfloat16))
    assert torch.equal(output[:, 2:], delta)
