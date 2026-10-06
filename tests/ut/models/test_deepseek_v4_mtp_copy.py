# SPDX-License-Identifier: Apache-2.0
"""Exercise actual MTP write/forward methods without model weights or NPUs."""

import ast
import sys
from itertools import islice
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

MODEL_PATH = Path(__file__).parents[3] / "vllm_ascend/models/deepseek_v4.py"


@pytest.fixture
def methods(monkeypatch):
    # Avoid importing the model's attention kernels and constructing weights.
    # Both methods below are executed directly from the production source.
    tree = ast.parse(MODEL_PATH.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4Model")
    selected = [
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_copy_mtp_hidden_states", "forward"}
    ]

    def scatter(buffer, rows, source):
        buffer.index_copy_(0, rows[:, 0], source)
        return buffer

    npu = SimpleNamespace(npu_scatter_nd_update_=Mock(side_effect=scatter))
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    gathered = Mock()
    namespace = {
        "torch": torch,
        "torch_npu": npu,
        "islice": islice,
        "IntermediateTensors": dict,
        "get_pp_group": lambda: pp_group,
        "tensor_model_parallel_all_gather": gathered,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MODEL_PATH), "exec"), namespace)
    context_module = ModuleType("vllm_ascend.ascend_forward_context")
    context_module.get_forward_context = Mock(return_value=None)
    monkeypatch.setitem(sys.modules, context_module.__name__, context_module)
    return namespace, npu, gathered, context_module


def state_for(methods, dtype=torch.bfloat16, enabled=True):
    namespace, _, _, _ = methods
    state = SimpleNamespace(
        _mtp_hidden_buffer=torch.full((16, 12), 5, dtype=dtype),
        _mtp_buffer_row_indices=torch.arange(16, dtype=torch.int64)[:, None] if enabled else None,
        layers=[],
        start_layer=0,
        end_layer=0,
        aux_hidden_state_layers=set(),
    )
    state._copy_mtp_hidden_states = MethodType(namespace["_copy_mtp_hidden_states"], state)
    return state


def initialize_mtp_buffer(enabled, method, runner_v2):
    """Execute the real constructor's buffer setup with a CPU-only model shell."""
    tree = ast.parse(MODEL_PATH.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4Model")
    initializer = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    selected = [
        node
        for node in initializer.body
        if isinstance(node, ast.Assign)
        and any(
            (isinstance(target, ast.Name) and target.id in {"speculative_config", "skip_mtp_buffer"})
            or (isinstance(target, ast.Attribute) and target.attr in {"_mtp_hidden_buffer", "_mtp_buffer_row_indices"})
            for target in node.targets
        )
    ]
    allocations = SimpleNamespace(empty=Mock(wraps=torch.empty), arange=Mock(wraps=torch.arange), int64=torch.int64)
    config = SimpleNamespace(
        speculative_config=None if method is None else SimpleNamespace(use_dspark=lambda: method == "dspark"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    state = SimpleNamespace(device="cpu")
    namespace = {
        "self": state,
        "torch": allocations,
        "vllm_config": config,
        "hc_dim": 12,
        "envs": SimpleNamespace(VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY=enabled),
        "vllm_envs": SimpleNamespace(VLLM_USE_V2_MODEL_RUNNER=runner_v2),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MODEL_PATH), "exec"), namespace)
    return state, allocations


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method", [None, "mtp", "dspark", "eagle3"])
@pytest.mark.parametrize("runner_v2", [False, True, None])
def test_only_explicit_v1_dspark_skips_mtp_allocations(enabled, method, runner_v2):
    state, allocations = initialize_mtp_buffer(enabled, method, runner_v2)
    skip = enabled and method == "dspark" and runner_v2 is False
    if skip:
        assert state._mtp_hidden_buffer is None
        assert state._mtp_buffer_row_indices is None
        allocations.empty.assert_not_called()
        allocations.arange.assert_not_called()
    else:
        assert state._mtp_hidden_buffer.shape == (32, 12)
        allocations.empty.assert_called_once()
        assert allocations.arange.call_count == int(enabled)
        if enabled:
            assert torch.equal(state._mtp_buffer_row_indices, torch.arange(32)[:, None])
        else:
            assert state._mtp_buffer_row_indices is None


@pytest.mark.parametrize("enabled", [False, True])
def test_model_initializer_allocates_persistent_indices_only_when_enabled(enabled):
    tree = ast.parse(MODEL_PATH.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4Model")
    initializer = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignment = next(
        node
        for node in initializer.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_mtp_buffer_row_indices" for target in node.targets
        )
    )
    state = SimpleNamespace(_mtp_hidden_buffer=torch.empty(16, 12), device="cpu")
    namespace = {"self": state, "torch": torch, "envs": SimpleNamespace(VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY=enabled)}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(MODEL_PATH), "exec"), namespace)
    if enabled:
        assert state._mtp_buffer_row_indices.dtype == torch.int64
        assert state._mtp_buffer_row_indices.is_contiguous()
        assert torch.equal(state._mtp_buffer_row_indices, torch.arange(16)[:, None])
    else:
        assert state._mtp_buffer_row_indices is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enabled", [False, True])
def test_refresh_changed_rows_preserves_alias_and_tail(methods, dtype, enabled):
    state = state_for(methods, dtype, enabled)
    alias = state._mtp_hidden_buffer
    ptr = alias.data_ptr()
    source = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12).to(dtype)
    for offset in (1, -3, 7):
        source.add_(offset)
        assert state._copy_mtp_hidden_states(source) is None
        assert state._mtp_hidden_buffer is alias
        assert alias.data_ptr() == ptr
        assert torch.equal(alias[:8], source)
        assert torch.equal(alias[8:], torch.full_like(alias[8:], 5))
    scatter = methods[1].npu_scatter_nd_update_
    assert scatter.call_count == (3 if enabled else 0)
    if enabled:
        assert scatter.call_args.args[0] is alias
        assert torch.equal(scatter.call_args.args[1], torch.arange(8)[:, None])


@pytest.mark.parametrize("enabled", [False, True])
def test_zero_rows_skip_native_write(methods, enabled):
    state = state_for(methods, enabled=enabled)
    original = state._mtp_hidden_buffer.clone()
    state._copy_mtp_hidden_states(original[:0])
    methods[1].npu_scatter_nd_update_.assert_not_called()
    assert torch.equal(state._mtp_hidden_buffer, original)


@pytest.mark.parametrize("shape, message", [((17, 12), "capacity"), ((8, 11), "width"), ((8, 3, 4), "width")])
@pytest.mark.parametrize("enabled", [False, True])
def test_invalid_write_fails_before_buffer_mutation(methods, shape, message, enabled):
    state = state_for(methods, enabled=enabled)
    before = state._mtp_hidden_buffer.clone()
    with pytest.raises(ValueError, match=message):
        state._copy_mtp_hidden_states(torch.zeros(shape, dtype=before.dtype))
    methods[1].npu_scatter_nd_update_.assert_not_called()
    assert torch.equal(state._mtp_hidden_buffer, before)


def test_mixed_dtype_preserves_legacy_conversion(methods):
    state = state_for(methods, dtype=torch.bfloat16)
    source = torch.randn(8, 12, dtype=torch.float32)
    state._copy_mtp_hidden_states(source)
    assert torch.equal(state._mtp_hidden_buffer[:8], source.to(torch.bfloat16))
    methods[1].npu_scatter_nd_update_.assert_not_called()


def test_full_capacity_write_updates_every_row(methods):
    state = state_for(methods)
    source = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12).to(torch.bfloat16)
    state._copy_mtp_hidden_states(source)
    assert torch.equal(state._mtp_hidden_buffer, source)
    assert methods[1].npu_scatter_nd_update_.call_args.args[1].shape == (16, 1)


@pytest.mark.parametrize("sp, padding", [(False, 0), (True, 0), (True, 2)])
@pytest.mark.parametrize("enabled", [False, True])
def test_forward_refreshes_through_same_helper_after_sp_gather(methods, sp, padding, enabled):
    namespace, _, gather, context = methods
    state = state_for(methods, enabled=enabled)
    states = torch.arange(4 * 3 * 4, dtype=torch.float32).reshape(4, 3, 4).to(torch.bfloat16)
    if sp:
        context.get_forward_context.return_value = SimpleNamespace(flash_comm_v1_enabled=True, pad_size=padding)
        gather.return_value = torch.cat((states.flatten(1), states.flatten(1) + 100), dim=0)
        expected = gather.return_value[:-padding] if padding else gather.return_value
    else:
        expected = states.flatten(1)
    original = state._copy_mtp_hidden_states
    state._copy_mtp_hidden_states = Mock(side_effect=original)
    output = namespace["forward"](state, None, torch.arange(4), {"hidden_states": states})
    state._copy_mtp_hidden_states.assert_called_once()
    assert torch.equal(state._copy_mtp_hidden_states.call_args.args[0], expected)
    assert torch.equal(state._mtp_hidden_buffer[: expected.shape[0]], expected)
    assert torch.equal(
        state._mtp_hidden_buffer[expected.shape[0] :], torch.full_like(state._mtp_hidden_buffer[expected.shape[0] :], 5)
    )
    assert output["hidden_states"] is states
    assert gather.call_count == int(sp)
    if sp:
        assert gather.call_args.kwargs == {"dim": 0}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method", [None, "mtp", "dspark"])
@pytest.mark.parametrize("runner_v2", [False, True, None])
@pytest.mark.parametrize("tp_size", [1, 8])
@pytest.mark.parametrize("sp", [False, True])
def test_speculative_mode_preserves_aux_and_target_output(methods, enabled, method, runner_v2, tp_size, sp):
    namespace, _, gather, context = methods
    state, _ = initialize_mtp_buffer(enabled, method, runner_v2)
    source = torch.arange(3 * 3 * 4, dtype=torch.float32).reshape(3, 3, 4).to(torch.bfloat16)
    expected_hidden = source + 1

    class Layer:
        layer_idx = 0

        def __call__(self, positions, states, *args):
            return states + 1, None

    state.layers = [Layer()]
    state.start_layer, state.end_layer = 0, 1
    state.aux_hidden_state_layers = {1}
    state.hc_head_fn = state.hc_head_scale = state.hc_head_base = None
    state.hc_head = Mock(side_effect=lambda hidden, *args: hidden.mean(dim=1))
    state.norm = Mock(side_effect=lambda hidden: hidden + 2)
    state._copy_mtp_hidden_states = Mock(side_effect=MethodType(namespace["_copy_mtp_hidden_states"], state))
    namespace["get_pp_group"]().is_last_rank = True
    if sp:
        context.get_forward_context.return_value = SimpleNamespace(flash_comm_v1_enabled=True, pad_size=0)
        gather.return_value = torch.cat([expected_hidden.flatten(1)] * tp_size, dim=0)

    output, aux = namespace["forward"](state, None, torch.arange(3), {"hidden_states": source})
    assert torch.equal(output, expected_hidden.mean(dim=1) + 2)
    assert len(aux) == 1 and torch.equal(aux[0], expected_hidden.mean(dim=1))
    assert torch.equal(state.hc_head.call_args.args[0], expected_hidden)
    skip = enabled and method == "dspark" and runner_v2 is False
    if skip:
        context.get_forward_context.assert_not_called()
        gather.assert_not_called()
        state._copy_mtp_hidden_states.assert_not_called()
    else:
        expected_stash = gather.return_value if sp else expected_hidden.flatten(1)
        state._copy_mtp_hidden_states.assert_called_once()
        assert torch.equal(state._mtp_hidden_buffer[: expected_stash.shape[0]], expected_stash)
        assert gather.call_count == int(sp)


def test_disabled_buffer_helper_does_not_write(methods):
    state = state_for(methods)
    state._mtp_hidden_buffer = state._mtp_buffer_row_indices = None
    assert state._copy_mtp_hidden_states(torch.zeros(3, 12)) is None
    methods[1].npu_scatter_nd_update_.assert_not_called()
