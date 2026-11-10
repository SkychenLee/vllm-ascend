# SPDX-License-Identifier: Apache-2.0
"""CPU behavior checks for opt-in attention projection LoRA diagnostics."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from vllm_ascend.lora import debug
from vllm_ascend.lora import utils as lora_utils

PREFIX = "language_model.model.layers.0.self_attn.q_b_proj"


@pytest.fixture
def debug_runtime(monkeypatch):
    context = SimpleNamespace(
        capturing=False,
        in_profile_run=False,
        attn_metadata=SimpleNamespace(num_actual_tokens=3, num_prefills=1, num_decode_tokens=0),
    )
    logger = Mock()
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))
    monkeypatch.setattr(debug, "envs", SimpleNamespace(VLLM_ASCEND_LORA_DEBUG=True))
    monkeypatch.setattr(debug, "logger", logger)
    monkeypatch.setattr(debug, "get_current_vllm_config", Mock(return_value=config))
    monkeypatch.setattr(debug, "is_forward_context_available", Mock(return_value=True))
    monkeypatch.setattr(debug, "get_forward_context", Mock(return_value=context))
    return SimpleNamespace(context=context, logger=logger, config=config)


def _layer(prefix=PREFIX, tp_rank=0, columns=200):
    return SimpleNamespace(
        base_layer=SimpleNamespace(prefix=prefix, weight=torch.ones(columns, 4, dtype=torch.bfloat16)),
        tp_rank=tp_rank,
        lora_a_stacked=(torch.ones(1, 1, 2, 4, dtype=torch.bfloat16),),
        lora_b_stacked=(torch.ones(1, 1, columns, 2, dtype=torch.bfloat16),),
        lora_b_packed=torch.ones(1, 1, 2, columns, dtype=torch.bfloat16),
        output_slices=(columns,),
        punica_wrapper=SimpleNamespace(
            _ascend_lora_debug_enabled=True,
            _ascend_lora_debug_mapping=(0, 17, 17),
            _ascend_lora_debug_slots=(17,),
            no_lora=False,
        ),
    )


def _logs(logger):
    return [call.args[0] % call.args[1:] for call in logger.info.call_args_list]


def test_disabled_debug_never_reads_context_clones_or_logs(debug_runtime, monkeypatch):
    debug.envs.VLLM_ASCEND_LORA_DEBUG = False
    wrapper = SimpleNamespace()
    debug.configure_lora_debug(wrapper)
    debug.update_lora_debug_metadata(wrapper, None, None)
    layer = _layer()
    debug.initialize_linear_lora_debug(layer)
    assert layer._ascend_lora_debug is None
    assert not hasattr(wrapper, "_ascend_lora_debug_mapping")
    debug.get_current_vllm_config.assert_not_called()
    monkeypatch.setattr(lora_utils, "current_platform", SimpleNamespace(can_update_inplace=lambda: True))
    layer.punica_wrapper.add_lora_linear = Mock(side_effect=lambda output, *_a, **_kw: output.add_(1))
    output = torch.zeros(3, 200, dtype=torch.bfloat16)
    with patch.object(torch.Tensor, "clone", side_effect=AssertionError("debug cloned while disabled")):
        actual = lora_utils._apply_packed_lora(layer, torch.ones(3, 4, dtype=torch.bfloat16), output)
    assert actual is output
    assert torch.all(actual == 1)
    layer.punica_wrapper.add_lora_linear.assert_called_once()
    debug.get_forward_context.assert_not_called()
    debug_runtime.logger.info.assert_not_called()
    debug_runtime.logger.warning_once.assert_not_called()


@pytest.mark.parametrize("eager,single_slot,match", [(False, True, "enforce-eager"), (True, False, "max-loras 1")])
def test_debug_requires_eager_and_supported_single_slot(debug_runtime, eager, single_slot, match):
    debug_runtime.config.model_config.enforce_eager = eager
    with pytest.raises(ValueError, match=match):
        debug.configure_lora_debug(SimpleNamespace(_single_lora_slot=single_slot))


def test_debug_configuration_and_metadata_copy_use_host_values(debug_runtime):
    wrapper = SimpleNamespace(_single_lora_slot=True)
    debug.configure_lora_debug(wrapper)
    mapping = SimpleNamespace(index_mapping=[17, 0, 17], is_prefill=True)
    slots = [17]
    debug.update_lora_debug_metadata(wrapper, mapping, slots)
    mapping.index_mapping[0] = 99
    slots[0] = 99
    assert wrapper._ascend_lora_debug_mapping == (17, 0, 17)
    assert wrapper._ascend_lora_debug_slots == (17,)
    debug_runtime.logger.warning_once.assert_called_once()


@pytest.mark.parametrize(
    "name,tp_rank,enabled",
    [
        ("q_b_proj", 0, True),
        ("kv_b_proj", 0, True),
        ("o_proj", 0, True),
        ("q_b_proj", 1, False),
        ("down_proj", 0, False),
    ],
)
def test_debug_initialization_filters_targets_and_tp_rank(debug_runtime, name, tp_rank, enabled):
    layer = _layer(PREFIX.rsplit(".", 1)[0] + "." + name, tp_rank)
    debug.initialize_linear_lora_debug(layer)
    assert (layer._ascend_lora_debug is not None) is enabled
    assert debug_runtime.logger.info.call_count == int(enabled)


@pytest.mark.parametrize(
    "prefills,decode_tokens,phase", [(1, 0, "prefill"), (0, 3, "decode"), (1, 1, "mixed"), (0, 0, "unknown")]
)
@pytest.mark.parametrize("dictionary_metadata", [False, True])
def test_debug_phase_comes_from_attention_metadata(debug_runtime, prefills, decode_tokens, phase, dictionary_metadata):
    layer = _layer()
    layer.punica_wrapper.is_prefill = True  # Not the real execution phase.
    metadata = SimpleNamespace(num_actual_tokens=3, num_prefills=prefills, num_decode_tokens=decode_tokens)
    debug_runtime.context.attn_metadata = (
        {PREFIX.rsplit(".", 1)[0] + ".attn": metadata} if dictionary_metadata else metadata
    )
    state = debug.LinearLoRADebug(PREFIX, 0)
    sample = state.before(layer, torch.ones(3, 4), torch.zeros(3, 200))
    assert sample[0] == phase
    assert sample[2:4] == (1, 128)
    assert sample[-1].shape == (128,)
    assert f"phase={phase}" in _logs(debug_runtime.logger)[-1]


@pytest.mark.parametrize("reason", ["no_context", "capturing", "in_profile_run", "no_metadata", "empty"])
def test_debug_skips_capture_profile_and_empty_calls(debug_runtime, reason):
    layer = _layer()
    state = debug.LinearLoRADebug(PREFIX, 0)
    output = torch.zeros(3, 200)
    if reason == "no_context":
        debug.is_forward_context_available.return_value = False
    elif reason == "no_metadata":
        debug_runtime.context.attn_metadata = None
    elif reason == "empty":
        output = torch.empty(0, 200)
    else:
        setattr(debug_runtime.context, reason, True)
    with patch.object(torch.Tensor, "clone", side_effect=AssertionError("invalid call sampled")):
        assert state.before(layer, torch.ones(3, 4), output) is None
    assert not state.seen
    debug_runtime.logger.info.assert_not_called()


@pytest.mark.parametrize("actual,mapping", [(4, (17,)), (4, (17, 17, 17, 17)), (None, (17,)), (0, ())])
def test_debug_rejects_unattributable_token_mapping(debug_runtime, actual, mapping):
    layer = _layer()
    layer.punica_wrapper._ascend_lora_debug_mapping = mapping
    debug_runtime.context.attn_metadata.num_actual_tokens = actual
    state = debug.LinearLoRADebug(PREFIX, 0)
    assert state.before(layer, torch.ones(3, 4), torch.zeros(3, 200)) is None
    assert not state.seen
    debug_runtime.logger.warning_once.assert_called_once()


@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_debug_distinguishes_active_mapping_and_loaded_weights(debug_runtime, loaded, active):
    layer = _layer()
    if not active:
        layer.punica_wrapper._ascend_lora_debug_mapping = (0, 0, 0)
        layer.punica_wrapper.no_lora = True
    state = debug.LinearLoRADebug(PREFIX, 0)
    if loaded:
        state.set_slot(0, loaded=True)
    state.before(layer, torch.ones(3, 4), torch.zeros(3, 200))
    log = _logs(debug_runtime.logger)[-1]
    assert f"active_tokens={2 if active else 0}/3" in log
    assert f"active_slots={'(0,)' if active else '()'}" in log
    assert f"loaded_slots={'(0,)' if loaded and active else '()'}" in log
    assert f"no_lora={not active}" in log


@pytest.mark.parametrize("delta", [0.0, 0.5])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("batch_dimension", [False, True])
def test_packed_apply_logs_observed_delta_without_changing_output(
    debug_runtime, monkeypatch, delta, inplace, batch_dimension
):
    layer = _layer()
    debug.initialize_linear_lora_debug(layer)
    layer._ascend_lora_debug.set_slot(0, loaded=True)
    monkeypatch.setattr(lora_utils, "current_platform", SimpleNamespace(can_update_inplace=lambda: inplace))

    def apply(output, x, *_args, **_kwargs):
        assert x.shape == (3, 4)
        return output.add_(delta) if inplace else output + delta

    layer.punica_wrapper.add_lora_linear = Mock(side_effect=apply)
    shape = (1, 3, 200) if batch_dimension else (3, 200)
    xshape = (1, 3, 4) if batch_dimension else (3, 4)
    output = torch.ones(shape, dtype=torch.bfloat16)
    expected = output + delta
    actual = lora_utils._apply_packed_lora(layer, torch.ones(xshape, dtype=torch.bfloat16), output)
    layer.punica_wrapper.add_lora_linear.assert_called_once()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    done = [line for line in _logs(debug_runtime.logger) if "event=apply_done" in line]
    assert len(done) == 1
    assert f"sample_local_delta_abs_max={delta:g}" in done[0]


def test_debug_deduplicates_by_phase_adapter_and_resets_on_slot_changes(debug_runtime):
    layer = _layer()
    state = debug.LinearLoRADebug(PREFIX, 0)
    x, output = torch.ones(3, 4), torch.zeros(3, 200)
    state.set_slot(0, loaded=True)
    assert state.before(layer, x, output) is not None
    assert state.before(layer, x, output) is None
    debug_runtime.context.attn_metadata.num_prefills = 0
    debug_runtime.context.attn_metadata.num_decode_tokens = 3
    assert state.before(layer, x, output) is not None
    layer.punica_wrapper._ascend_lora_debug_mapping = (0, 23, 23)
    layer.punica_wrapper._ascend_lora_debug_slots = (23,)
    assert state.before(layer, x, output) is not None
    state.set_slot(0, loaded=False)
    assert state.loaded_slots == set()
    assert state.before(layer, x, output) is not None
    state.set_slot(0, loaded=True)
    assert state.before(layer, x, output) is not None


def test_real_transposed_mixin_set_reset_updates_debug_state(debug_runtime):
    layer = lora_utils.AscendColumnParallelLinearWithLoRA.__new__(lora_utils.AscendColumnParallelLinearWithLoRA)
    nn.Module.__init__(layer)
    layer.tp_size = layer.n_slices = 1
    layer.lora_a_stacked = (torch.zeros(1, 1, 2, 4),)
    layer.lora_b_stacked = (torch.zeros(1, 1, 6, 2),)
    layer.lora_b_packed = torch.zeros(1, 1, 2, 6)
    layer._ascend_lora_debug = debug.LinearLoRADebug(PREFIX, 0)
    layer.set_lora(0, torch.ones(2, 4), torch.ones(6, 2))
    assert layer._ascend_lora_debug.loaded_slots == {0}
    assert torch.all(layer.lora_b_packed == 1)
    layer._ascend_lora_debug.seen.add(("decode", (17,), (0,)))
    layer.reset_lora(0)
    assert not layer._ascend_lora_debug.loaded_slots
    assert not layer._ascend_lora_debug.seen
    assert torch.count_nonzero(layer.lora_b_packed) == 0
