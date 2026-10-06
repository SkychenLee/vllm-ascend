# SPDX-License-Identifier: Apache-2.0
"""CPU/fake checks of the bounded V1 DSpark Cube Prefill candidate."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.runner import moe_runner

from tests.ut.lora.test_quant_moe import QUANT_MOE, _make_input
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.lora.cube_moe import CubeW13State, static_config_supported, supports_cube_target_prefill_context
from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoELoRARouting
from vllm_ascend.quantization.quant_type import QuantType


def configuration(method="dspark", tokens=5, captures=(6, 12, 24, 48), eager=False):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=16384),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=captures,
        ),
        speculative_config=(
            None
            if method is None
            else SimpleNamespace(use_dspark=lambda: method == "dspark", num_speculative_tokens=tokens)
        ),
        model_config=SimpleNamespace(enforce_eager=eager),
        additional_config={"moe_lora_ep_backend": "allgather"},
    )
    layer = SimpleNamespace(
        use_ep=True,
        fully_sharded=False,
        _w13_slices=2,
        hidden_size=4096,
        local_num_experts=32,
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=8), experts_per_token=6),
    )
    lora = SimpleNamespace(max_lora_rank=16, lora_dtype=torch.bfloat16)
    return config, layer, lora


@pytest.mark.parametrize(
    "method,tokens,runner,captures,eager,allowed",
    [
        ("dspark", 5, "0", (6, 12, 24, 48), False, True),
        ("dspark", 5, "0", (48,), False, True),
        ("dspark", 5, "0", (49,), False, False),
        ("dspark", 5, "0", (), False, False),
        ("dspark", 5, "0", (0, 48), False, False),
        ("dspark", 5, "0", (49,), True, False),
        ("dspark", 5, "1", (48,), False, False),
        ("dspark", 5, None, (48,), False, False),
        ("dspark", 7, "0", (48,), False, False),
        ("mtp", 5, "0", (48,), True, False),
        ("eagle3", 5, "0", (48,), True, False),
        (None, 0, "0", (1, 8), False, True),
        (None, 0, "0", (9,), False, False),
        (None, 0, "0", (48,), True, True),
    ],
)
def test_static_gate_only_adds_explicit_v1_dspark5(monkeypatch, method, tokens, runner, captures, eager, allowed):
    if runner is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner)
    assert static_config_supported(*configuration(method, tokens, captures, eager)) is allowed


@pytest.mark.parametrize("change", ["tp", "ep", "dp", "rank", "dtype", "fully_sharded", "backend", "overlap", "mode"])
def test_dspark_keeps_other_static_limits(monkeypatch, change):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    config, layer, lora = configuration()
    if change == "tp":
        config.parallel_config.tensor_parallel_size = 4
    elif change == "ep":
        layer.moe_config.moe_parallel_config.ep_size = 4
    elif change == "dp":
        config.parallel_config.data_parallel_size = 2
    elif change == "rank":
        lora.max_lora_rank = 32
    elif change == "dtype":
        lora.lora_dtype = torch.float16
    elif change == "fully_sharded":
        layer.fully_sharded = True
    elif change == "backend":
        config.additional_config["moe_lora_ep_backend"] = "alltoall"
    elif change == "overlap":
        config.additional_config["multistream_overlap_shared_expert"] = True
    else:
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    assert not static_config_supported(config, layer, lora)


def target_context(**changes):
    values = dict(
        cudagraph_runtime_mode=CUDAGraphMode.NONE, in_profile_run=False, attn_metadata={}, is_draft_model=False
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "changes,allowed",
    [
        ({}, True),
        ({"cudagraph_runtime_mode": CUDAGraphMode.FULL}, False),
        ({"cudagraph_runtime_mode": CUDAGraphMode.PIECEWISE}, False),
        ({"in_profile_run": True}, False),
        ({"in_profile_run": None}, False),
        ({"is_draft_model": True}, False),
        ({"is_draft_model": None}, False),
        ({"attn_metadata": None}, False),
        ({"cudagraph_runtime_mode": None}, False),
    ],
)
def test_target_runtime_gate(changes, allowed):
    assert supports_cube_target_prefill_context(target_context(**changes)) is allowed


@pytest.mark.parametrize("field", ["cudagraph_runtime_mode", "in_profile_run", "attn_metadata", "is_draft_model"])
def test_missing_runtime_field_falls_back(field):
    context = target_context()
    delattr(context, field)
    assert not supports_cube_target_prefill_context(context)
    assert not supports_cube_target_prefill_context(None)


@pytest.mark.parametrize(
    "mode,profile,draft,allowed",
    [
        (CUDAGraphMode.NONE, False, False, True),
        (CUDAGraphMode.FULL, False, False, False),
        (CUDAGraphMode.NONE, True, False, False),
        (CUDAGraphMode.NONE, False, True, False),
    ],
)
def test_actual_quant_mlp_caller_uses_target_runtime_gate(mode, profile, draft, allowed):
    state = Mock(prepare=Mock(return_value="cube-routing"))
    payload = _make_input(
        lora_context=SimpleNamespace(
            use_ep=True, cube_w13_state=state, punica_wrapper=SimpleNamespace(cube_single_slot=0)
        ),
        lora_routing=MoELoRARouting(torch.tensor([0, 1]), torch.tensor([0, 0])),
    )
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
        patch(
            f"{QUANT_MOE}.get_forward_context",
            return_value=target_context(
                cudagraph_runtime_mode=mode,
                in_profile_run=profile,
                is_draft_model=draft,
            ),
        ),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant", return_value=(torch.ones(2, 4), torch.ones(2))),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[torch.zeros(2, 6)]),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=torch.zeros(2, 3)),
        patch(f"{QUANT_MOE}.can_fuse_moe_lora_activation", return_value=False),
        patch(f"{QUANT_MOE}.DeviceOperator.npu_grouped_matmul_gmm2", return_value=torch.zeros(2, 4)),
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2"),
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=Mock()),
    ):
        quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
    assert state.prepare.call_count == int(allowed)
    assert ("cube_routing" in w13.call_args.kwargs) is allowed


@pytest.mark.parametrize("rows,allowed", [(240, False), (288, False), (6143, False), (6144, True)])
def test_prepare_rejects_decode_before_counts_readback_and_accepts_prefill(rows, allowed):
    # Minimal CPU shell exercises the production method, including its plan
    # and metadata copy, without allocating the large packed/scratch arenas.
    state = object.__new__(CubeW13State)
    state.loaded = {0}
    state.weights = (SimpleNamespace(shape=(1, 32, 16, 4096)),)
    state.packed = SimpleNamespace(device=torch.device("cpu"))
    state.generation = 3
    state.note = Mock()
    state.scratch = SimpleNamespace(same_stream=lambda: True, metadata=torch.empty(64, 3, dtype=torch.int64))
    x = SimpleNamespace(
        ndim=2, shape=(rows, 4096), dtype=torch.bfloat16, device=torch.device("cpu"), is_contiguous=lambda: True
    )
    counts = Mock(shape=(32,), dtype=torch.int64, device=torch.device("cpu"))
    counts.cpu.return_value.tolist.return_value = [24] * 32
    result = state.prepare(x, counts, 1, 0)
    assert (result is not None) is allowed
    assert counts.cpu.call_count == int(allowed)
    if allowed:
        assert result.total == 768 and result.tile_count == 32
        assert result.x is x and result.generation == 3


def test_native_opaque_moe_fake_does_not_enter_cube_python():
    # Test the registered upstream fake dispatch, not a stand-in operator.
    assert moe_runner.MoERunner._select_forward
    with (
        patch.object(CubeW13State, "prepare", side_effect=AssertionError("fake must not read counts")) as prepare,
        patch.object(moe_runner, "get_layer_from_name", side_effect=AssertionError("fake must not enter real MoE")),
        FakeTensorMode(),
    ):
        hidden = torch.empty(1024, 4096, dtype=torch.bfloat16)
        logits = torch.empty(1024, 256)
        output = torch.ops.vllm.moe_forward(hidden, logits, None, None, "target.layers.0", 0)
        assert output.shape == hidden.shape and output.dtype == hidden.dtype
    prepare.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_wrapper_creates_cube_state_only_with_existing_opt_in(monkeypatch, enabled):
    path = Path(__file__).parents[3] / "vllm_ascend/lora/fused_moe.py"
    source = ast.parse(path.read_text())
    wrapper = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AscendFusedMoEWithLoRA"
    )
    method = next(
        node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "create_lora_weights"
    )
    tree = ast.Module(
        body=[
            ast.ClassDef(
                name="Wrapper",
                bases=[ast.Name(id="Base", ctx=ast.Load())],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )

    class Base:
        def create_lora_weights(self, *args):
            pass

    namespace = {
        "Base": Base,
        "envs_ascend": SimpleNamespace(VLLM_ASCEND_MOE_LORA_W13_CUBE_PREFILL=enabled),
        "logger": Mock(),
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    owner = namespace["Wrapper"]()
    owner.base_layer = SimpleNamespace(quant_type=QuantType.W8A8, prefix="target.layers.0")
    owner.w13_lora_a_stacked = (object(), object())
    config, layer, lora = configuration()
    owner.__dict__.update(vars(layer))
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with (
        patch("vllm.config.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.lora.cube_moe.CubeW13State") as allocation,
    ):
        owner.create_lora_weights(3, lora)
    assert allocation.call_count == int(enabled)
    assert (owner._cube_w13_state is not None) is enabled
