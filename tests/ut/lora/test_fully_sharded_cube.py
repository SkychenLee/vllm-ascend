# SPDX-License-Identifier: Apache-2.0
"""Dispatch, lifetime and TP layout contracts for eager Cube projections."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from tests.ut.lora.test_cube_dspark import target_context
from tests.ut.lora.test_punica_npu import _make_wrapper
from tests.ut.lora.test_quant_moe import QUANT_MOE, _make_input
from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.lora import fully_sharded_cube as cube
from vllm_ascend.lora.cube_moe import single_adapter_slot
from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora


class Descriptor:
    def __init__(self, shape, dtype, device="npu:0"):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.contiguous = True
        self.cumsum = Mock(return_value="device-prefix")

    def is_contiguous(self):
        return self.contiguous

    def __getattr__(self, name):
        if name in ("cpu", "item", "tolist", "numpy", "data_ptr"):
            raise AssertionError(f"Cube dispatch must not read device data: {name}")
        raise AttributeError(name)


def descriptors(rows=24576):
    context = SimpleNamespace(
        fully_sharded=True,
        use_ep=False,
        tp_size=8,
        top_k=6,
        punica_wrapper=SimpleNamespace(cube_single_slot=1),
        w13_lora_a_packed=Descriptor((2, 3, 256, 2, 4096), torch.bfloat16),
        w2_lora_a_stacked=(Descriptor((3, 256, 16, 256), torch.bfloat16),),
    )
    return context, Descriptor((rows, 4096), torch.int8), Descriptor((256,), torch.int64)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_FULLY_SHARDED_CUBE", "1")
    with (
        patch.object(cube.torch_npu, "get_npu_format", return_value=2),
        patch.object(cube, "get_ascend_device_type", return_value=cube.AscendDeviceType.A3),
    ):
        yield


def test_flag_is_strict_and_off_by_default(monkeypatch):
    name = "VLLM_ASCEND_MOE_LORA_FULLY_SHARDED_CUBE"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    for value, expected in (("0", False), ("1", True)):
        monkeypatch.setenv(name, value)
        assert getattr(envs, name) is expected
    for value in ("", "2", "true", "-1"):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError):
            getattr(envs, name)


@pytest.mark.parametrize(
    "rows,allowed", [(48, False), (288, False), (16383, False), (16384, True), (98304, True), (98305, False)]
)
def test_row_gate_precedes_device_work(enabled, rows, allowed):
    context, x, counts = descriptors(rows)
    result = cube.prepare_cube_prefill(context, x, counts, 1, target_context())
    assert (result is not None) is allowed
    assert counts.cumsum.call_count == int(allowed)
    if allowed:
        assert result.ends == "device-prefix" and result.slot == 1 and result.rows == rows


@pytest.mark.parametrize("field,value", [("fully_sharded", False), ("use_ep", True), ("tp_size", 4), ("top_k", 8)])
def test_layout_guard(enabled, field, value):
    context, x, counts = descriptors()
    setattr(context, field, value)
    assert cube.prepare_cube_prefill(context, x, counts, 1, target_context()) is None
    counts.cumsum.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"in_profile_run": True},
        {"is_draft_model": True},
        {"attn_metadata": None},
        {"cudagraph_runtime_mode": CUDAGraphMode.FULL},
        {"cudagraph_runtime_mode": CUDAGraphMode.PIECEWISE},
    ],
)
def test_graph_dummy_and_draft_never_allocate(enabled, changes):
    context, x, counts = descriptors()
    assert cube.prepare_cube_prefill(context, x, counts, 1, target_context(**changes)) is None
    counts.cumsum.assert_not_called()


@pytest.mark.parametrize("slot", [None, -1, 3, True])
def test_single_adapter_slot_is_required(enabled, slot):
    context, x, counts = descriptors()
    context.punica_wrapper.cube_single_slot = slot
    assert cube.prepare_cube_prefill(context, x, counts, 1, target_context()) is None


@pytest.mark.parametrize(
    "mapping,expected", [([7, 7, 7], 1), ([7, 8], None), ([7, 0], None), ([0, 0], None), ([], None)]
)
def test_cpu_mapping_excludes_mixed_and_base(mapping, expected):
    assert single_adapter_slot(mapping, [8, 7, 9]) == expected


def test_storage_and_compilation_fallback(enabled):
    context, x, counts = descriptors()
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        assert cube.prepare_cube_prefill(context, x, counts, 1, target_context()) is None
    with patch.object(cube.torch_npu, "get_npu_format", return_value=29):
        assert cube.prepare_cube_prefill(context, x, counts, 1, target_context()) is None
    assert cube.prepare_cube_prefill(context, x, counts, 0, target_context()) is None
    with patch.object(cube, "get_ascend_device_type", return_value=cube.AscendDeviceType._310P):
        assert cube.prepare_cube_prefill(context, x, counts, 1, target_context()) is None
    counts.cumsum.assert_not_called()
    with patch.object(cube.torch_npu, "get_npu_format", side_effect=lambda t: 30 if t.ndim == 5 else 2):
        assert cube.prepare_cube_prefill(context, x, counts, 1, target_context()) is not None


def test_new_flag_updates_punica_slot_without_ep_cube_flag(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_FULLY_SHARDED_CUBE", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_W13_CUBE_PREFILL", "0")
    wrapper = _make_wrapper()
    with patch("vllm.lora.punica_wrapper.punica_base.PunicaWrapperBase.update_metadata"):
        for mapping, expected in [((7, 7), 1), ((7, 0), None), ((8, 8), 0), ((0, 0), None)]:
            wrapper.update_metadata(SimpleNamespace(index_mapping=mapping), [8, 7], 2, 128)
            assert wrapper.cube_single_slot == expected


@pytest.mark.parametrize("allowed", [False, True])
def test_quant_caller_shares_cube_routing_with_w13_and_w2(monkeypatch, allowed):
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_FULLY_SHARDED_CUBE", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_SINGLE_ADAPTER", "0")
    context = SimpleNamespace(fully_sharded=True, use_ep=False)
    payload = _make_input(
        hidden_states=torch.ones(2, 4, dtype=torch.int8),
        dynamic_scale=torch.ones(2),
        output_dtype=torch.bfloat16,
        lora_context=context,
    )
    metadata = object() if allowed else None
    forward = target_context()
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
        patch(f"{QUANT_MOE}.get_forward_context", return_value=forward),
        patch(f"{QUANT_MOE}.prepare_cube_prefill", return_value=metadata) as prepare,
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[torch.zeros(2, 6)]),
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather", return_value=(torch.zeros(2), torch.zeros(2))),
        patch(f"{QUANT_MOE}._can_fuse_int8_lora_swiglu", return_value=True),
        patch(f"{QUANT_MOE}.can_use_grouped_moe_lora", return_value=False),
        patch(
            f"{QUANT_MOE}.moe_lora_apply_w13_swiglu_quant",
            return_value=(torch.ones(2, 3, dtype=torch.int8), torch.ones(2), torch.zeros(2, dtype=torch.int64)),
        ) as w13,
        patch(f"{QUANT_MOE}.DeviceOperator.npu_grouped_matmul_gmm2", return_value=torch.zeros(2, 4)),
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as w2,
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=Mock()),
    ):
        quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
    prepare.assert_called_once_with(context, payload.hidden_states, payload.group_list, 1, forward)
    for projection in (w13, w2):
        assert ("sharded_cube_routing" in projection.call_args.kwargs) is allowed
        if allowed:
            assert projection.call_args.kwargs["sharded_cube_routing"] is metadata


def cpu_gmm(*, x, weight, group_list, **kwargs):
    result = torch.empty(x[0].shape[0], weight[0].shape[-1])
    start = 0
    for expert, end in enumerate(group_list.tolist()):
        result[start:end] = x[0][start:end] @ weight[0][expert]
        start = end
    assert start == x[0].shape[0]
    return [result]


@pytest.mark.parametrize("slices", [1, 2])
def test_chunk_boundaries_zero_groups_mask_and_live_weights(monkeypatch, slices):
    torch.manual_seed(37)
    rows, hidden, rank = 17, 32, 2
    counts = torch.tensor([0, 8, 0, 2, 7])
    ids = torch.repeat_interleave(torch.arange(5), counts) + 5
    ids[::3] = -1
    x = torch.randint(-127, 128, (rows, hidden), dtype=torch.int8)
    weights = tuple(torch.randn(3, 5, rank, hidden).bfloat16() for _ in range(slices))
    scales = torch.rand(rows)
    scales[::3] = float("nan")
    output = torch.empty(rows, slices * rank)
    routing = cube.CubePrefillRouting(counts.cumsum(0), 1, rows)
    monkeypatch.setattr(cube, "MAX_INPUT_BYTES", 4 * hidden * 6)
    with patch.object(cube.torch_npu, "npu_grouped_matmul", side_effect=cpu_gmm) as gmm:
        for generation in range(2):
            cube.shrink_cube_prefill(x, weights, ids, scales, output, routing)
            expected = torch.zeros_like(output)
            for row in range(rows):
                if ids[row] >= 0:
                    expert = ids[row] % 5
                    expected[row] = torch.cat([w[1, expert].float() @ x[row].float() for w in weights]) * scales[row]
            torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-5)
            assert torch.equal(output[::3], torch.zeros_like(output[::3]))
            if generation == 0:
                for weight in weights:
                    weight[1].add_(1)
        assert gmm.call_count == 6
        assert all(call.kwargs["x"][0].shape[0] <= 6 for call in gmm.call_args_list)


@pytest.mark.parametrize("paired", [False, True])
def test_punica_preserves_collective_and_expand_layout(paired):
    wrapper = _make_wrapper()
    rows, hidden, local_rank, full_rank = 3, 32, (2 if paired else 16), 16
    weights = torch.ones(2 if paired else 1, 2, 2, local_rank, hidden, dtype=torch.bfloat16)
    b = tuple(torch.ones(2, 2, 8, full_rank, dtype=torch.bfloat16) for _ in weights)
    routing = object()
    calls = []

    def cube_shrink(x, a, ids, scales, out, metadata):
        assert metadata is routing
        out.fill_(3)
        calls.append("cube")

    def gather(t, dim=0):
        assert t.shape == (rows, 4) and dim == 0
        calls.append("gather")
        return t.repeat(8, 1)

    def reduce(t):
        assert t.shape == (rows, 16)
        calls.append("reduce")
        return t * 8

    def fused_expand(y, a, *args):
        assert a.shape == (8, rows, 4) and torch.all(a == 3)
        calls.append("expand")
        return torch.zeros(rows, 8, dtype=torch.int8), torch.ones(rows)

    def expand(delta, *args, **kwargs):
        assert delta.shape == (rows, 16) and torch.all(delta == 24)
        calls.append("expand")

    with (
        patch("vllm_ascend.lora.punica_npu.shrink_cube_prefill", side_effect=cube_shrink),
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather", side_effect=gather),
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce", side_effect=reduce),
        patch("torch.ops._C_ascend.bgmv_shrink_int8", create=True) as vector,
        patch("torch.ops._C_ascend.bgmv_shrink_int8_pair", create=True) as vector_pair,
        patch("torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair", side_effect=fused_expand, create=True),
        patch.object(wrapper, "bgmv_expand_slice", side_effect=expand),
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(rows, 16 if paired else 8, dtype=torch.bfloat16),
            x=torch.ones(rows, hidden, dtype=torch.int8),
            lora_a_stacked=weights.unbind(0),
            lora_b_stacked=b,
            expert_ids=None,
            adapter_enabled=torch.ones(2, dtype=torch.int32),
            input_scale=torch.ones(rows),
            fully_sharded=True,
            combined_indices=torch.zeros(rows, dtype=torch.int64),
            paired_a_stacked=weights if paired else None,
            swiglu_quant_limit=10.0 if paired else None,
            sharded_cube_routing=routing,
        )
    vector.assert_not_called()
    vector_pair.assert_not_called()
    assert calls == ["cube", "gather" if paired else "reduce", "expand"]
