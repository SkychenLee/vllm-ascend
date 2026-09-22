# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.lora import lora_ops
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def _make_wrapper(*, is_prefill=False, no_lora=False) -> PunicaWrapperNPU:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.is_prefill = is_prefill
    wrapper.no_lora = no_lora
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand = Mock()
    wrapper.bgmv_expand_slice = Mock()
    wrapper.sgmv_shrink = Mock()
    wrapper.sgmv_expand = Mock()
    wrapper.sgmv_expand_slice = Mock()
    # PunicaWrapperBase exposes these as read-only properties.
    wrapper.batch_size = 1
    wrapper.max_length = 1
    wrapper.token_nums = 1
    wrapper._seq_start_locs = torch.tensor([0])
    wrapper._seq_lengths = torch.tensor([1])
    wrapper._lora_indices_per_batch = torch.tensor([0])
    wrapper._token_lora_indices = torch.tensor([0, 1, 2, 3])
    wrapper._sampler_indices = torch.tensor([1, 0, 1, 0])
    wrapper.indices_len = [2, 2, 2, 2]
    return wrapper


@pytest.mark.parametrize("local_rank,full_rank", [(2, 16), (16, 16)])
def test_int8_moe_shrink_communicates_before_fused_expand(local_rank, full_rank):
    wrapper = _make_wrapper()
    x = torch.ones(3, 32, dtype=torch.int8)
    scale = torch.ones(3)
    a = tuple(torch.ones(2, 2, local_rank, 32, dtype=torch.bfloat16) for _ in range(2))
    b = tuple(torch.ones(2, 2, 8, full_rank, dtype=torch.bfloat16) for _ in range(2))
    calls = []

    def shrink(x, w, ids, s, out):
        calls.append("shrink")
        assert x.dtype == torch.int8 and s is not None
        out.fill_(2)

    def communicate(t):
        calls.append("communicate")
        return torch.full((3, full_rank), 7.0)

    def expand(base, g, u, bg, bu, ids, topk, limit):
        calls.append("expand")
        assert torch.all(g == 7) and torch.all(u == 7)
        assert g.shape == (3, full_rank)
        torch.testing.assert_close(ids, torch.tensor([0, 3, -1]))
        return torch.zeros(3, 8, dtype=torch.int8), torch.ones(3)

    with (
        patch("torch.ops._C_ascend.bgmv_shrink_int8", side_effect=shrink, create=True),
        patch("torch.ops._C_ascend.moe_lora_expand_swiglu_quant", side_effect=expand, create=True),
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather", side_effect=communicate) as gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce", side_effect=communicate) as reduce,
    ):
        result = wrapper.add_lora_fused_moe(
            y=torch.zeros(3, 16, dtype=torch.bfloat16),
            x=x,
            lora_a_stacked=a,
            lora_b_stacked=b,
            expert_ids=torch.tensor([0, 1, 0]),
            adapter_enabled=torch.tensor([True, True]),
            token_lora_mapping=torch.tensor([0, 1, -1]),
            input_scale=scale,
            swiglu_quant_limit=10.0,
            fully_sharded=True,
        )
    assert calls == ["shrink", "communicate", "shrink", "communicate", "expand"]
    assert result[0].dtype == torch.int8
    torch.testing.assert_close(result[2], torch.tensor([0, 3, -1]))
    assert gather.call_count == (2 if local_rank != full_rank else 0)
    assert reduce.call_count == (2 if local_rank == full_rank else 0)
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


def test_int8_moe_reuses_w13_indices_for_w2():
    wrapper = _make_wrapper()
    combined = torch.tensor([0, 3, -1])
    with patch("torch.ops._C_ascend.bgmv_shrink_int8", create=True) as shrink:
        wrapper.add_lora_fused_moe(
            y=torch.zeros(3, 16, dtype=torch.bfloat16),
            x=torch.ones(3, 8, dtype=torch.int8),
            lora_a_stacked=[torch.ones(2, 2, 16, 8, dtype=torch.bfloat16)],
            lora_b_stacked=[torch.ones(2, 2, 16, 16, dtype=torch.bfloat16)],
            expert_ids=torch.tensor([0, 1, 0]),
            adapter_enabled=torch.ones(2, dtype=torch.bool),
            token_lora_mapping=torch.tensor([0, 1, -1]),
            input_scale=torch.ones(3),
            combined_indices=combined,
        )
    assert shrink.call_args.args[2] is combined
    assert wrapper.bgmv_expand_slice.call_args.args[3] is combined


@pytest.mark.parametrize(
    ("device_type", "max_lora_rank", "expect_torch_ops"),
    [
        (AscendDeviceType._310P, 8, True),
        (AscendDeviceType.A2, 128, True),
        (AscendDeviceType.A2, 16, False),
    ],
)
def test_punica_init_selects_kernel_backend(device_type, max_lora_rank, expect_torch_ops) -> None:
    with (
        patch(
            "vllm_ascend.lora.punica_npu.get_current_hardware_profile",
            return_value=get_hardware_profile(device_type),
        ),
        patch(
            "vllm_ascend.lora.punica_npu.get_ascend_device_type",
            return_value=device_type,
        ),
        patch("vllm_ascend.lora.punica_npu.refresh_all_lora_classes") as refresh,
    ):
        wrapper = PunicaWrapperNPU(
            8,
            2,
            torch.device("cpu"),
            lora_config=SimpleNamespace(
                max_lora_rank=max_lora_rank,
                max_loras=2,
                fully_sharded_loras=False,
            ),
        )
    refresh.assert_called_once()
    assert not hasattr(wrapper, "bgmv_shrink_pair")
    if expect_torch_ops:
        from vllm.lora.ops.torch_ops import bgmv_shrink

        assert wrapper.bgmv_shrink is bgmv_shrink
    else:
        assert wrapper.bgmv_shrink is lora_ops.bgmv_shrink


def test_prefill_calls_sgmv_when_lora_active() -> None:
    wrapper = _make_wrapper(is_prefill=True, no_lora=False)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    weights = torch.ones(2, 8, 4)
    wrapper._apply_shrink(y, x, weights, 1.0)
    wrapper._apply_expand(y, x, weights, 0, 8, True)
    wrapper.sgmv_shrink.assert_called_once()
    wrapper.sgmv_expand_slice.assert_called_once()


def test_prefill_shrink_and_expand_skip_when_no_lora() -> None:
    wrapper = _make_wrapper(is_prefill=True, no_lora=True)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    weights = torch.ones(2, 8, 4)
    wrapper._apply_shrink(y, x, weights, 1.0)
    wrapper._apply_expand(y, x, weights, 0, 8, True)
    wrapper.sgmv_shrink.assert_not_called()
    wrapper.sgmv_expand_slice.assert_not_called()


def test_decode_always_invokes_bgmv() -> None:
    wrapper = _make_wrapper(is_prefill=False, no_lora=True)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    weights = torch.ones(2, 8, 4)
    wrapper._apply_shrink(y, x, weights, 0.5)
    wrapper._apply_expand(y, x, weights, 2, 4, False)
    wrapper.bgmv_shrink.assert_called_once()
    wrapper.bgmv_expand_slice.assert_called_once()
    assert torch.equal(wrapper.bgmv_shrink.call_args.args[3], torch.tensor([0, 1]))


def test_add_shrink_and_expand_walk_slices_with_offsets() -> None:
    wrapper = _make_wrapper(is_prefill=False)
    wrapper._apply_shrink = Mock()
    wrapper._apply_expand = Mock()
    x = torch.ones(2, 8)
    y = torch.zeros(2, 12)
    a = (torch.ones(1, 4, 8), torch.ones(1, 4, 8))
    b = (torch.ones(1, 4, 4), torch.ones(1, 8, 4))
    wrapper.add_shrink((torch.zeros(2, 4), torch.zeros(2, 4)), x, a, 0.25)
    wrapper.add_expand(y, (torch.ones(2, 4), torch.ones(2, 4)), b, (4, 8), offset_start=0)
    assert wrapper._apply_shrink.call_count == 2
    assert wrapper._apply_expand.call_args_list[0].args[3] == 0
    assert wrapper._apply_expand.call_args_list[1].args[3] == 4
    assert wrapper._apply_expand.call_args_list[1].args[4] == 8


def test_add_lora_embedding_casts_input_to_fp32() -> None:
    wrapper = _make_wrapper(is_prefill=False)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 8, dtype=torch.float16)
    weights = torch.ones(2, 8, 8)
    wrapper.add_lora_embedding(y, x, weights)
    passed_x = wrapper.bgmv_expand.call_args.args[0]
    assert passed_x.dtype == torch.float32


def test_add_lora_embedding_prefill_uses_sgmv_expand() -> None:
    wrapper = _make_wrapper(is_prefill=True, no_lora=False)
    wrapper.add_lora_embedding(torch.zeros(2, 8), torch.ones(2, 8), torch.ones(2, 8, 8))
    wrapper.sgmv_expand.assert_called_once()
    wrapper.bgmv_expand.assert_not_called()


def test_lora_linear_kernel_allocates_fp32_buffer_when_missing() -> None:
    wrapper = _make_wrapper()
    wrapper._lora_shrink_buffers = {}
    wrapper._max_num_batched_tokens = 8
    wrapper.add_shrink = Mock()
    wrapper.add_expand = Mock()
    x = torch.ones(3, 8)
    y = torch.zeros(3, 16)
    a = (torch.ones(2, 4, 8),)
    b = (torch.ones(2, 16, 4),)
    wrapper._lora_linear_kernel(y, x, a, b, 1.0, (16,))
    buffer = wrapper.add_shrink.call_args.args[0]
    assert len(buffer) == 1
    assert buffer[0].shape == (3, 4)
    assert buffer[0].dtype == torch.float32
    wrapper.add_expand.assert_called_once()


def test_add_lora_fused_moe_builds_graph_safe_combined_index() -> None:
    wrapper = _make_wrapper()
    max_loras, num_experts, rank, in_f, out_f = 2, 4, 8, 16, 32
    a = torch.ones(max_loras, num_experts, rank, in_f)
    b = torch.ones(max_loras, num_experts, out_f, rank)
    x = torch.ones(3, in_f)
    y = torch.zeros(3, out_f)
    mapping = torch.tensor([0, -1, 1])
    expert_ids = torch.tensor([1, 2, 3])
    adapter_enabled = torch.tensor([1, 1])
    wrapper.add_lora_fused_moe(
        y,
        x,
        (a,),
        (b,),
        expert_ids=expert_ids,
        adapter_enabled=adapter_enabled,
        token_lora_mapping=mapping,
        offset=5,
    )
    combined = wrapper.bgmv_shrink.call_args.args[3]
    assert torch.equal(combined, torch.tensor([1, -1, 7]))
    assert wrapper.bgmv_expand_slice.call_args.args[4:6] == (5, 32)
    assert wrapper.bgmv_expand_slice.call_args.kwargs["add_inputs"] is True


def test_add_lora_fused_moe_masks_disabled_adapter_rows() -> None:
    wrapper = _make_wrapper()
    a = torch.ones(2, 4, 2, 8)
    b = torch.ones(2, 4, 8, 2)
    wrapper.add_lora_fused_moe(
        torch.zeros(2, 8),
        torch.ones(2, 8),
        (a,),
        (b,),
        expert_ids=torch.tensor([1, 2]),
        adapter_enabled=torch.tensor([0, 1]),
        token_lora_mapping=torch.tensor([0, 1]),
    )
    combined = wrapper.bgmv_shrink.call_args.args[3]
    assert torch.equal(combined, torch.tensor([-1, 6]))


def test_add_lora_fused_moe_rejects_unexpanded_rows() -> None:
    wrapper = _make_wrapper()
    with pytest.raises(AssertionError, match="top_k_num=1"):
        wrapper.add_lora_fused_moe(
            torch.zeros(1, 4),
            torch.ones(1, 4),
            (torch.ones(1, 1, 2, 4),),
            (torch.ones(1, 1, 4, 2),),
            expert_ids=torch.tensor([0]),
            adapter_enabled=torch.tensor([1]),
            top_k_num=2,
        )


def test_add_lora_fused_moe_scales_shrink_buffer_by_routed_weight() -> None:
    wrapper = _make_wrapper()

    def _shrink(x, a, shrink_out, idx, scale):
        shrink_out.fill_(2.0)

    wrapper.bgmv_shrink.side_effect = _shrink
    a = torch.ones(1, 1, 2, 4)
    b = torch.ones(1, 1, 4, 2)
    wrapper.add_lora_fused_moe(
        torch.zeros(2, 4),
        torch.ones(2, 4),
        (a,),
        (b,),
        expert_ids=torch.tensor([0, 0]),
        adapter_enabled=torch.tensor([1]),
        token_lora_mapping=torch.tensor([0, 0]),
        mul_routed_weight=True,
        topk_weights=torch.tensor([0.5, 1.5]),
    )
    delta = wrapper.bgmv_expand_slice.call_args.args[0]
    torch.testing.assert_close(delta, torch.tensor([[1.0, 1.0], [3.0, 3.0]]))


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("projection", ["gate_up", "down"])
@pytest.mark.parametrize("fully_sharded", [False, True])
def test_moe_lora_tp_matches_unsharded_projection(tp_size, projection, fully_sharded) -> None:
    """Check rank gathering, output offsets and expert/adapter isolation numerically."""
    generator = torch.Generator().manual_seed(1024)
    rows, in_size, out_size, rank = 4, 32, 32, 16
    slices = 2 if projection == "gate_up" else 1
    x = torch.randn(rows, in_size, generator=generator)
    a = [torch.randn(3, 2, rank, in_size, generator=generator) for _ in range(slices)]
    b = [torch.randn(3, 2, out_size, rank, generator=generator) for _ in range(slices)]
    mapping = torch.tensor([0, 1, -1, 2])
    experts = torch.tensor([1, 0, 1, 0])
    enabled = torch.tensor([1, 1, 0])
    active = (mapping >= 0) & enabled[mapping.clamp(min=0)].bool()
    indices = mapping.clamp(min=0) * 2 + experts

    def project(value, weights):
        result = torch.bmm(weights.flatten(0, 1)[indices], value.unsqueeze(-1)).squeeze(-1)
        return result * active[:, None]

    reference = torch.cat([project(project(x, wa), wb) for wa, wb in zip(a, b)], dim=-1)
    local_inputs, local_as, local_bs = [], [], []
    for tp_rank in range(tp_size):
        if projection == "gate_up":
            local_inputs.append(x)
            local_as.append([wa.chunk(tp_size, dim=2)[tp_rank].contiguous() if fully_sharded else wa for wa in a])
            local_bs.append([wb.chunk(tp_size, dim=2)[tp_rank].contiguous() for wb in b])
        else:
            local_inputs.append(x.chunk(tp_size, dim=1)[tp_rank].contiguous())
            local_as.append([wa.chunk(tp_size, dim=3)[tp_rank].contiguous() for wa in a])
            local_bs.append([wb.chunk(tp_size, dim=2)[tp_rank].contiguous() if fully_sharded else wb for wb in b])

    projections = [[project(local_inputs[r], wa) for wa in local_as[r]] for r in range(tp_size)]

    def shrink(value, weights, output, ids, scale):
        output.copy_(torch.bmm(weights[ids.clamp(min=0)], value.unsqueeze(-1)).squeeze(-1) * scale)
        output[ids < 0] = 0

    def expand(value, weights, output, ids, offset, size, add_inputs=True):
        delta = torch.bmm(weights[ids.clamp(min=0)], value.unsqueeze(-1)).squeeze(-1)
        delta[ids < 0] = 0
        output[:, offset : offset + size].add_(delta)

    parts = []
    for tp_rank in range(tp_size):
        wrapper = _make_wrapper()
        wrapper.bgmv_shrink = shrink
        wrapper.bgmv_expand_slice = expand
        slice_index = 0

        def collective(value, tp_rank=tp_rank):
            nonlocal slice_index
            if value.ndim == 3:
                assert value.is_contiguous()
                assert all(part.is_contiguous() for part in value)
                torch.testing.assert_close(value, torch.stack(projections[tp_rank]))
                slice_index += slices
                return torch.cat([torch.stack(projections[r]) for r in range(tp_size)], dim=-1)
            torch.testing.assert_close(value, projections[tp_rank][slice_index])
            values = [projections[r][slice_index] for r in range(tp_size)]
            slice_index += 1
            return torch.cat(values, dim=-1) if projection == "gate_up" else torch.stack(values).sum(0)

        local_out_size = out_size // tp_size if projection == "gate_up" else out_size
        original = torch.randn(rows, local_out_size * slices, generator=generator)
        output = original.clone()
        with (
            patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather", side_effect=collective) as gather,
            patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce", side_effect=collective) as reduce,
        ):
            wrapper.add_lora_fused_moe(
                output,
                local_inputs[tp_rank],
                tuple(local_as[tp_rank]),
                tuple(local_bs[tp_rank]),
                expert_ids=experts,
                adapter_enabled=enabled,
                fully_sharded=fully_sharded,
                offset=tp_rank * (out_size // tp_size) if projection == "down" and fully_sharded else 0,
                token_lora_mapping=mapping,
            )
        expected_collectives = slices if fully_sharded else 0
        assert gather.call_count + reduce.call_count == expected_collectives
        torch.testing.assert_close(output[~active], original[~active])
        parts.append(output - original)

    if projection == "down":
        actual = torch.stack(parts).sum(0)
    else:
        actual = torch.cat(
            [torch.cat([part.chunk(slices, dim=-1)[s] for part in parts], dim=-1) for s in range(slices)], dim=-1
        )
    torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("local_ranks,full_ranks", [((2, 4), (16, 16)), ((2, 2), (16, 8))])
def test_moe_lora_keeps_separate_gathers_for_incompatible_slices(local_ranks, full_ranks) -> None:
    wrapper = _make_wrapper()
    a = tuple(torch.ones(1, 1, rank, 32) for rank in local_ranks)
    b = tuple(torch.ones(1, 1, 32, rank) for rank in full_ranks)
    with patch(
        "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather",
        side_effect=[torch.zeros(3, rank) for rank in full_ranks],
    ) as gather:
        wrapper.add_lora_fused_moe(
            torch.zeros(3, 64),
            torch.ones(3, 32),
            a,
            b,
            expert_ids=torch.zeros(3, dtype=torch.long),
            adapter_enabled=torch.ones(1),
            token_lora_mapping=torch.zeros(3, dtype=torch.long),
            fully_sharded=True,
        )
    assert gather.call_count == 2
    assert all(call.args[0].ndim == 2 for call in gather.call_args_list)
    assert wrapper.bgmv_shrink.call_count == wrapper.bgmv_expand_slice.call_count == 2


def test_moe_lora_separate_gathers_preserve_slice_offsets_and_routed_weights() -> None:
    wrapper = _make_wrapper()
    a = (torch.ones(1, 1, 2, 32), torch.full((1, 1, 2, 32), 2.0))
    b = (torch.ones(1, 1, 8, 16), torch.ones(1, 1, 12, 16))
    gathered = (torch.ones(3, 16), torch.full((3, 16), 2.0))
    routed_weights = torch.tensor([0.5, 0.0, 1.5])
    with patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather", side_effect=gathered) as gather:
        wrapper.add_lora_fused_moe(
            torch.zeros(3, 32),
            torch.ones(3, 32),
            a,
            b,
            expert_ids=torch.zeros(3, dtype=torch.long),
            adapter_enabled=torch.ones(1),
            token_lora_mapping=torch.tensor([0, -1, 0]),
            fully_sharded=True,
            offset=5,
            mul_routed_weight=True,
            topk_weights=routed_weights,
        )
    assert gather.call_count == 2
    assert all(call.args[0].shape == (3, 2) for call in gather.call_args_list)
    assert all(call.args[0].dtype == torch.float32 for call in gather.call_args_list)
    for slice_idx, (offset, size) in enumerate(((5, 8), (13, 12))):
        call = wrapper.bgmv_expand_slice.call_args_list[slice_idx]
        torch.testing.assert_close(call.args[0], gathered[slice_idx] * routed_weights[:, None])
        assert call.args[4:6] == (offset, size)
        assert torch.equal(call.args[3], torch.tensor([0, -1, 0]))


@pytest.mark.parametrize("paired", [False, True])
def test_moe_lora_rejects_rank_mismatch_before_expand(paired) -> None:
    wrapper = _make_wrapper()
    slices = 2 if paired else 1
    with (
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather", return_value=torch.zeros(2, 1, 8)),
        pytest.raises(ValueError, match="MoE LoRA rank mismatch"),
    ):
        wrapper.add_lora_fused_moe(
            torch.zeros(1, 4),
            torch.ones(1, 4),
            tuple(torch.ones(1, 1, 2, 4) for _ in range(slices)),
            tuple(torch.ones(1, 1, 4, 16) for _ in range(slices)),
            expert_ids=torch.tensor([0]),
            adapter_enabled=torch.tensor([1]),
            token_lora_mapping=torch.tensor([0]),
            fully_sharded=paired,
        )
    wrapper.bgmv_expand_slice.assert_not_called()


def test_add_lora_logits_uses_sampler_indices() -> None:
    wrapper = _make_wrapper()
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    a = torch.ones(3, 4, 4)
    b = torch.ones(3, 8, 4)
    wrapper.add_lora_logits(y, x, a, b, 0.5)
    assert torch.equal(wrapper.bgmv_shrink.call_args.args[3], torch.tensor([1, 0]))
    wrapper.bgmv_expand.assert_called_once()


def test_bgmv_shrink_pair_wrapper_forwards_output_mutation_contract() -> None:
    x = torch.ones(4, 11, dtype=torch.bfloat16)
    weights = [torch.ones(6, 2, 11, dtype=torch.bfloat16) for _ in range(2)]
    indices = torch.tensor([0, -1, 4, 5])
    output = torch.zeros(2, 4, 2, dtype=torch.float32)
    operation = Mock(return_value=None)
    with patch.object(torch.ops, "_C_ascend", SimpleNamespace(bgmv_shrink_pair=operation)):
        assert lora_ops.bgmv_shrink_pair(x, weights[0], weights[1], output, indices, 0.5) is None
    operation.assert_called_once_with(x, weights[0], weights[1], indices, output, 0.5)


def test_moe_lora_uses_independent_shrink_and_gather_without_packing_weights() -> None:
    wrapper = _make_wrapper()
    x = torch.ones(4, 11, dtype=torch.bfloat16)
    a = (torch.ones(2, 3, 2, 11, dtype=torch.bfloat16), torch.full((2, 3, 2, 11), 3, dtype=torch.bfloat16))
    b = (torch.ones(2, 3, 5, 8, dtype=torch.bfloat16), torch.ones(2, 3, 7, 8, dtype=torch.bfloat16))
    indices = torch.tensor([1, -1, 4, 5])
    routed = torch.tensor([0.5, 1.0, 0.0, 2.0])
    gathered_inputs = []

    def shrink(value, weight, output, actual_indices, scale):
        projection = len(gathered_inputs)
        assert value.data_ptr() == x.data_ptr()
        assert weight.data_ptr() == a[projection].data_ptr()
        assert weight.shape == (6, 2, 11)
        assert output.shape == (4, 2) and output.dtype == torch.float32
        assert scale == 1.0
        torch.testing.assert_close(actual_indices, indices)
        output.fill_(1.0 if projection == 0 else 3.0)
        output[actual_indices < 0] = 0

    def gather(value):
        gathered_inputs.append(value.clone())
        return value.repeat(1, 4)

    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    with patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather", side_effect=gather) as collective:
        wrapper.add_lora_fused_moe(
            torch.zeros(4, 20),
            x,
            a,
            b,
            expert_ids=torch.tensor([1, 2, 1, 2]),
            adapter_enabled=torch.ones(2),
            token_lora_mapping=torch.tensor([0, -1, 1, 1]),
            fully_sharded=True,
            offset=3,
            mul_routed_weight=True,
            topk_weights=routed,
        )
    assert wrapper.bgmv_shrink.call_count == collective.call_count == 2
    assert wrapper.bgmv_expand_slice.call_count == 2
    for projection, (offset, size, fill) in enumerate(((3, 5, 1.0), (8, 7, 3.0))):
        expected = torch.full((4, 2), fill)
        expected[1] = 0
        torch.testing.assert_close(gathered_inputs[projection], expected)
        call = wrapper.bgmv_expand_slice.call_args_list[projection]
        torch.testing.assert_close(call.args[0], expected.repeat(1, 4) * routed[:, None])
        torch.testing.assert_close(call.args[3], indices)
        assert call.args[4:6] == (offset, size)


def test_moe_lora_reuses_cube_routing_for_two_projections() -> None:
    wrapper = _make_wrapper()
    wrapper.bgmv_shrink = lora_ops.bgmv_shrink
    wrapper.bgmv_expand_slice = lora_ops.bgmv_expand_slice
    rows, hidden, rank, width, groups = 2049, 80, 8, 64, 2
    x = torch.ones(rows, hidden, dtype=torch.bfloat16)
    y = torch.zeros(rows, 2 * width, dtype=torch.bfloat16)
    a = tuple(torch.ones(1, groups, rank, hidden, dtype=torch.bfloat16) for _ in range(2))
    b = tuple(torch.ones(1, groups, width, rank, dtype=torch.bfloat16) for _ in range(2))
    routing = SimpleNamespace(groups=groups)
    with (
        patch("vllm_ascend.lora.punica_npu.can_use_cube_bgmv", return_value=True),
        patch("vllm_ascend.lora.punica_npu.prepare_cube_bgmv_routing", return_value=routing) as prepare,
        patch("vllm_ascend.lora.punica_npu.cube_bgmv_shrink") as shrink,
        patch("vllm_ascend.lora.punica_npu.cube_bgmv_expand") as expand,
    ):
        wrapper.add_lora_fused_moe(
            y,
            x,
            a,
            b,
            expert_ids=torch.arange(rows) % groups,
            adapter_enabled=torch.ones(1),
            token_lora_mapping=torch.zeros(rows, dtype=torch.long),
        )
    prepare.assert_called_once()
    assert shrink.call_count == expand.call_count == 2
    assert all(call.kwargs["routing"] is routing for call in shrink.call_args_list + expand.call_args_list)
    assert [call.args[4:6] for call in expand.call_args_list] == [(0, width), (width, width)]


@pytest.mark.parametrize("local_rank,full_rank,rows", [(2, 16, 3), (16, 16, 3), (2, 16, 0)])
def test_paired_moe_single_collective_and_rank_major_layout(local_rank, full_rank, rows):
    wrapper = _make_wrapper()
    packed = torch.ones(2, 2, 2, local_rank, 32, dtype=torch.bfloat16)
    b = tuple(torch.ones(2, 2, 8, full_rank, dtype=torch.bfloat16) for _ in range(2))
    calls = []
    shards = full_rank // local_rank

    def shrink(x, w, ids, scale, out):
        calls.append("shrink")
        out.fill_(2)

    def gather(t, dim):
        calls.append("gather")
        assert dim == 0
        return torch.cat([t + i for i in range(shards)], dim=0)

    def reduce(t):
        calls.append("reduce")
        return t + 3

    def expand(base, a, bg, bu, ids, topk, limit):
        calls.append("expand")
        assert a.shape == (shards, rows, 2 * local_rank)
        if rows:
            assert torch.all(a[0] == (2 if shards > 1 else 5))
            if shards > 1:
                assert torch.all(a[-1] == 2 + shards - 1)
        return torch.zeros(rows, 8, dtype=torch.int8), torch.ones(rows)

    with (
        patch("torch.ops._C_ascend.bgmv_shrink_int8_pair", side_effect=shrink, create=True),
        patch("torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair", side_effect=expand, create=True),
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather", side_effect=gather),
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce", side_effect=reduce),
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(rows, 16, dtype=torch.bfloat16),
            x=torch.ones(rows, 32, dtype=torch.int8),
            lora_a_stacked=packed.unbind(0),
            lora_b_stacked=b,
            paired_a_stacked=packed,
            expert_ids=torch.zeros(rows, dtype=torch.int64),
            adapter_enabled=torch.ones(2, dtype=torch.bool),
            token_lora_mapping=torch.zeros(rows, dtype=torch.int64),
            input_scale=torch.ones(rows),
            swiglu_quant_limit=10.0,
            fully_sharded=True,
        )
    expected = ["shrink"]
    if rows:
        expected.append("gather" if shards > 1 else "reduce")
    assert calls == expected + ["expand"]
