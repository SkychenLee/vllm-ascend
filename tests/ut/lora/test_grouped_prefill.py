# SPDX-License-Identifier: Apache-2.0
"""Static dispatch and independent adapter/expert routing contracts."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm_ascend.lora.grouped_prefill as grouped
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput, MoEQuantParams, MoEWeights
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize(
    "enabled,fully_sharded,use_ep,rows,hidden,width,local_rank,accepted",
    [
        (True, True, False, 49152, 4096, 256, 2, True),
        (True, True, False, 65536, 2048, 96, 2, True),
        (False, True, False, 49152, 4096, 256, 2, False),
        (True, True, False, 48, 4096, 256, 2, False),
        (True, False, False, 49152, 4096, 256, 2, False),
        (True, True, True, 49152, 4096, 256, 2, False),
        (True, True, False, 49152, 4096, 2048, 2, False),
        (True, True, False, 49152, 4096, 256, 8, False),
        (True, True, False, 49152, 4101, 256, 2, False),
    ],
)
def test_dispatch_keeps_disabled_decode_ep_and_unsupported_layout_on_legacy(
    enabled, fully_sharded, use_ep, rows, hidden, width, local_rank, accepted
):
    context = SimpleNamespace(
        fully_sharded=fully_sharded,
        use_ep=use_ep,
        w13_lora_a_packed=torch.empty(2, 3, 256, local_rank, hidden, device="meta"),
        w13_lora_b_stacked=[torch.empty(3, 256, width, 16, device="meta")],
    )
    inputs = SimpleNamespace(dtype=torch.int8, device=SimpleNamespace(type="npu"), shape=(rows, hidden))
    with (
        patch.object(grouped, "ENABLE_GROUPED_PREFILL", enabled),
        patch("torch.ops._C_ascend.bgmv_shrink_int8_pair_grouped", create=True),
        patch("torch.ops._C_ascend.moe_lora_expand_swiglu_quant_pair_grouped", create=True),
        patch("torch.ops._C_ascend.bgmv_expand_grouped", create=True),
    ):
        assert grouped.can_use_grouped_moe_lora(context, inputs, width) is accepted


@pytest.mark.parametrize("invalid", ["dtype", "dimensions", "groups_zero", "groups_precision"])
def test_grouping_rejects_invalid_metadata(invalid):
    indices, groups = torch.tensor([1, -1, 0], dtype=torch.int64), 2
    if invalid == "dtype":
        indices = indices.int()
    elif invalid == "dimensions":
        indices = indices[:, None]
    elif invalid == "groups_zero":
        groups = 0
    else:
        groups = 1 << 24
    with pytest.raises(ValueError):
        grouped.prepare_grouped_moe_lora_routing(indices, groups)


@pytest.mark.parametrize(
    "missing",
    ["bgmv_shrink_int8_pair_grouped", "moe_lora_expand_swiglu_quant_pair_grouped", "bgmv_expand_grouped"],
)
def test_dispatch_falls_back_when_any_grouped_native_schema_is_missing(missing):
    context = SimpleNamespace(
        fully_sharded=True,
        use_ep=False,
        w13_lora_a_packed=torch.empty(2, 3, 256, 2, 4096, device="meta"),
        w13_lora_b_stacked=[torch.empty(3, 256, 256, 16, device="meta")],
    )
    inputs = SimpleNamespace(dtype=torch.int8, device=SimpleNamespace(type="npu"), shape=(49152, 4096))
    names = ("bgmv_shrink_int8_pair_grouped", "moe_lora_expand_swiglu_quant_pair_grouped", "bgmv_expand_grouped")
    native = SimpleNamespace(**{name: Mock() for name in names if name != missing})
    with patch.object(grouped, "ENABLE_GROUPED_PREFILL", True), patch.object(torch.ops, "_C_ascend", native):
        assert not grouped.can_use_grouped_moe_lora(context, inputs, 256)


@pytest.mark.parametrize("single_adapter", [False, True])
def test_mlp_builds_own_combined_route_once_and_reuses_for_both_projections(single_adapter):
    rows, hidden, width, experts = 6, 64, 32, 2
    slots = torch.tensor([2, -1, 2, 2, 2, 2]) if single_adapter else torch.tensor([2, 1, 0, -1, 2, 0])
    expert_ids = torch.tensor([1, 0, 1, 1, 0, 0])
    context = SimpleNamespace(
        fully_sharded=True,
        use_ep=False,
        adapter_enabled=torch.tensor([1, 0, 1], dtype=torch.int32),
        w13_lora_a_stacked=[torch.empty(3, experts, 2, hidden) for _ in range(2)],
    )
    # Deliberately unrelated expert-only counts: the adapter helper has no
    # access to them, even in the single-adapter case.
    base_groups = torch.tensor([4, 2])
    payload = MoEMlpComputeInput(
        hidden_states=torch.ones(rows, hidden, dtype=torch.int8),
        dynamic_scale=torch.full((rows,), 0.01),
        output_dtype=torch.bfloat16,
        group_list=base_groups,
        group_list_type=1,
        topk_scales=None,
        weights=MoEWeights(
            w1=[torch.ones(experts, hidden, 2 * width, dtype=torch.int8)],
            w2=[torch.ones(experts, width, hidden, dtype=torch.int8)],
            w1_scale=[torch.ones(experts, 2 * width, dtype=torch.bfloat16)],
            w2_scale=[torch.ones(experts, hidden, dtype=torch.bfloat16)],
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
        fusion=True,
        swiglu_limit=10.0,
        expanded_row_idx=torch.arange(rows, dtype=torch.int32),
        topk_ids=torch.zeros(rows, 1, dtype=torch.int32),
        lora_context=context,
    )
    expected_ids = torch.tensor([5, -1, 5, 5, 4, 4]) if single_adapter else torch.tensor([5, -1, 1, -1, 4, 0])
    route = grouped.prepare_grouped_moe_lora_routing(expected_ids, 3 * experts)
    projected = torch.ones(rows, width, dtype=torch.int8)
    projected_scale = torch.full((rows,), 0.02)
    result = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    prefix = "vllm_ascend.lora.quant_moe"
    with (
        patch(f"{prefix}._EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
        patch(f"{prefix}._can_fuse_int8_lora_swiglu", return_value=True),
        patch(f"{prefix}.can_use_grouped_moe_lora", return_value=True),
        patch(f"{prefix}._recover_moe_lora_routing_allgather", return_value=(expert_ids, slots)),
        patch(f"{prefix}.prepare_grouped_moe_lora_routing", return_value=route) as prepare,
        patch(
            f"{prefix}.torch_npu.npu_grouped_matmul", return_value=[torch.zeros(rows, 2 * width, dtype=torch.bfloat16)]
        ),
        patch(
            f"{prefix}.moe_lora_apply_w13_swiglu_quant", return_value=(projected, projected_scale, expected_ids)
        ) as w13,
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=result) as gmm2,
        patch(f"{prefix}.moe_lora_apply_w2") as w2,
        patch(f"{prefix}.torch.npu.current_stream", return_value=Mock(record_event=Mock(return_value=object()))),
    ):
        actual, _ = quant_apply_mlp_with_moe_lora(mlp_compute_input=payload)
    assert actual is result
    prepare.assert_called_once()
    assert len(prepare.call_args.args) == 2 and prepare.call_args.kwargs == {}
    torch.testing.assert_close(prepare.call_args.args[0], expected_ids)
    assert prepare.call_args.args[1] == 3 * experts
    assert w13.call_args.kwargs["grouped_routing"] is route
    assert w2.call_args.kwargs["grouped_routing"] is route
    assert gmm2.call_args.kwargs["group_list"] is base_groups
    assert w2.call_args.kwargs["silu_out"] is projected
    assert w2.call_args.kwargs["input_scale"] is projected_scale
