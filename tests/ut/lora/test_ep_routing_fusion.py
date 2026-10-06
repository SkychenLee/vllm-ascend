# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.lora import routing


class _NpuMetadata(torch.Tensor):
    """Exercise host-only dispatch guards without allocating on an NPU."""

    @property
    def device(self):
        return torch.device("npu:0")


def _metadata(tokens=8, experts=256):
    return {
        "expanded_row_idx": torch.arange(tokens * 6, dtype=torch.int32).as_subclass(_NpuMetadata),
        "topk_ids": torch.zeros(tokens, 6, dtype=torch.int32).as_subclass(_NpuMetadata),
        "prepared_lora_indices": torch.zeros(tokens, dtype=torch.int64).as_subclass(_NpuMetadata),
        "expert_map": torch.zeros(experts, dtype=torch.int32).as_subclass(_NpuMetadata),
        "num_dispatched_rows": tokens * 6,
    }


@pytest.mark.parametrize("tokens", [1, 8, 64])
def test_decode_metadata_uses_fixed_shape_native_operator(tokens):
    metadata = _metadata(tokens)
    expected = (torch.arange(tokens * 6, dtype=torch.int64), torch.ones(tokens * 6, dtype=torch.int64))
    with (
        patch.object(routing.envs, "VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", True),
        patch.object(torch.ops._C_ascend, "moe_lora_ep_routing", Mock(return_value=expected), create=True) as native,
        patch.object(routing.torch_npu, "get_npu_format", return_value=2),
    ):
        actual = routing.recover_allgather_ep_lora_routing(**metadata)
    assert actual.expert_ids is expected[0]
    assert actual.adapter_slots is expected[1]
    native.assert_called_once_with(
        metadata["expanded_row_idx"],
        metadata["topk_ids"],
        metadata["prepared_lora_indices"],
        metadata["expert_map"],
        metadata["num_dispatched_rows"],
    )


@pytest.mark.parametrize("reason", ["disabled", "prefill", "large_map", "strided", "float_map", "cpu"])
def test_unsupported_metadata_preserves_original_fallback(reason):
    metadata = _metadata(tokens=65 if reason == "prefill" else 8, experts=513 if reason == "large_map" else 256)
    if reason == "strided":
        metadata["expanded_row_idx"] = torch.arange(96, dtype=torch.int32)[::2].as_subclass(_NpuMetadata)
    if reason == "float_map":
        metadata["expert_map"] = metadata["expert_map"].float()
    if reason == "cpu":
        metadata = {k: v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v for k, v in metadata.items()}
    with (
        patch.object(routing.envs, "VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", reason != "disabled"),
        patch.object(torch.ops._C_ascend, "moe_lora_ep_routing", Mock(), create=True) as native,
        patch.object(torch, "argsort", side_effect=RuntimeError("original fallback reached")),
        pytest.raises(RuntimeError, match="original fallback reached"),
    ):
        routing.recover_allgather_ep_lora_routing(**metadata)
    native.assert_not_called()


@pytest.mark.parametrize("name", ["expanded_row_idx", "topk_ids", "prepared_lora_indices", "expert_map"])
@pytest.mark.parametrize("format_id", [3, 4, 29])
def test_logically_contiguous_blocked_metadata_preserves_fallback(name, format_id):
    metadata = _metadata()
    target = metadata[name]
    assert target.is_contiguous()
    with (
        patch.object(routing.envs, "VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", True),
        patch.object(torch.ops._C_ascend, "moe_lora_ep_routing", Mock(), create=True) as native,
        patch.object(routing.torch_npu, "get_npu_format", side_effect=lambda t: format_id if t is target else 2),
        patch.object(torch, "argsort", side_effect=RuntimeError("original fallback reached")),
        pytest.raises(RuntimeError, match="original fallback reached"),
    ):
        routing.recover_allgather_ep_lora_routing(**metadata)
    native.assert_not_called()


def test_compile_dispatch_leaves_storage_validation_to_native_runtime():
    metadata = _metadata()
    expected = (torch.zeros(48, dtype=torch.int64), torch.ones(48, dtype=torch.int64))
    with (
        patch.object(routing.envs, "VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", True),
        patch.object(torch.ops._C_ascend, "moe_lora_ep_routing", Mock(return_value=expected), create=True) as native,
        patch.object(torch.compiler, "is_compiling", return_value=True),
        patch.object(routing.torch_npu, "get_npu_format", side_effect=RuntimeError("FakeTensor has no descriptor")),
    ):
        result = routing.recover_allgather_ep_lora_routing(**metadata)
    assert result.expert_ids is expected[0]
    native.assert_called_once()


def test_negative_sentinels_and_internal_holes_retain_original_semantics():
    # Unique but malformed expanded destinations have an internal gap at 1.
    # The original sorted ordinal comparison invalidates every later row.
    expanded = torch.tensor([0, 2, 3, -1, -(1 << 31), 100], dtype=torch.int32)
    with patch.object(routing.envs, "VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", True):
        result = routing.recover_allgather_ep_lora_routing(
            expanded_row_idx=expanded,
            topk_ids=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
            prepared_lora_indices=torch.tensor([(1 << 53) + 101, -1], dtype=torch.int64),
            expert_map=torch.tensor([3, -1], dtype=torch.int32),
            num_dispatched_rows=6,
        )
    torch.testing.assert_close(result.expert_ids, torch.tensor([3, -1, -1, -1, -1, -1]))
    torch.testing.assert_close(result.adapter_slots, torch.tensor([(1 << 53) + 101, -1, -1, -1, -1, -1]))
