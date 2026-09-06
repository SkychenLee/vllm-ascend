from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: F401

from vllm_ascend.lora.lora_ops import (
    moe_lora_prepare_allgather_bgmv_indices,
    moe_lora_prepare_sparse_group_list,
)


def test_prepare_allgather_bgmv_indices_allocates_fixed_output() -> None:
    expanded_row_idx = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    topk_ids = torch.tensor([[4, 5], [0, 1]], dtype=torch.int32)
    token_lora_indices = torch.tensor([0, 1], dtype=torch.int64)
    expert_map = torch.tensor([-1, -1, -1, -1, 0, 1], dtype=torch.int32)
    adapter_enabled = torch.tensor([1, 1], dtype=torch.int32)

    with patch.object(
        torch.ops._C_ascend,
        "moe_lora_prepare_allgather_bgmv_indices",
        create=True,
    ) as prepare:
        output = moe_lora_prepare_allgather_bgmv_indices(
            expanded_row_idx,
            topk_ids,
            token_lora_indices,
            expert_map,
            adapter_enabled,
            num_local_experts=2,
        )

    assert output.shape == expanded_row_idx.shape
    assert output.dtype == torch.int64
    assert output.device == expanded_row_idx.device
    prepare.assert_called_once_with(
        expanded_row_idx,
        topk_ids,
        token_lora_indices,
        expert_map,
        adapter_enabled,
        output,
        2,
    )


@pytest.mark.skipif(torch.npu.is_available() is not True, reason="requires an Ascend NPU")
def test_prepare_allgather_bgmv_indices_matches_recover_contract() -> None:
    expanded_row_idx = torch.tensor(
        [0, -1, -1, 2, -1, -1, 1, -1],
        dtype=torch.int32,
        device="npu",
    )
    topk_ids = torch.tensor(
        [[5, 1], [0, 7], [2, 1], [6, 3]],
        dtype=torch.int32,
        device="npu",
    )
    token_lora_indices = torch.tensor([0, 1, -1, 2], dtype=torch.int64, device="npu")
    expert_map = torch.tensor(
        [-1, -1, -1, -1, 0, 1, 2, 3],
        dtype=torch.int32,
        device="npu",
    )
    adapter_enabled = torch.tensor([1, 1, 0], dtype=torch.int32, device="npu")

    output = moe_lora_prepare_allgather_bgmv_indices(
        expanded_row_idx,
        topk_ids,
        token_lora_indices,
        expert_map,
        adapter_enabled,
        num_local_experts=4,
    )
    torch.npu.synchronize()

    expected = torch.tensor([1, -1, 7, -1, -1, -1, -1, -1], dtype=torch.int64)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_prepare_sparse_group_list_allocates_fixed_shape_output() -> None:
    group_list = torch.tensor([2, 0, 3, 0], dtype=torch.int64)

    with patch.object(
        torch.ops._C_ascend,
        "moe_lora_prepare_sparse_group_list",
        create=True,
    ) as prepare:
        output = moe_lora_prepare_sparse_group_list(group_list)

    assert output.shape == (4, 2)
    assert output.dtype == torch.int64
    assert output.device == group_list.device
    prepare.assert_called_once_with(group_list, output)


@pytest.mark.skipif(torch.npu.is_available() is not True, reason="requires an Ascend NPU")
def test_prepare_sparse_group_list_puts_nonempty_experts_first() -> None:
    group_list = torch.tensor([2, 0, 3, 0], dtype=torch.int64, device="npu")

    output = moe_lora_prepare_sparse_group_list(group_list)
    torch.npu.synchronize()

    expected = torch.tensor(
        [[0, 2], [2, 3], [1, 0], [3, 0]],
        dtype=torch.int64,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(torch.npu.is_available() is not True, reason="requires an Ascend NPU")
def test_sparse_group_list_matches_count_mode_w8a8_gmm() -> None:
    group_list = torch.tensor([2, 0, 3, 0], dtype=torch.int64, device="npu")
    sparse_group_list = moe_lora_prepare_sparse_group_list(group_list)
    inputs = (torch.arange(5 * 16, dtype=torch.int64) % 15 - 7).reshape(5, 16).to(torch.int8).npu()
    stored_weight = (torch.arange(4 * 16 * 16, dtype=torch.int64) % 15 - 7).reshape(4, 16, 16).to(torch.int8).npu()
    scale = torch.ones((4, 16), dtype=torch.bfloat16, device="npu")
    per_token_scale = torch.ones(5, dtype=torch.float32, device="npu")

    def grouped_matmul(groups: torch.Tensor, group_list_type: int) -> torch.Tensor:
        return torch_npu.npu_grouped_matmul(
            x=[inputs],
            weight=[stored_weight.transpose(-1, -2)],
            scale=[scale],
            per_token_scale=[per_token_scale],
            split_item=2,
            group_type=0,
            group_list=groups,
            group_list_type=group_list_type,
            output_dtype=torch.bfloat16,
        )[0]

    expected = grouped_matmul(group_list, 1)
    actual = grouped_matmul(sparse_group_list, 2)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
