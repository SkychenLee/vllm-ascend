# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("enabled_dtype", [torch.bool, torch.int32, torch.int64])
@torch.inference_mode()
def test_moe_lora_allgather_routing(index_dtype: torch.dtype, enabled_dtype: torch.dtype):
    expanded = torch.tensor([2, 0, 1, 3, 4, 5], dtype=index_dtype, device="npu")
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=index_dtype, device="npu")
    token_lora_indices = torch.tensor([0, -1, 1], dtype=torch.int64, device="npu")
    adapter_enabled = torch.tensor([1, 0], dtype=enabled_dtype, device="npu")
    combined_indices = torch.empty(6, dtype=torch.int64, device="npu")

    torch.ops._C_ascend.moe_lora_routing(
        expanded,
        topk_ids,
        token_lora_indices,
        adapter_enabled,
        combined_indices,
        2,
        2,
    )

    torch.testing.assert_close(combined_indices.cpu(), torch.tensor([0, -1, 1, -1, -1, -1]))


@torch.inference_mode()
def test_moe_lora_allgather_routing_aclgraph_capture_and_replay():
    expanded = torch.tensor([2, 0, 1, 3, 4, 5], dtype=torch.int32, device="npu")
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32, device="npu")
    token_lora_indices = torch.tensor([0, -1, 1], dtype=torch.int64, device="npu")
    adapter_enabled = torch.tensor([1, 0], dtype=torch.bool, device="npu")
    combined_indices = torch.empty(6, dtype=torch.int64, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        torch.ops._C_ascend.moe_lora_routing(
            expanded,
            topk_ids,
            token_lora_indices,
            adapter_enabled,
            combined_indices,
            2,
            2,
        )
    combined_indices.fill_(99)
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(combined_indices.cpu(), torch.tensor([0, -1, 1, -1, -1, -1]))


def _reference(
    x: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    indices: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = indices >= 0
    safe_indices = indices.clamp(min=0)
    workspace = []
    output = y.float().clone()
    slice_size = b0.shape[1]
    for slice_idx, (a, b) in enumerate(((a0, b0), (a1, b1))):
        shrink = torch.bmm(
            x.float().unsqueeze(1),
            a[safe_indices].float().transpose(1, 2),
        ).squeeze(1)
        shrink = torch.where(valid.view(-1, 1), shrink, torch.zeros_like(shrink))
        delta = torch.bmm(
            shrink.unsqueeze(1),
            b[safe_indices].float().transpose(1, 2),
        ).squeeze(1)
        delta = torch.where(valid.view(-1, 1), delta, torch.zeros_like(delta))
        start = slice_idx * slice_size
        output[:, start : start + slice_size] += delta
        workspace.append(shrink)
    return output.to(y.dtype), torch.stack(workspace)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_bgmv_moe_w13_rank16_decode(dtype: torch.dtype):
    batch_size, hidden_dim, rank, output_slice, num_loras = 6, 4096, 16, 2048, 4
    torch.manual_seed(7)
    x = torch.randn(batch_size, hidden_dim, dtype=dtype)
    a0 = torch.randn(num_loras, rank, hidden_dim, dtype=dtype) * 0.05
    a1 = torch.randn_like(a0) * 0.05
    b0 = torch.randn(num_loras, output_slice, rank, dtype=dtype) * 0.05
    b1 = torch.randn_like(b0) * 0.05
    indices = torch.tensor([0, 1, -1, 3, 0, 2], dtype=torch.int64)
    y = torch.randn(batch_size, output_slice * 2, dtype=dtype)
    expected_y, expected_workspace = _reference(x, a0, a1, b0, b1, indices, y)

    tensors = [value.npu() for value in (x, a0, a1, b0, b1, indices, y)]
    x_npu, a0_npu, a1_npu, b0_npu, b1_npu, indices_npu, y_npu = tensors
    workspace_npu = torch.empty((2, batch_size, rank), dtype=torch.float32, device="npu")
    torch.ops._C_ascend.bgmv_moe_w13(
        x_npu,
        a0_npu,
        a1_npu,
        b0_npu,
        b1_npu,
        indices_npu,
        workspace_npu,
        y_npu,
        0,
        1.0,
    )

    torch.testing.assert_close(y_npu.cpu(), expected_y, atol=0.08, rtol=0.02)
    torch.testing.assert_close(workspace_npu.cpu(), expected_workspace, atol=0.03, rtol=0.01)


@torch.inference_mode()
def test_bgmv_moe_w13_aclgraph_capture_and_replay():
    batch_size, hidden_dim, rank, output_slice, num_loras = 6, 4096, 16, 2048, 2
    dtype = torch.bfloat16
    x = torch.randn(batch_size, hidden_dim, dtype=dtype)
    a0 = torch.randn(num_loras, rank, hidden_dim, dtype=dtype) * 0.05
    a1 = torch.randn_like(a0) * 0.05
    b0 = torch.randn(num_loras, output_slice, rank, dtype=dtype) * 0.05
    b1 = torch.randn_like(b0) * 0.05
    indices = torch.arange(batch_size, dtype=torch.int64) % num_loras
    y = torch.zeros(batch_size, output_slice * 2, dtype=dtype)
    expected_y, _ = _reference(x, a0, a1, b0, b1, indices, y)

    x_npu, a0_npu, a1_npu, b0_npu, b1_npu, indices_npu, y_npu = [
        value.npu() for value in (x, a0, a1, b0, b1, indices, y)
    ]
    workspace_npu = torch.empty((2, batch_size, rank), dtype=torch.float32, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        torch.ops._C_ascend.bgmv_moe_w13(
            x_npu,
            a0_npu,
            a1_npu,
            b0_npu,
            b1_npu,
            indices_npu,
            workspace_npu,
            y_npu,
            0,
            1.0,
        )
    y_npu.zero_()
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(y_npu.cpu(), expected_y, atol=0.08, rtol=0.02)
