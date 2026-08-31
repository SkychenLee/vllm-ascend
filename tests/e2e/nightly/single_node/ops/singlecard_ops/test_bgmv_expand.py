import gc

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

BF16_ATOL = 2e-2
BF16_RTOL = 2e-2
FP16_ATOL = 1e-3
FP16_RTOL = 1e-3


def bgmv_expand_cpu_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    indices: torch.Tensor,
    y: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool,
) -> torch.Tensor:
    output_slice = y[:, slice_offset : slice_offset + slice_size]
    if not add_inputs:
        output_slice.zero_()
    active = indices >= 0
    W = w[indices[active], :, :].transpose(-1, -2).to(torch.float32)
    z = torch.bmm(x[active].unsqueeze(1).to(torch.float32), W).squeeze(1)
    if add_inputs:
        output_slice[active] += z
    else:
        output_slice[active] = z.to(output_slice.dtype)
    return y


@torch.inference_mode()
@pytest.mark.parametrize(
    ("dtype", "slice_size"),
    [
        (torch.float16, 128),
        (torch.bfloat16, 2048),
        (torch.bfloat16, 4096),
    ],
)
@pytest.mark.parametrize("add_inputs", [True, False])
def test_bgmv_expand(dtype: torch.dtype, slice_size: int, add_inputs: bool):
    B = 4
    slice_offset = 128
    x = torch.randn([B, 16], dtype=torch.float)
    w = torch.randn([64, slice_size, 16], dtype=dtype)
    indices = torch.tensor([0, -1, 1, -1], dtype=torch.int64)
    y = torch.randn([B, slice_size * 3], dtype=dtype)

    x_npu = x.npu()
    w_npu = w.npu()
    indices_npu = indices.npu()
    y_npu = y.clone().npu()

    y_out = bgmv_expand_cpu_impl(x, w, indices, y, slice_offset, slice_size, add_inputs)
    if add_inputs:
        # Exercise the schema default to preserve existing direct callers.
        y_out_npu = torch.ops._C_ascend.bgmv_expand(
            x_npu,
            w_npu,
            indices_npu,
            y_npu,
            slice_offset,
            slice_size,
        )
    else:
        y_out_npu = torch.ops._C_ascend.bgmv_expand(
            x_npu,
            w_npu,
            indices_npu,
            y_npu,
            slice_offset,
            slice_size,
            False,
        )

    # Compare the results.
    atol = FP16_ATOL if dtype == torch.float16 else BF16_ATOL
    rtol = FP16_RTOL if dtype == torch.float16 else BF16_RTOL
    torch.testing.assert_close(y_out_npu.cpu(), y_out.cpu(), atol=atol, rtol=rtol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_bgmv_expand_overwrite_aclgraph_clears_stale_rows():
    batch_size = 4
    slice_offset = 128
    slice_size = 4096
    output_size = slice_offset + slice_size + 128
    x = torch.randn([batch_size, 16], dtype=torch.float32).npu()
    w = torch.randn([2, slice_size, 16], dtype=torch.bfloat16).npu()
    indices = torch.zeros([batch_size], dtype=torch.int64).npu()
    y = torch.empty([batch_size, output_size], dtype=torch.bfloat16, device="npu")

    def overwrite():
        return torch.ops._C_ascend.bgmv_expand(
            x,
            w,
            indices,
            y,
            slice_offset,
            slice_size,
            False,
        )

    overwrite()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = overwrite()

    y.fill_(7)
    indices.fill_(-1)
    graph.replay()
    torch.npu.synchronize()

    output_cpu = output.cpu()
    assert torch.count_nonzero(output_cpu[:, slice_offset : slice_offset + slice_size]) == 0
    torch.testing.assert_close(output_cpu[:, :slice_offset], torch.full_like(output_cpu[:, :slice_offset], 7))
    torch.testing.assert_close(
        output_cpu[:, slice_offset + slice_size :],
        torch.full_like(output_cpu[:, slice_offset + slice_size :], 7),
    )
