import gc

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


def bgmv_shrink_cpu_impl(
    x: torch.Tensor, w: torch.Tensor, indices: torch.Tensor, y: torch.tensor, scaling: float
) -> torch.Tensor:
    W = w[indices, :, :].transpose(-1, -2).to(torch.float32)
    z = torch.bmm(x.unsqueeze(1).to(torch.float32), W).squeeze()
    y[:, :] += z * scaling
    return y


@torch.inference_mode()
def test_bgmv_shrink():
    B = 1
    x = torch.randn([B, 128], dtype=torch.float16)
    w = torch.randn([64, 16, 128], dtype=torch.float16)
    indices = torch.zeros([B], dtype=torch.int64)
    y = torch.zeros([B, 16])

    x_npu = x.npu()
    w_npu = w.npu()
    indices_npu = indices.npu()
    y_npu = y.npu()

    y = bgmv_shrink_cpu_impl(x, w, indices, y, 0.5)
    torch.ops._C_ascend.bgmv_shrink(x_npu, w_npu, indices_npu, y_npu, 0.5)

    # Compare the results.
    torch.testing.assert_close(y_npu.cpu(), y.cpu(), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size,hidden_dim", [(1, 4096), (6, 2048), (6, 4096)])
@torch.inference_mode()
def test_bgmv_shrink_rank16_decode_shape(dtype: torch.dtype, batch_size: int, hidden_dim: int):
    """Cover the A2 rank-parallel path used by top-k MoE decode."""
    num_loras = 4
    rank = 16
    scaling = 0.5
    x = torch.randn([batch_size, hidden_dim], dtype=dtype)
    w = torch.randn([num_loras, rank, hidden_dim], dtype=dtype)
    indices = torch.arange(batch_size, dtype=torch.int64) % num_loras
    y = torch.zeros([batch_size, rank], dtype=torch.float32)

    expected = bgmv_shrink_cpu_impl(x, w, indices, y.clone(), scaling)
    actual = y.npu()
    torch.ops._C_ascend.bgmv_shrink(x.npu(), w.npu(), indices.npu(), actual, scaling)

    atol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    rtol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)
