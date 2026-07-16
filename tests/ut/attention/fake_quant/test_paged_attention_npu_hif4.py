"""Unit tests for the HIF4 fake-quant path in Triton PagedAttention.

These tests cover three things:

1. **Kernel-level accuracy**: a standalone Triton kernel wrapping ``to_hif4``
   is compared element-wise against the torch reference
   ``quant_dequant_hif4`` from ``vllm_ascend.quantization.utils`` on random
   probability-like tensors.  This is the tightest check that the Triton port
   matches the reference math.
2. **End-to-end sanity**: running ``paged_attention`` (prefill) and
   ``paged_attention_decode_out`` (decode) with ``use_hif4_p=True`` produces
   finite output and deviates from the unquantized baseline by a sensible
   amount (and crucially, is *different* from the MXFP4 path).
3. **Performance**: a rough wall-clock timing of hif4 vs baseline decode so
   regressions are visible; the assertion only checks it runs without error.

Requires an NPU device and torch_npu/triton.
"""

import time

import pytest
import torch
import triton
import triton.language as tl

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from vllm_ascend.ops.triton.paged_attn.paged_attention_npu import (
    paged_attention,
    paged_attention_decode_out,
    to_hif4,
)
from vllm_ascend.quantization.utils import quant_dequant_hif4

NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 128
DTYPE = torch.bfloat16
BLOCK_SIZE = 128
BLOCK_M = 16
BLOCK_N = 128  # multiple of 64, required by HIF4


def _npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


# ---------------------------------------------------------------------------
# Standalone kernel that applies to_hif4 to a whole [M, N] tensor.
# One program per row so each row is a self-contained BLOCK_M=1, BLOCK_N tile.
# ---------------------------------------------------------------------------
@triton.jit
def _hif4_full_kernel(
    IN,
    OUT,
    M,
    stride_in_m,
    stride_out_m,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    x = tl.load(IN + row * stride_in_m + offs_n)
    # to_hif4 expects a 2D [BLOCK_M, BLOCK_N] tile; promote the row to 2D.
    x2d = tl.reshape(x, (1, BLOCK_N))
    y = to_hif4(x2d, 1, BLOCK_N)
    y = tl.reshape(y, (BLOCK_N,))
    tl.store(OUT + row * stride_out_m + offs_n, y)


def _triton_hif4(x: torch.Tensor) -> torch.Tensor:
    """Apply the Triton ``to_hif4`` to a 2D [M, N] tensor, N multiple of 64."""
    assert x.dim() == 2
    M, N = x.shape
    assert N % 64 == 0
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _hif4_full_kernel[(M,)](
        x_c,
        out,
        M,
        x_c.stride(0),
        out.stride(0),
        BLOCK_N=N,
    )
    torch.npu.synchronize()
    return out


# ---------------------------------------------------------------------------
# 1. Kernel-level accuracy against torch reference.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M,N", [(16, 64), (16, 128), (32, 128), (8, 256)])
def test_to_hif4_matches_torch_reference(M, N):
    if not _npu_available():
        pytest.skip("NPU is required")
    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    device = "npu"

    # probability-like non-negative input in [0, 1]; include exact zeros.
    p = torch.rand(M, N, dtype=torch.float32, device=device)
    p = p * 0.9 + 1e-3
    p[0, :4] = 0.0  # exercise the zero / log2-protection path

    ref = quant_dequant_hif4(p, "hifx4", axe=-1)  # float32 reference
    tri = _triton_hif4(p)

    ref_cpu = ref.float().cpu()
    tri_cpu = tri.float().cpu()

    abs_err = (tri_cpu - ref_cpu).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()

    print(
        f"\n[HIF4 kernel vs torch] M={M} N={N}: "
        f"max_abs_err={max_err:.6e} mean_abs_err={mean_err:.6e}"
    )

    assert torch.isfinite(tri_cpu).all(), "Triton HIF4 output has NaN/Inf"
    # HIF4 involves bf16 scale rounding in the reference; allow a small margin.
    # The largest representable quant step at p~1 is 0.25*scale, and scales are
    # shared per 8 channels, so ~3e-2 is a generous upper bound.
    assert max_err < 3e-2, f"HIF4 max abs error too large: {max_err}"


# ---------------------------------------------------------------------------
# Helpers for end-to-end prefill/decode.
# ---------------------------------------------------------------------------
def _build_paged_inputs(q_lens, kv_lens, num_q_heads, num_kv_heads, head_dim,
                        dtype, device):
    batch_size = len(q_lens)
    capacity = max(max(kv_lens), 1) + BLOCK_SIZE
    blocks_per_seq = (capacity + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch_size * blocks_per_seq
    flat = num_kv_heads * head_dim

    query = torch.randn(sum(q_lens), num_q_heads, head_dim, dtype=dtype,
                        device=device) * 0.25
    key_cache = torch.zeros(num_blocks, BLOCK_SIZE, flat, dtype=dtype,
                            device=device)
    value_cache = torch.zeros_like(key_cache)

    rows = [list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq))
            for i in range(batch_size)]
    block_table = torch.tensor(rows, dtype=torch.int32, device=device)

    for seq_idx, kv_len in enumerate(kv_lens):
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype,
                        device=device) * 0.25
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype,
                        device=device) * 0.25
        for t in range(kv_len):
            lb = t // BLOCK_SIZE
            off = t % BLOCK_SIZE
            pb = int(block_table[seq_idx, lb])
            key_cache[pb, off] = k[t].reshape(-1)
            value_cache[pb, off] = v[t].reshape(-1)

    cu_q = torch.zeros(batch_size + 1, dtype=torch.int64, device=device)
    cu_q[1:] = torch.tensor(q_lens, dtype=torch.int64, device=device).cumsum(0)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int64, device=device)
    return query, key_cache, value_cache, block_table, cu_q, kv_lens_t


def _causal_mask(max_len, device):
    return torch.triu(torch.ones(max_len, max_len), diagonal=1).to(
        torch.int8).to(device).contiguous()


# ---------------------------------------------------------------------------
# 2. End-to-end prefill: hif4 on vs off, sanity + distinct from mxfp4.
# ---------------------------------------------------------------------------
def test_prefill_hif4_end_to_end():
    if not _npu_available():
        pytest.skip("NPU is required")
    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    device = "npu"
    scale = HEAD_DIM ** -0.5

    q_lens = [128]
    kv_lens = [128]
    query, k_cache, v_cache, block_table, cu_q, kv_lens_t = _build_paged_inputs(
        q_lens, kv_lens, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, DTYPE, device)
    atten_mask = _causal_mask(128, device)

    common = dict(
        query=query,
        key_cache=k_cache,
        value_cache=v_cache,
        block_table=block_table,
        actual_seq_qlen=cu_q,
        actual_seq_kvlen=kv_lens_t,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        softmax_scale=scale,
        block_size=BLOCK_SIZE,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        atten_mask=atten_mask,
    )

    base = paged_attention(**common, use_mxfp4_p=False, use_hif4_p=False)
    hif4 = paged_attention(**common, use_mxfp4_p=False, use_hif4_p=True)
    mxfp4 = paged_attention(**common, use_mxfp4_p=True, use_hif4_p=False)
    torch.npu.synchronize()

    base_cpu = base.float().cpu()
    hif4_cpu = hif4.float().cpu()
    mxfp4_cpu = mxfp4.float().cpu()

    assert torch.isfinite(hif4_cpu).all(), "HIF4 prefill output has NaN/Inf"

    hif4_err = (hif4_cpu - base_cpu).abs()
    mxfp4_err = (mxfp4_cpu - base_cpu).abs()
    print(
        f"\n[prefill hif4] vs base: max_abs={hif4_err.max():.6e} "
        f"mean_abs={hif4_err.mean():.6e}"
    )
    print(
        f"[prefill mxfp4] vs base: max_abs={mxfp4_err.max():.6e} "
        f"mean_abs={mxfp4_err.mean():.6e}"
    )

    # Quantization must perturb the output, but not blow it up.
    assert hif4_err.max().item() < 1.0
    # HIF4 and MXFP4 are different algorithms -> different outputs.
    assert not torch.allclose(hif4_cpu, mxfp4_cpu, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. End-to-end decode + rough performance timing.
# ---------------------------------------------------------------------------
def test_decode_hif4_end_to_end_and_perf():
    if not _npu_available():
        pytest.skip("NPU is required")
    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    device = "npu"
    scale = HEAD_DIM ** -0.5

    q_lens = [32]
    kv_lens = [32768]
    query, k_cache, v_cache, block_table, cu_q, kv_lens_t = _build_paged_inputs(
        q_lens, kv_lens, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, DTYPE, device)

    def run(use_hif4):
        out = torch.empty_like(query)
        return paged_attention_decode_out(
            query=query,
            key_cache=k_cache,
            value_cache=v_cache,
            block_table=block_table,
            actual_seq_qlen=cu_q,
            actual_seq_kvlen=kv_lens_t,
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            softmax_scale=scale,
            block_size=BLOCK_SIZE,
            output=out,
            use_mxfp4_p=False,
            use_hif4_p=use_hif4,
        )

    base = run(False)
    hif4 = run(True)
    torch.npu.synchronize()

    # warmup + timing
    for _ in range(3):
        run(True)
    torch.npu.synchronize()
    t0 = time.time()
    iters = 20
    for _ in range(iters):
        run(True)
    torch.npu.synchronize()
    hif4_ms = (time.time() - t0) / iters * 1000.0

    base_cpu = base.float().cpu()
    hif4_cpu = hif4.float().cpu()
    err = (hif4_cpu - base_cpu).abs()
    print(
        f"\n[decode hif4] vs base: max_abs={err.max():.6e} "
        f"mean_abs={err.mean():.6e} | hif4_decode latency~{hif4_ms:.3f} ms"
    )

    assert torch.isfinite(hif4_cpu).all(), "HIF4 decode output has NaN/Inf"
    assert err.max().item() < 1.0


# ---------------------------------------------------------------------------
# 4. Mutual-exclusion guard.
# ---------------------------------------------------------------------------
def test_hif4_and_mxfp4_mutually_exclusive():
    if not _npu_available():
        pytest.skip("NPU is required")
    device = "npu"
    q_lens = [1]
    kv_lens = [64]
    query, k_cache, v_cache, block_table, cu_q, kv_lens_t = _build_paged_inputs(
        q_lens, kv_lens, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, DTYPE, device)
    out = torch.empty_like(query)
    with pytest.raises(AssertionError):
        paged_attention_decode_out(
            query=query,
            key_cache=k_cache,
            value_cache=v_cache,
            block_table=block_table,
            actual_seq_qlen=cu_q,
            actual_seq_kvlen=kv_lens_t,
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            softmax_scale=HEAD_DIM ** -0.5,
            block_size=BLOCK_SIZE,
            output=out,
            use_mxfp4_p=True,
            use_hif4_p=True,
        )
