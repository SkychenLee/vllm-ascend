"""Stage-level (prefill / decode) accuracy + performance tests for the
HIF4 / MXFP4 fake-quant paths in Triton PagedAttention.

This exercises the *same* operator entry points used in production
(``vllm_ascend/attention/attention_v1.py``):

* **Prefill** -> ``fia_triton`` (``paged_attention``), called from
  ``forward_fused_infer_attention`` (``attn_output = fia_triton(...)``).
* **Decode (specialized split-KV kernel)** -> ``fia_triton_decode_out``
  (``paged_attention_decode_out``), called from ``full_graph_fia``.
  This kernel is restricted to ``num_q_heads in {8, 16}`` and
  ``num_kv_heads == 1`` (see ``supports_decode_specialization``), so it is
  exercised with an 8/1 head config.
* **Decode (fallback)** -> for the 64/4 head config the specialized kernel is
  not applicable, so decode runs through the prefill entry ``fia_triton``
  with a single-token query (matching the real dispatch fallback).

For each stage we report:

1. ``fia_triton`` baseline (no quant) vs the official ``torch_npu``
   ``npu_fused_infer_attention_score`` (absolute reference).
2. ``fia_triton`` + MXFP4 vs the ``fia_triton`` baseline.
3. ``fia_triton`` + HIF4  vs the ``fia_triton`` baseline.
4. Wall-clock latency of each path.

Requires an NPU device and torch_npu/triton.
"""

import time

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from vllm_ascend.ops.triton.paged_attn import (
    paged_attention as fia_triton,
    paged_attention_decode_out as fia_triton_decode_out,
)
from vllm_ascend.ops.triton.paged_attn.decode_utils import (
    DECODE_SPLIT_KV_NUM_PROGRAMS,
    build_split_kv_descriptors,
    select_decode_heads_per_program,
)

# ---------------------------------------------------------------------------
# Model / scenario config
# ---------------------------------------------------------------------------
HEAD_DIM = 128
BLOCK_SIZE = 128
BLOCK_M = 16
BLOCK_N = 128  # must be a multiple of 64 for HIF4/MXFP4
DTYPE = torch.bfloat16
NUM_AI_CORES = 32

# Prefill (and 64/4 decode fallback) head config.
PREFILL_NUM_Q_HEADS = 64
PREFILL_NUM_KV_HEADS = 4

# Specialized decode kernel only supports {8,16} q-heads / 1 kv-head.
DECODE_NUM_Q_HEADS = 8
DECODE_NUM_KV_HEADS = 1

MAX_BATCH = 32
MAX_CTX = 32 * 1024

# A small representative matrix of (batch, ctx) scenarios. Kept modest so the
# suite runs quickly and stays within NPU memory; MAX_BATCH / MAX_CTX above are
# the documented upper bounds.
#
# NOTE: the prefill Triton kernel asserts grid = total_q_blocks * num_q_heads
# <= 65535 (Ascend coreDim limit). With BLOCK_M=16 and 64 q-heads this caps
# total_q_tokens at ~16k, so the largest prefill case stays under that.
PREFILL_CASES = [
    # (batch, q_len_per_seq, kv_len_per_seq)
    pytest.param(1, 512, 512, id="prefill-bs1-512"),
    pytest.param(4, 2048, 2048, id="prefill-bs4-2k"),
    pytest.param(8, 1024, 1024, id="prefill-bs8-1k"),
]
DECODE_CASES = [
    pytest.param(1, 4096, id="decode-bs1-kv4k"),
    pytest.param(2, 8192, id="decode-bs2-kv8k"),
    pytest.param(4, 8192, id="decode-bs4-kv8k"),
]
DECODE_FALLBACK_CASES = [
    pytest.param(1, 4096, id="decode-fb-bs1-kv4k"),
    pytest.param(4, 8192, id="decode-fb-bs4-kv8k"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _npu_ok() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _cumsum(lengths, device):
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int64, device=device)
    cu[1:] = torch.tensor(lengths, dtype=torch.int64, device=device).cumsum(0)
    return cu


def _causal_mask(ctx, device):
    # torch_npu's FIA tiling rejects masks smaller than ~2048, so always use a
    # mask of at least 2048 (matching test_paged_attention_npu.py).  Both the
    # Triton path and torch_npu treat the upper triangle (non-zero) as masked.
    n = max(2048, ctx)
    return torch.triu(torch.ones(n, n), diagonal=1).to(
        torch.int8).to(device).contiguous()


def _build_paged(q_lens, kv_lens, num_q_heads, num_kv_heads, device):
    """Build paged K/V cache + block_table + contiguous TND K/V (for the
    torch_npu reference).

    Returns ``(query, k_cache, v_cache, block_table, k_contig, v_contig)``.
    The contiguous tensors hold the *same* key/value tokens as the paged cache,
    so the official torch_npu reference (run on contiguous KV) is directly
    comparable to the Triton paged path.
    """
    batch = len(q_lens)
    max_kv = max(kv_lens)
    blocks_per_seq = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch * blocks_per_seq
    flat = num_kv_heads * HEAD_DIM

    query = torch.randn(sum(q_lens), num_q_heads, HEAD_DIM, dtype=DTYPE,
                        device=device) * 0.25
    k_cache = torch.zeros(num_blocks, BLOCK_SIZE, flat, dtype=DTYPE, device=device)
    v_cache = torch.zeros_like(k_cache)

    rows = [list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq))
            for i in range(batch)]
    block_table = torch.tensor(rows, dtype=torch.int32, device=device)

    k_contig = torch.zeros(sum(kv_lens), num_kv_heads, HEAD_DIM, dtype=DTYPE,
                           device=device)
    v_contig = torch.zeros_like(k_contig)

    for s, kv_len in enumerate(kv_lens):
        k = torch.randn(kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE,
                        device=device) * 0.25
        v = torch.randn(kv_len, num_kv_heads, HEAD_DIM, dtype=DTYPE,
                        device=device) * 0.25
        k_contig[sum(kv_lens[:s]):sum(kv_lens[:s]) + kv_len] = k
        v_contig[sum(kv_lens[:s]):sum(kv_lens[:s]) + kv_len] = v
        for t in range(kv_len):
            lb, off = t // BLOCK_SIZE, t % BLOCK_SIZE
            pb = int(block_table[s, lb])
            k_cache[pb, off] = k[t].reshape(-1)
            v_cache[pb, off] = v[t].reshape(-1)

    return query, k_cache, v_cache, block_table, k_contig, v_contig


def _build_tnd_kv(q_lens, kv_lens, num_kv_heads, device):
    """Build contiguous TND key/value (for torch_npu reference and contig KV)."""
    k = torch.randn(sum(kv_lens), num_kv_heads, HEAD_DIM, dtype=DTYPE,
                    device=device) * 0.25
    v = torch.randn_like(k)
    return k, v


def _time(fn, warmup=3, iters=10):
    """Median wall-clock latency (ms) of a no-arg callable on NPU."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.time()
        fn()
        torch.npu.synchronize()
        ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def _stats(name, a, b):
    a = a.float().cpu()
    b = b.float().cpu()
    err = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
    max_abs = err.max().item()
    print(f"  [{name}] max_abs={max_abs:.6e} mean_abs={err.mean().item():.6e} "
          f"cosine={cos:.6f}")
    return max_abs


# ---------------------------------------------------------------------------
# torch_npu official reference (absolute ground truth)
# ---------------------------------------------------------------------------
def _torch_npu_ref(query, k_contig, v_contig, q_lens, kv_lens,
                   num_q_heads, num_kv_heads, device, scale):
    """Official npu_fused_infer_attention_score on contiguous TND K/V.

    Uses the verified contiguous call shape (block_table=None) from
    test_paged_attention_npu.py: both actual_seq_lengths are cumulative,
    sparse_mode=3 with a (max_len, max_len) int8 causal mask; single-token
    queries use sparse_mode=0 / no mask.
    """
    def _cu(ls):
        total = 0
        out = []
        for l in ls:
            total += int(l)
            out.append(total)
        return out

    single_token = max(q_lens) == 1
    if single_token:
        atten_mask = None
        sparse_mode = 0
    else:
        atten_mask = _causal_mask(max(max(q_lens), max(kv_lens)), device)
        sparse_mode = 3
    return torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=k_contig,
        value=v_contig,
        num_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        scale=scale,
        atten_mask=atten_mask,
        block_table=None,
        input_layout="TND",
        block_size=BLOCK_SIZE,
        actual_seq_lengths=_cu(q_lens),
        actual_seq_lengths_kv=_cu(kv_lens),
        sparse_mode=sparse_mode,
    )[0]


# ===========================================================================
# PREFILL stage  (fia_triton, 64/4 heads)
# ===========================================================================
@pytest.mark.parametrize("batch,q_len,kv_len", PREFILL_CASES)
def test_prefill_stage(batch, q_len, kv_len):
    if not _npu_ok():
        pytest.skip("NPU is required")
    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    device = "npu"
    nq, nkv = PREFILL_NUM_Q_HEADS, PREFILL_NUM_KV_HEADS
    scale = HEAD_DIM ** -0.5
    q_lens = [q_len] * batch
    kv_lens = [kv_len] * batch

    query, k_cache, v_cache, block_table, k_contig, v_contig = _build_paged(
        q_lens, kv_lens, nq, nkv, device)
    cu_q = _cumsum(q_lens, device)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int64, device=device)
    atten_mask = _causal_mask(q_len, device)

    common = dict(
        query=query, key_cache=k_cache, value_cache=v_cache,
        block_table=block_table, actual_seq_qlen=cu_q, actual_seq_kvlen=kv_lens_t,
        num_q_heads=nq, num_kv_heads=nkv, softmax_scale=scale,
        block_size=BLOCK_SIZE, block_m=BLOCK_M, block_n=BLOCK_N,
        atten_mask=atten_mask,
    )

    out_base = fia_triton(**common, use_mxfp4_p=False, use_hif4_p=False)
    out_mxfp4 = fia_triton(**common, use_mxfp4_p=True, use_hif4_p=False)
    out_hif4 = fia_triton(**common, use_mxfp4_p=False, use_hif4_p=True)
    ref = _torch_npu_ref(query, k_contig, v_contig, q_lens, kv_lens,
                         nq, nkv, device, scale)
    torch.npu.synchronize()

    print(f"\n=== PREFILL bs={batch} q={q_len} kv={kv_len} (heads {nq}/{nkv}) ===")
    _stats("fia_triton vs torch_npu_ref", out_base, ref)
    _stats("fia_triton+mxfp4 vs fia_triton_base", out_mxfp4, out_base)
    _stats("fia_triton+hif4  vs fia_triton_base", out_hif4, out_base)

    t_base = _time(lambda: fia_triton(**common, use_mxfp4_p=False, use_hif4_p=False))
    t_mxfp4 = _time(lambda: fia_triton(**common, use_mxfp4_p=True, use_hif4_p=False))
    t_hif4 = _time(lambda: fia_triton(**common, use_mxfp4_p=False, use_hif4_p=True))
    print(f"  [latency ms] base={t_base:.3f}  mxfp4={t_mxfp4:.3f}  hif4={t_hif4:.3f}")

    # --- assertions ---
    assert torch.isfinite(out_base.float().cpu()).all()
    assert torch.isfinite(out_mxfp4.float().cpu()).all()
    assert torch.isfinite(out_hif4.float().cpu()).all()
    # baseline must be close to the official reference (Triton vs torch_npu)
    base_vs_ref = _stats("fia_triton vs torch_npu_ref (assert)", out_base, ref)
    assert base_vs_ref < 0.5, f"fia_triton diverges from torch_npu ref: {base_vs_ref}"
    # quantization perturbs output but stays bounded
    mxfp4_err = (out_mxfp4.float() - out_base.float()).abs().max().item()
    hif4_err = (out_hif4.float() - out_base.float()).abs().max().item()
    assert mxfp4_err < 1.0
    assert hif4_err < 1.0


# ===========================================================================
# DECODE stage  (specialized split-KV kernel fia_triton_decode_out, 8/1 heads)
# ===========================================================================
def _decode_split_kv_workspace(num_q_heads, device):
    partial_out = torch.empty(
        DECODE_SPLIT_KV_NUM_PROGRAMS, num_q_heads, HEAD_DIM,
        dtype=torch.float32, device=device)
    partial_lse = torch.empty(
        DECODE_SPLIT_KV_NUM_PROGRAMS, num_q_heads,
        dtype=torch.float32, device=device)
    return partial_out, partial_lse


@pytest.mark.parametrize("batch,kv_len", DECODE_CASES)
def test_decode_stage_specialized(batch, kv_len):
    if not _npu_ok():
        pytest.skip("NPU is required")
    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    device = "npu"
    nq, nkv = DECODE_NUM_Q_HEADS, DECODE_NUM_KV_HEADS
    scale = HEAD_DIM ** -0.5
    q_lens = [1] * batch
    kv_lens = [kv_len] * batch

    query, k_cache, v_cache, block_table, k_contig, v_contig = _build_paged(
        q_lens, kv_lens, nq, nkv, device)
    cu_q = _cumsum(q_lens, device)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int64, device=device)

    heads_per_program = select_decode_heads_per_program(batch, nq, NUM_AI_CORES)
    work_desc_list, seq_desc_list, _ = build_split_kv_descriptors(
        kv_lens, block_size=BLOCK_SIZE)
    work_desc = torch.tensor(work_desc_list, dtype=torch.int32, device=device)
    seq_desc = torch.tensor(seq_desc_list, dtype=torch.int32, device=device)

    def run(use_mxfp4, use_hif4):
        out = torch.empty_like(query)
        ws = _decode_split_kv_workspace(nq, device)
        fia_triton_decode_out(
            query=query, key_cache=k_cache, value_cache=v_cache,
            block_table=block_table, actual_seq_qlen=cu_q,
            actual_seq_kvlen=kv_lens_t, num_q_heads=nq, num_kv_heads=nkv,
            softmax_scale=scale, block_size=BLOCK_SIZE, output=out,
            use_mxfp4_p=use_mxfp4, use_hif4_p=use_hif4,
            heads_per_program_override=heads_per_program,
            split_kv_num_programs=DECODE_SPLIT_KV_NUM_PROGRAMS,
            split_kv_workspace=ws,
            split_kv_descriptors=(work_desc, seq_desc),
        )
        return out

    out_base = run(False, False)
    out_mxfp4 = run(True, False)
    out_hif4 = run(False, True)
    ref = _torch_npu_ref(query, k_contig, v_contig, q_lens, kv_lens,
                         nq, nkv, device, scale)
    torch.npu.synchronize()

    print(f"\n=== DECODE(specialized) bs={batch} kv={kv_len} (heads {nq}/{nkv}) ===")
    _stats("fia_decode vs torch_npu_ref", out_base, ref)
    _stats("fia_decode+mxfp4 vs base", out_mxfp4, out_base)
    _stats("fia_decode+hif4  vs base", out_hif4, out_base)

    t_base = _time(lambda: run(False, False))
    t_mxfp4 = _time(lambda: run(True, False))
    t_hif4 = _time(lambda: run(False, True))
    print(f"  [latency ms] base={t_base:.3f}  mxfp4={t_mxfp4:.3f}  hif4={t_hif4:.3f}")

    assert torch.isfinite(out_base.float().cpu()).all()
    assert torch.isfinite(out_mxfp4.float().cpu()).all()
    assert torch.isfinite(out_hif4.float().cpu()).all()
    base_vs_ref = _stats("fia_decode vs torch_npu_ref (assert)", out_base, ref)
    assert base_vs_ref < 0.5, f"fia_decode diverges from torch_npu ref: {base_vs_ref}"


# ===========================================================================
# DECODE stage  (fallback: 64/4 heads -> single-token fia_triton prefill path)
# ===========================================================================
@pytest.mark.parametrize("batch,kv_len", DECODE_FALLBACK_CASES)
def test_decode_stage_fallback(batch, kv_len):
    if not _npu_ok():
        pytest.skip("NPU is required")
    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    device = "npu"
    nq, nkv = PREFILL_NUM_Q_HEADS, PREFILL_NUM_KV_HEADS
    scale = HEAD_DIM ** -0.5
    q_lens = [1] * batch
    kv_lens = [kv_len] * batch

    query, k_cache, v_cache, block_table, k_contig, v_contig = _build_paged(
        q_lens, kv_lens, nq, nkv, device)
    cu_q = _cumsum(q_lens, device)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int64, device=device)
    # single-token decode has no causal interaction within the query tile.
    atten_mask = None

    common = dict(
        query=query, key_cache=k_cache, value_cache=v_cache,
        block_table=block_table, actual_seq_qlen=cu_q, actual_seq_kvlen=kv_lens_t,
        num_q_heads=nq, num_kv_heads=nkv, softmax_scale=scale,
        block_size=BLOCK_SIZE, block_m=BLOCK_M, block_n=BLOCK_N,
        atten_mask=atten_mask,
    )

    out_base = fia_triton(**common, use_mxfp4_p=False, use_hif4_p=False)
    out_mxfp4 = fia_triton(**common, use_mxfp4_p=True, use_hif4_p=False)
    out_hif4 = fia_triton(**common, use_mxfp4_p=False, use_hif4_p=True)
    ref = _torch_npu_ref(query, k_contig, v_contig, q_lens, kv_lens,
                         nq, nkv, device, scale)
    torch.npu.synchronize()

    print(f"\n=== DECODE(fallback 64/4) bs={batch} kv={kv_len} ===")
    _stats("fia_triton vs torch_npu_ref", out_base, ref)
    _stats("fia_triton+mxfp4 vs base", out_mxfp4, out_base)
    _stats("fia_triton+hif4  vs base", out_hif4, out_base)

    t_base = _time(lambda: fia_triton(**common, use_mxfp4_p=False, use_hif4_p=False))
    t_mxfp4 = _time(lambda: fia_triton(**common, use_mxfp4_p=True, use_hif4_p=False))
    t_hif4 = _time(lambda: fia_triton(**common, use_mxfp4_p=False, use_hif4_p=True))
    print(f"  [latency ms] base={t_base:.3f}  mxfp4={t_mxfp4:.3f}  hif4={t_hif4:.3f}")

    assert torch.isfinite(out_base.float().cpu()).all()
    assert torch.isfinite(out_mxfp4.float().cpu()).all()
    assert torch.isfinite(out_hif4.float().cpu()).all()
    base_vs_ref = _stats("fia_triton vs torch_npu_ref (assert)", out_base, ref)
    assert base_vs_ref < 0.5, f"fia_triton diverges from torch_npu ref: {base_vs_ref}"
