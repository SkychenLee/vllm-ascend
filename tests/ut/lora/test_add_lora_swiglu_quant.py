# SPDX-License-Identifier: Apache-2.0
"""Numeric tests for the fused w13 epilogue kernel (delta add + swiglu + int8 quant).

Kernel under test: csrc/kernels/add_lora_swiglu_quant.cpp, reached directly
through torch.ops._C_ascend.add_lora_swiglu_quant -- the same call sites are
the sequential and aux-stream overlap w13 epilogues in
vllm_ascend/lora/quant_moe.py (replaces swiglu -> npu_dynamic_quant there).

Semantics tested (gate = gate_up[:, :W], up = gate_up[:, W:], fp32 math):
    g = gate + delta_gate -> clamp(max=limit)             if limit > 0
    u = up   + delta_up   -> clamp(min=-limit, max=limit) if limit > 0
    act_f32 = u * g * sigmoid(g)    (AscendC SwiGLU, beta=1, same silu-on-gate
                                     convention as npu_swiglu / gmm swiglu kernels)
    act   = bf16/fp16(act_f32)      [T, W]  -> W2 lora shrink input
    scale = rowmax(act_f32) / 127   [T] fp32
    y     = int8(rint(fp16(act_f32 * 127 / rowmax)))  [T, W]; fp32 -> fp16 ->
            int8 double rounding (no fp32->int8 cast on device), per-row
            recip = Div(127, rowmax) then broadcast via Brcb
Zero-max rows: scale is 0 and the row dequantizes to exactly 0 whatever the
(garbage) int8 payload -- the y*scale product downstream is what matters.

Kernel/binding-declared constraints and mirrored grid constants:
  - binding (torch_binding.cpp): 2-D contiguous bf16/fp16 [T, 2W], W a
    multiple of 64 in [64, 4096] (kSwigluQuantMaxWidth == kernel MAX_WIDTH
    = 64 * 64). The W % 64 tail of the pass-1 WholeReduceMax is broken for
    every tail != 0 (bz-ascend sweep 2026-10-14: UB-misaligned vector faults
    for tail 8/40/56/63, SILENT row-max corruption for tail 16/48, tail 32
    correct only by luck), so out-of-contract widths must never reach the
    kernel -- test_illegal_width_rejected below pins the binding rejection.
  - TILE_ELEMENTS=8192, MAX_TOKENS_PER_TILE=16: rowsPerTile = min(8192//W, 16),
    a block walks its token chunk in tiles of that size
  - numTokensPerCore = ceil(B / aiv_num); aiv count unknown host-side, so the
    batch sweep straddles the plausible 48/52/56/64 core counts and includes
    B = 1024 (> 16 * 64) to force multi-tile blocks on any part
"""

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401

_HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
pytestmark = pytest.mark.skipif(
    not _HAS_NPU, reason="add_lora_swiglu_quant requires an NPU"
)

DEV = "npu:0"

# magnitudes: gate/up ~ N(0, 0.495) after the optional delta, so |act| tails
# stay <= ~4.5; one bf16 ulp there is 0.03125 (fp32-vs-fp32 sigmoid drift can
# flip the final bf16 cast by 1 ulp), 4e-2 covers that and still catches any
# semantic break (wrong half, missing clamp, wrong scale -> errors of O(1))
ACT_TOL = 4e-2
# int8 payload may differ from the fp32-oracle double rounding at rounding
# boundaries (device fp32 sigmoid is approximate); bounded +-1 and rare
PAYLOAD_MAX_DIFF = 1
PAYLOAD_MAX_MISMATCH_FRAC = 0.02
# per-row quant step == scale; fp16 intermediate costs <= 127*2^-11 ~= 0.06
# step, RINT <= half a step; 0.6 + 1e-3 absolute slack for the tiny-scale rows
DEQUANT_STEP_FRAC = 0.6
DEQUANT_ABS_SLACK = 1e-3

# rowsPerTile = min(8192//W, 16) transitions and the full = W // 64 partial
# counts of the pass-1 max-reduce, over LEGAL (W % 64 == 0) widths only:
#   64  -> 16 rows (capped), full=1    | 1024 -> 8 rows,  full=16
#   512 -> 16 rows, full=8             | 2048 -> 4 rows,  full=32
#   960 -> 8 rows, full=15 (odd)       | 2944 -> 2 rows,  full=46 (odd)
#   4096 -> 2 rows, exactly 64 partials (pass-2 WholeReduceMax mask max)
WIDTH_SWEEP = [64, 512, 960, 1024, 2048, 2944, 4096]

# 1/2 tokens, tile-size boundaries around 16, and both sides of the plausible
# aiv counts so numTokensPerCore crosses 1 (B > aiv) on every part
BATCH_SWEEP = [1, 2, 8, 15, 16, 17, 31, 32, 47, 48, 49, 52, 53, 56, 57, 64, 111, 112, 113]


def _make_inputs(batch: int, width: int, dtype, seed: int, with_delta: bool = True):
    torch.manual_seed(seed)
    gate_up = (torch.randn(batch, 2 * width, device=DEV) * 0.35).to(dtype)
    delta = None
    if with_delta:
        delta = (torch.randn(batch, 2 * width, device=DEV) * 0.35).to(dtype)
    return gate_up, delta


def _oracle(gate_up, delta, limit):
    """fp32 replication of the kernel math, same operation order.

    Returns (act_f32, scale, y_ref); y_ref is garbage (nan) on zero-max rows
    because Div(127, 0) = inf -- the kernel writes garbage there too and the
    y*scale product is what stays correct.
    """
    gu = gate_up.float()
    if delta is not None:
        gu = gu + delta.float()
    w = gu.shape[1] // 2
    g, u = gu[:, :w], gu[:, w:]
    if limit > 0.0:
        g = g.clamp(max=limit)
        u = u.clamp(-limit, limit)
    act = u * (g * torch.sigmoid(g))
    rowmax = act.abs().amax(dim=1)
    scale = rowmax * (1.0 / 127.0)  # kernel: Muls(rowMax, 1/127)
    recip = 127.0 / rowmax          # kernel: Div(127, rowMax)
    y_ref = (act * recip.unsqueeze(1)).half().float().round()
    return act, scale, y_ref


def _check(gate_up, delta, limit, tag=""):
    act, y, scale = torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, delta, limit)
    batch, w = gate_up.shape[0], gate_up.shape[1] // 2
    assert act.shape == (batch, w) and act.dtype == gate_up.dtype
    assert y.shape == (batch, w) and y.dtype == torch.int8
    assert scale.shape == (batch,) and scale.dtype == torch.float32

    ref_act, ref_scale, ref_y = _oracle(gate_up, delta, limit)

    # act: bf16/fp16 of the fp32 oracle, +-1 dtype ulp of slack for sigmoid drift
    err = (act.float() - ref_act.to(gate_up.dtype).float()).abs().max().item()
    assert err < ACT_TOL, f"{tag}: act max_err={err:.3e}"

    # scale: order-free max reduce, only the division form may differ
    assert torch.allclose(scale, ref_scale, rtol=1e-3, atol=1e-7), \
        f"{tag}: scale mismatch, max={(scale - ref_scale).abs().max()}"

    # y payload: exact double-rounded oracle except +-1 at rounding boundaries;
    # zero-max rows hold device garbage by contract (scale is 0, see the header)
    valid = ref_scale > 0
    if bool(valid.any()):
        d = (y.float() - ref_y)[valid].abs()
        assert d.max().item() <= PAYLOAD_MAX_DIFF, f"{tag}: y off by >1"
        frac = (d > 0).float().mean().item()
        assert frac < PAYLOAD_MAX_MISMATCH_FRAC, f"{tag}: y mismatch frac={frac:.4f}"
        assert y[valid].min().item() >= -127, f"{tag}: y overflowed the int8 range"

    # the real contract: dequantized output tracks the fp32 activations within
    # one quant step on every row, including zero-max rows (garbage * 0 == 0)
    deq = y.float() * scale.unsqueeze(1)
    bound = DEQUANT_STEP_FRAC * ref_scale.unsqueeze(1) + DEQUANT_ABS_SLACK
    assert (deq - ref_act).abs().le(bound).all(), f"{tag}: dequant out of bound"
    print(f"{tag}: act_err={err:.3e}")
    return act, y, scale


# ------------------------------------------------------------------ core ---


@pytest.mark.parametrize("width", WIDTH_SWEEP)
def test_width_sweep(width):
    gate_up, delta = _make_inputs(32, width, torch.bfloat16, seed=width)
    _check(gate_up, delta, 0.0, tag=f"W={width}")


@pytest.mark.parametrize("limit", [0.0, 0.75, 2.0])
@pytest.mark.parametrize("with_delta", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dtypes_delta_limit(dtype, with_delta, limit):
    # limit 0.75 saturates most of the N(0, 0.495) mass, 2.0 clips the tails;
    # without delta the kernel runs its hasDelta=0 branch (no inQueueDelta_)
    gate_up, delta = _make_inputs(16, 1024, dtype, seed=77, with_delta=with_delta)
    _check(gate_up, delta, limit,
           tag=f"{dtype} delta={with_delta} limit={limit}")


# ----------------------------------------------------------------- batch ---


@pytest.mark.parametrize("batch", BATCH_SWEEP)
def test_batch_sweep(batch):
    # W=512 -> rowsPerTile 16; numTokensPerCore = ceil(batch / aiv_num) splits
    # the batch across blocks, block tails exercise partial tiles
    gate_up, delta = _make_inputs(batch, 512, torch.bfloat16, seed=batch)
    _check(gate_up, delta, 0.0, tag=f"B={batch}")


@pytest.mark.parametrize("batch", [769, 1024])
def test_multi_tile_blocks(batch):
    # 1024 > 16 * 64 guarantees numTokensPerCore > rowsPerTile(512) = 16 on
    # any part with <= 64 AIV cores, so blocks run their tile loop more than
    # once; 769 does the same on 48-core parts
    gate_up, delta = _make_inputs(batch, 512, torch.bfloat16, seed=batch)
    _check(gate_up, delta, 0.0, tag=f"B={batch} multi-tile")


# ------------------------------------------------------------ behaviors ---


def test_delta_zeros_bitwise_equal_none():
    # adding 0.0 in fp32 must not perturb a single bit vs the hasDelta=0 branch
    gate_up, _ = _make_inputs(16, 512, torch.bfloat16, seed=5)
    zeros = torch.zeros_like(gate_up)
    a1, y1, s1 = torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, None, 0.0)
    a2, y2, s2 = torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, zeros, 0.0)
    assert torch.equal(a1, a2) and torch.equal(y1, y2) and torch.equal(s1, s2)


def test_determinism_two_launches():
    gate_up, delta = _make_inputs(64, 1024, torch.bfloat16, seed=3)
    for limit in (0.0, 2.0):
        r1 = torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, delta, limit)
        r2 = torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, delta, limit)
        assert all(torch.equal(x, z) for x, z in zip(r1, r2)), \
            f"non-deterministic output, limit={limit}"


def test_zero_activation_rows():
    # a token with an all-zero gate half has silu(0) = 0 -> whole act row 0:
    # scale must be exactly 0 and the row must dequantize to exactly 0 no
    # matter what int8 bytes the div-by-zero recip left in y
    gate_up, delta = _make_inputs(8, 512, torch.bfloat16, seed=11)
    gate_up[:3, : gate_up.shape[1] // 2] = 0.0
    delta[:3, : gate_up.shape[1] // 2] = 0.0
    act, y, scale = _check(gate_up, delta, 0.0, tag="zero rows")
    assert torch.equal(scale[:3], torch.zeros(3, device=DEV, dtype=torch.float32))
    assert torch.equal(act[:3], torch.zeros_like(act[:3]))
    assert torch.equal(y[:3].float() * scale[:3, None],
                       torch.zeros(3, 512, device=DEV))


def test_parity_unfused_production_path():
    # the fused epilogue must reproduce what quant_moe.py does without it:
    # bf16 clamp -> npu_swiglu -> per-row absmax/127 dynamic quant.
    # Also the orientation guard: a wrong silu half shows up here as O(1) error.
    gate_up, delta = _make_inputs(16, 512, torch.bfloat16, seed=23)
    limit = 2.0
    act, _, scale = _check(gate_up, delta, limit, tag="parity")

    gu = (gate_up.float() + delta.float()).bfloat16()
    g, u = gu.chunk(2, dim=-1)
    g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    act_ref = torch_npu.npu_swiglu(torch.cat((g, u), dim=-1))
    assert torch.allclose(act.float(), act_ref.float(), rtol=2e-2, atol=2e-2)

    scale_ref = act_ref.float().abs().amax(dim=1) / 127.0
    assert torch.allclose(scale, scale_ref, rtol=2e-2, atol=1e-6)


def test_empty_batch():
    # binding returns the shaped outputs without launching the kernel
    gate_up = torch.empty(0, 1024, device=DEV, dtype=torch.bfloat16)
    act, y, scale = torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, None, 0.0)
    assert act.shape == (0, 512) and act.dtype == torch.bfloat16
    assert y.shape == (0, 512) and y.dtype == torch.int8
    assert scale.shape == (0,) and scale.dtype == torch.float32


# ------------------------------------------------------------- rejections ---


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda gu: gu.float(), "bf16 or fp16"),
        (lambda gu: gu.unsqueeze(0), "2-D"),
        (lambda gu: gu.t(), "contiguous"),
        # fresh allocation: a [:, :129] view of an even-width row would die on
        # the contiguity check first (binding order: dim/contig/dtype/even)
        (lambda gu: gu.new_empty(gu.shape[0], 129), "must be even"),
        (lambda gu: gu[:, :0], "W must be in"),
        (lambda gu: torch.empty(gu.shape[0], 8208, device=DEV, dtype=gu.dtype),
         "W must be in"),
    ],
)
def test_gate_up_rejections(mutate, match):
    gate_up, _ = _make_inputs(4, 512, torch.bfloat16, seed=1)
    with pytest.raises(RuntimeError, match=match):
        torch.ops._C_ascend.add_lora_swiglu_quant(mutate(gate_up), None, 0.0)


@pytest.mark.parametrize(
    "delta_fn, match",
    [
        (lambda gu: torch.zeros_like(gu[:, : gu.shape[1] // 2]), "delta shape"),
        (lambda gu: torch.zeros_like(gu).float(), "delta dtype"),
        # right shape and dtype but strided rows: the binding checks shape and
        # dtype before contiguity, so this must be a same-shape non-contiguous view
        (lambda gu: torch.zeros(gu.shape[0], gu.shape[1] + 16, device=DEV,
                                dtype=gu.dtype)[:, : gu.shape[1]],
         "delta must be contiguous"),
    ],
)
def test_delta_rejections(delta_fn, match):
    gate_up, _ = _make_inputs(4, 512, torch.bfloat16, seed=1)
    with pytest.raises(RuntimeError, match=match):
        torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, delta_fn(gate_up), 0.0)


# ------------------------------------------------- out-of-contract widths ---
# Pre-guard behavior on bz-ascend (2026-10-14), classified by W % 64 tail:
#   tail 16 (80, 528, 1040):  NO fault, int8 y payload silently corrupted
#   tail 32 (96, 4000):       numerically OK by luck only -- still illegal
#   tail 8/40/48/56/63 (8, 32, 48, 63, 1000, 1008, 2000): UB-misaligned
#     vector fault ("The UB address accessed by the VEC instruction is not
#     aligned"), poisons the whole context; 2000 = 8000/4 is the realistic
#     TP-sharded intermediate width production can hit
# 8/32/48 also subsume the old sub-64 degenerate-reduce probe (repeatTime 0).
# All of these must raise at the binding instead of reaching the kernel.


@pytest.mark.parametrize(
    "width", [8, 32, 48, 63, 80, 96, 528, 1000, 1008, 1040, 2000, 4000])
def test_illegal_width_rejected(width):
    gate_up, delta = _make_inputs(8, width, torch.bfloat16, seed=width)
    with pytest.raises(RuntimeError, match="multiple of 64"):
        torch.ops._C_ascend.add_lora_swiglu_quant(gate_up, delta, 0.0)
