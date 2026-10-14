# SPDX-License-Identifier: Apache-2.0
"""Extensive numeric tests for the fused add_lora kernel (z1 + z2).

Kernel under test: csrc/kernels/add_lora_fused.cpp, reached through
PunicaWrapperNPU.add_lora -> torch.ops._C_ascend.add_lora with the decode
(use_gmm=False) dispatch, fused kernel enabled.

Semantics tested (add == accumulate, overwrite == add_inputs=False):
    y[t, col_s : col_s + h2_s] (+)= ((x[t] @ A_s[slot]^T * scale) @ B_s[slot]^T)
for each slice s laid out consecutively from offset_start=0; slot==-1 tokens
leave y untouched under accumulate and get their slice range zeroed under
overwrite (columns beyond the concatenated slices are NEVER touched).

Kernel-declared constraints (violations reaching the op are ELIGIBILITY
BUGS; correct results with them come from the bgmv fallback):
  - R in {16, 32, 64}; H1, every h2 and yWidth multiples of 16; <= 4 slices
  - offset_start == 0; indices int64 with -1 = no-adapter; fp16/bf16 in/out,
    fp32 accumulate inside, deterministic (two launches bitwise equal)
Grid-boundary constants mirrored from the kernel (keep in sync when editing
either side; a mismatch here only weakens sweep targeting, never correctness):
"""

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

_HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
pytestmark = pytest.mark.skipif(not _HAS_NPU, reason="fused add_lora requires an NPU")

DEV = "npu:0"

# mirrors TILE_H in add_lora_fused.cpp (H1 <= TILE_H takes the single-copy
# A-load branch, H1 > TILE_H the per-row strided branch)
TILE_H_REF = 4096
# mirrors W_IN_TILE: z2's per-output-tile quantum is W_IN_TILE / R
W_IN_TILE_REF = 8192
# fp32 reference vs bf16/fp16 kernel output; magnitude of the test tensors
# matches the existing MoE fused tests (x*0.3, weights*0.05), which use 3e-2
REF_TOL = 3e-2

# batch points around the grid-plan transitions of PlanGrid / add_lora (rank
# groups split the rank axis at small B to fill the machine; unitsPerCore > 1
# once B * groups exceeds the AIV count; a run of same-slot tokens may also
# straddle one block's unit range -- all-same / long-run index patterns at
# these sizes hit that, since the run length cap only applies within a
# block's contiguous unit range)
BATCH_SWEEP = [1, 2, 7, 8, 16, 32, 33, 40, 41, 48, 64, 96, 128]


class _BareWrapper(PunicaWrapperNPU):
    """Test-only subclass. prefill_metadata is a read-only property on the
    upstream base class, so we shadow it with a class attribute instead of
    assigning. add_lora unpacks the 6-tuple but [1] (seq_len) / [2]
    (lora_indices) reach the C++ only through the gmm branch, which the
    use_gmm=False decode dispatch here never takes; the fused kernel and the
    bgmv fallback consume token_lora_indices exclusively."""

    # [1]/[2] must be real tensors anyway: the op's schema declares them
    # at::Tensor and the binding converts arguments regardless of the branch
    _empty_idx = torch.empty(0, dtype=torch.long)
    prefill_metadata = (None, _empty_idx, _empty_idx, None, None, None)
    del _empty_idx

    # token_lora_indices is a read-only property upstream as well, but unlike
    # prefill_metadata it needs a per-test value -- shadow it with a writable
    # property over a uniquely named attribute (no upstream name to collide)
    @property
    def token_lora_indices(self):
        return self._test_token_lora_indices

    @token_lora_indices.setter
    def token_lora_indices(self, value):
        self._test_token_lora_indices = value


def _make_wrapper() -> PunicaWrapperNPU:
    """Bare wrapper: decode dispatch, fused kernel enabled, flags only."""
    wrapper = object.__new__(_BareWrapper)
    wrapper._use_gmm_shrink_cpu = torch.tensor(False, dtype=torch.bool)
    wrapper._no_lora_cpu = torch.tensor(False, dtype=torch.bool)
    wrapper._use_add_lora_cpu = torch.tensor(True, dtype=torch.bool)
    return wrapper


def _make_case(
    batch: int,
    h1: int,
    rank: int,
    h2s: tuple,
    slots: list,
    dtype: torch.dtype,
    seed: int,
    y_width_extra: int = 0,
    num_slots: int = 8,
):
    """x, y0, per-slice (a, b) stacks and the slot mapping for one run.

    Weight layouts are the linear (non-MoE) ones the op expects:
    lora_a [num_slots, 1, R, H1], lora_b [num_slots, 1, h2, R].
    """
    torch.manual_seed(seed)
    x = (torch.randn(batch, h1, device=DEV) * 0.3).to(dtype)
    a = [(torch.randn(num_slots, 1, rank, h1, device=DEV) * 0.05).to(dtype) for _ in h2s]
    b = [(torch.randn(num_slots, 1, h2, rank, device=DEV) * 0.05).to(dtype) for h2 in h2s]
    total_h2 = sum(h2s)
    y0 = (torch.randn(batch, total_h2 + y_width_extra, device=DEV) * 0.2).to(dtype)
    mapping = torch.tensor(slots, device=DEV, dtype=torch.long)
    return x, y0, a, b, mapping


def _reference(y0, x, a, b, mapping, h2s, scale, add_inputs, offset=0):
    """fp32 oracle; scale folded into z (after the A reduction, before B),
    matching z1's stage-and-scale ordering. Returns the fp32 expectation
    including the -1-token semantics and the untouched padding columns."""
    ref = y0.float().clone()
    for t in range(x.shape[0]):
        slot = int(mapping[t])
        if slot < 0:
            if not add_inputs:
                ref[t, offset : offset + sum(h2s)] = 0.0
            continue
        col = offset
        for s, h2 in enumerate(h2s):
            z = x[t].float() @ a[s][slot, 0].float().T * scale
            delta = z @ b[s][slot, 0].float().T
            if add_inputs:
                ref[t, col : col + h2] += delta
            else:
                ref[t, col : col + h2] = delta
            col += h2
    return ref


def _apply(y, x, a, b, mapping, h2s, *, scale=1.0, add_inputs=True, offset_start=0):
    wrapper = _make_wrapper()
    wrapper.token_lora_indices = mapping
    wrapper.add_lora(
        y, x, tuple(a), tuple(b), scale, tuple(h2s),
        offset_start=offset_start, add_inputs=add_inputs,
    )
    return y


def _check(x, y0, a, b, mapping, h2s, *, scale=1.0, add_inputs=True,
           offset=0, y_width_extra=0, tag=""):
    y = y0.clone()
    _apply(y, x, a, b, mapping, h2s, scale=scale, add_inputs=add_inputs,
           offset_start=offset)
    ref = _reference(y0, x, a, b, mapping, h2s, scale, add_inputs, offset)
    err = (y.float() - ref).abs().max().item()
    print(f"{tag}: err={err:.3e}")
    assert err < REF_TOL, f"{tag}: max_err={err}"
    if y_width_extra:
        # padding beyond the concatenated slices must be bit-untouched even
        # in overwrite mode (z2 zeroes only the slice range)
        pad = sum(h2s) + offset
        assert torch.equal(y[:, pad:], y0[:, pad:]), f"{tag}: padding columns modified"
    return y


# ---------------------------------------------------------------- shape ---

# TILE_H branch boundary + the R in {16,32,64} domain, on one batch/mapping
@pytest.mark.parametrize("rank", [16, 32, 64])
@pytest.mark.parametrize("h1", [4096, TILE_H_REF + 16, 6144, 7168, 8192, 12288])
def test_h1_rank_sweep(h1, rank):
    # fixed mixed mapping: same-slot runs, -1 interruptions, all slices live
    slots = [0, 0, 1, 1, -1, 2, 2, 3, 3, 3, -1, 0, 1, 2, 2, 3]
    x, y0, a, b, mapping = _make_case(16, h1, rank, (512, 256), slots,
                                      torch.bfloat16, seed=h1 + rank)
    _check(x, y0, a, b, mapping, (512, 256), tag=f"h1={h1} R={rank}")


# h2 sizes around the z2 quantum outPerTile = W_IN_TILE_REF / R: full tiles,
# remainder tiles (len % outPerTile), sub-quantum slices (pure remainder ->
# ComputeTile's custom repeat-count path) and unequal slices so chunks
# straddle slice boundaries
@pytest.mark.parametrize(
    "rank,h2s",
    [
        (16, (W_IN_TILE_REF // 16,)),                          # exactly 1 tile
        (16, (2 * 512, 3 * 512)),                              # exact multiples
        (16, (512 + 16,)),                                     # tile + tiny tail
        (16, (64, 96)),                                        # sub-quantum pair
        (32, (256, 256 + 16, 3 * 256 - 16, 32)),               # 4 uneven slices
        (64, (128, 128 + 16)),                                 # mixed full/tail
        (64, (128 * 3, 128)),                                  # tail first slice
        (16, (2048, 2048, 2048, 2048)),                        # chunked straddle
    ],
)
def test_h2_quantum_paths(rank, h2s):
    slots = [1, 1, 2, -1, 3, 3, 0, 0]
    x, y0, a, b, mapping = _make_case(8, 4096, rank, h2s, slots,
                                      torch.bfloat16, seed=sum(h2s) + rank)
    _check(x, y0, a, b, mapping, h2s, tag=f"R={rank} h2={h2s}")


@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("y_width_extra", [0, 16, 4096])
def test_y_width_padding(add_inputs, y_width_extra):
    # nonzero extras must stay untouched; zero extras are the exact-fit case
    slots = [2, 2, -1, 1, 1, 1, -1, 2]
    h2s = (512, 256)
    x, y0, a, b, mapping = _make_case(8, 2048, 16, h2s, slots, torch.bfloat16,
                                      seed=y_width_extra + int(add_inputs),
                                      y_width_extra=y_width_extra)
    _check(x, y0, a, b, mapping, h2s, add_inputs=add_inputs,
           y_width_extra=y_width_extra, tag=f"pad+{y_width_extra} add={add_inputs}")


# ---------------------------------------------------------------- batch ---

# unitsPerCore transitions and rank-group splits depend on the device AIV
# count, which is unknown host-side; the sweep straddles the plausible 40/48
# core counts so both the one-token-per-block and multi-unit-per-block
# mappings are hit on any part
@pytest.mark.parametrize("batch", BATCH_SWEEP)
def test_batch_sweep(batch):
    h2s = (2048, 1024)
    slots = [-1 if t % 7 == 6 else (t // 2) % 4 for t in range(batch)]
    x, y0, a, b, mapping = _make_case(batch, 4096, 16, h2s, slots,
                                      torch.bfloat16, seed=batch)
    _check(x, y0, a, b, mapping, h2s, tag=f"B={batch}")


# --------------------------------------------------------------- indices ---

INDEX_PATTERNS = {
    "all_same": lambda b: [3] * b,
    "pairs": lambda b: [(t // 2) % 4 for t in range(b)],
    "runs2": lambda b: [(t // 2) % 2 for t in range(b)],
    "runs3": lambda b: [(t // 3) % 2 for t in range(b)],
    "runs4": lambda b: [(t // 4) % 2 for t in range(b)],
    "one_long_run": lambda b: [1] * 10 + [2] * (b - 10),
    "round_robin": lambda b: [t % 8 for t in range(b)],  # adjacent never equal
    "minus_breaks_runs": lambda b: [2, 2, -1, 2, 2, 2, -1, -1,
                                    3, 3, 3, 3, -1, 2, 2, 2][:b],
    "all_minus": lambda b: [-1] * b,
    "high_unsorted": lambda b: [7, 7, 5, 5, 5, 6, 6, -1,
                                4, 4, 4, 4, 2, 2, 1, 0][:b],
    # same slot on both sides of a -1: the run must not jump across it
    "run_around_minus": lambda b: [0] * 8 + [-1] + [0] * 7,
}


@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("pattern", sorted(INDEX_PATTERNS))
def test_index_patterns(pattern, add_inputs):
    batch = 16
    h2s = (512,)
    slots = INDEX_PATTERNS[pattern](batch)
    # stable seed (Python's hash() is salted per process)
    x, y0, a, b, mapping = _make_case(batch, 1024, 32, h2s, slots,
                                      torch.bfloat16, seed=sum(map(ord, pattern)) % 1000)
    y = _check(x, y0, a, b, mapping, h2s, add_inputs=add_inputs,
               tag=f"{pattern} add={add_inputs}")
    # explicit no-adapter row semantics (beyond the shared oracle tolerance)
    inactive = mapping < 0
    if inactive.any():
        if add_inputs:
            assert torch.equal(y[inactive], y0[inactive]), f"{pattern}: -1 rows changed"
        else:
            # overwrite zeroes exactly the slice range of -1 rows
            assert torch.equal(y[inactive, : sum(h2s)],
                               torch.zeros_like(y[inactive, : sum(h2s)])), \
                f"{pattern}: -1 rows not zeroed in overwrite"


# ------------------------------------------------------- dtype and scale ---

@pytest.mark.parametrize("scale", [1.0, 0.25, 0.7313])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dtypes_and_scales(dtype, scale):
    slots = [0, 0, 1, -1, 2, 2, 2, 1, 1, 0, 3, 3, -1, 3, 2, 0]
    h2s = (512,)
    x, y0, a, b, mapping = _make_case(16, 4096, 32, h2s, slots, dtype, seed=77)
    _check(x, y0, a, b, mapping, h2s, scale=scale, tag=f"{dtype} scale={scale}")


# ---------------------------------------------- token-batching regressions ---

# z1's Z1_TOK_SUB batching must not change any token's arithmetic: one
# batched launch has to be BITWISE equal to per-token launches (the run path
# only reuses the loaded A tile, it never reorders the per-token ops).
@pytest.mark.parametrize("rank,h1,h2", [(16, 2048, 256), (64, 1024, 128)])
def test_same_slot_batching_bitwise(rank, h1, h2):
    batch = 8
    h2s = (h2,)
    x, y0, a, b, mapping = _make_case(batch, h1, rank, h2s, [1] * batch,
                                      torch.bfloat16, seed=rank)
    # overwrite mode keeps single-token launches independent of y history
    y_batch = _apply(y0.clone(), x, a, b, mapping, h2s, add_inputs=False)
    singles = [
        _apply(y0[t : t + 1].clone(), x[t : t + 1], a, b, mapping[t : t + 1], h2s,
               add_inputs=False)
        for t in range(batch)
    ]
    y_singles = torch.cat(singles, dim=0)
    assert torch.equal(y_batch, y_singles), (
        f"batched-vs-per-token bitwise mismatch, R={rank} h1={h1}"
    )


def test_determinism_two_launches():
    for h1, rank, h2s, seed in ((4096, 16, (512, 256), 3), (8192, 64, (256,), 4)):
        slots = [0, 0, 1, -1, 2, 2, 3, 3, 3, -1, 1, 1, 2, -1, 0, 0]
        x, y0, a, b, mapping = _make_case(16, h1, rank, h2s, slots,
                                          torch.bfloat16, seed=seed)
        y1 = _apply(y0.clone(), x, a, b, mapping, h2s)
        y2 = _apply(y0.clone(), x, a, b, mapping, h2s)
        assert torch.equal(y1, y2), f"non-deterministic output, R={rank} h1={h1}"


# ------------------------------------------------- fallback-path numerics ---

# configs add_lora_eligible must reject where the bgmv fallback is a sound
# substitute; the TEST ONLY asserts correct numerics, never which path ran
FALLBACK_CASES = {
    # rank 8: fused rejects it ("rank is not a multiple of 16"), bgmv
    # supports small ranks (cf. test_bgmv_small_rank.py)
    "rank8": dict(h1=1024, rank=8, h2s=(64,), offset=0),
    # more than 4 slices (only the fused kernel supports 4)
    "five_slices": dict(h1=1024, rank=16, h2s=(64, 64, 64, 64, 64), offset=0),
    # non-zero offset_start: fused kernel addresses y from column 0
    "offset_start": dict(h1=1024, rank=16, h2s=(256,), offset=32, y_width_extra=16),
}

# measured on hardware: the fused kernel correctly REFUSES these shapes
# (torch_binding add_lora_eligible: "x width ... multiple of 16" / "output
# slice ... multiple of 16"), but the bgmv FALLBACK is numerically broken on
# them as well -- odd x width yields NaN (misaligned A-row streaming in
# bgmv_shrink), an odd output slice yields a wrong tail (h2=100: 4-element
# ragged write in bgmv_expand). Pre-existing bgmv limitation, not a
# fused-kernel regression; production linear dims are always 16-aligned.
# STRICT xfail: if bgmv ever handles these, XPASS fails the suite and the
# case must be promoted into FALLBACK_CASES.
LIMITATION_CASES = {
    "odd_h1": dict(h1=4104, rank=16, h2s=(512,), offset=0),
    "odd_slice": dict(h1=2048, rank=16, h2s=(100,), offset=0, y_width_extra=28),
}


def _run_fallback_case(cfg):
    case = cfg["name"]
    slots = [0, 0, 1, -1, 2, 2, -1, 3]
    h2s = cfg["h2s"]
    offset = cfg["offset"]
    y_width_extra = cfg.get("y_width_extra", 0)
    batch, h1, rank = 8, cfg["h1"], cfg["rank"]
    torch.manual_seed(sum(map(ord, case)) % 997)  # hash() is salted per process
    x = (torch.randn(batch, h1, device=DEV) * 0.3).bfloat16()
    a = [(torch.randn(8, 1, rank, h1, device=DEV) * 0.05).bfloat16() for _ in h2s]
    b = [(torch.randn(8, 1, h2, rank, device=DEV) * 0.05).bfloat16() for h2 in h2s]
    y0 = (torch.randn(batch, offset + sum(h2s) + y_width_extra,
                      device=DEV) * 0.2).bfloat16()
    mapping = torch.tensor(slots, device=DEV, dtype=torch.long)

    y = _apply(y0.clone(), x, a, b, mapping, h2s, offset_start=offset)
    ref = _reference(y0, x, a, b, mapping, h2s, 1.0, True, offset=offset)
    err = (y.float() - ref).abs().max().item()
    print(f"{case}: err={err:.3e}")
    assert err < REF_TOL, f"{case}: max_err={err}"
    untouched = slice(None, offset) if offset else slice(sum(h2s) + offset, None)
    assert torch.equal(y[:, untouched], y0[:, untouched]), f"{case}: out-of-slice region modified"


@pytest.mark.parametrize("case", sorted(FALLBACK_CASES))
def test_ineligible_falls_back_correctly(case):
    cfg = dict(FALLBACK_CASES[case])
    cfg["name"] = case
    _run_fallback_case(cfg)


@pytest.mark.xfail(strict=True,
                   reason="bgmv fallback is numerically wrong for non-16-aligned "
                          "dims (NaN / ragged tail); the fused kernel's refusal is "
                          "the actual contract under test -- see LIMITATION_CASES")
@pytest.mark.parametrize("case", sorted(LIMITATION_CASES))
def test_ineligible_bgmv_limitation(case):
    cfg = dict(LIMITATION_CASES[case])
    cfg["name"] = case
    _run_fallback_case(cfg)
