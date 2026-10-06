# SPDX-License-Identifier: Apache-2.0
"""Exercise same-arena masking across waves, layers and adapter reloads."""

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.lora import cube_moe as cube


@pytest.fixture
def cpu_cube(monkeypatch):
    # A small wave count exercises the real wave loop with few dot products.
    # Keep the production allocation and H/rank/row contracts unchanged.
    monkeypatch.setattr(cube, "WAVE_TILES", 2)
    monkeypatch.setattr(cube, "_SCRATCH", {})
    return cube


def sparse_weights(multiplier):
    weights = tuple(torch.zeros(2, 32, 16, 4096, dtype=torch.bfloat16) for _ in range(2))
    rank = torch.arange(16)
    for plane, weight in enumerate(weights):
        for slot in range(2):
            for expert in range(32):
                weight[slot, expert, rank, rank + plane * 16] = multiplier * (1 + slot) * (1 + expert % 4) / 8
    return weights


def batch(counts, slot=0, seed=7):
    counts = torch.tensor(counts, dtype=torch.int64)
    total = int(counts.sum())
    # Padded input is deliberately NaN: zero multiplication cannot replace
    # masking, and a later expert/tile must not retain the previous batch.
    x = torch.full((6144, 4096), float("nan"), dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(seed)
    x[:total] = (torch.randint(-16, 17, (total, 4096), generator=generator) / 8).to(torch.bfloat16)
    combined = torch.full((x.shape[0],), -(2**63), dtype=torch.int64)
    combined[:total] = torch.repeat_interleave(torch.arange(32), counts) + slot * 32
    return x, counts, combined


def reference(x, counts, weights, slot):
    outputs = [torch.zeros(x.shape[0], 16) for _ in range(2)]
    start = 0
    rank = torch.arange(16)
    for expert, count in enumerate(counts.tolist()):
        for plane, weight in enumerate(weights):
            coefficient = weight[slot, expert, rank, rank + plane * 16].float()
            outputs[plane][start : start + count] = (
                x[start : start + count, plane * 16 : (plane + 1) * 16].float() * coefficient
            )
        start += count
    return outputs


def run_and_check(state, x, counts, combined, slot=0):
    routing = state.prepare(x, counts, 1, slot)
    assert routing is not None and routing.tile_count > 2
    rows = x.shape[0]
    for output in state.scratch.outputs:
        output[rows + 1 : rows + 5].fill_(123)
    products = []

    def bmm(a, b):
        assert a.untyped_storage().data_ptr() == state.scratch.gathered.untyped_storage().data_ptr()
        valid = state.scratch.valid[: a.shape[0]]
        assert torch.count_nonzero(a[~valid]) == 0
        assert torch.isfinite(a).all()
        products.append(a.shape[0])
        return torch.bmm(a.float(), b.float())

    legacy = Mock(side_effect=AssertionError("accepted Cube should not call BGMV"))
    actual = state.shrink_pair(x, combined, state.weights, routing, legacy, bmm=bmm)
    expected = reference(x, counts, state.weights, slot)
    for a, e, backing in zip(actual, expected, state.scratch.outputs):
        assert torch.equal(a, e)
        assert torch.equal(backing[rows + 1 : rows + 5], torch.full((4, 16), 123.0))
        assert torch.count_nonzero(a[int(counts.sum()) :]) == 0
    assert len(products) == (routing.tile_count + 1) // 2
    legacy.assert_not_called()
    # The next layer/forward intentionally reuses these outputs; consumers
    # finish on the same stream before that happens.
    return tuple(a.clone() for a in actual)


def test_single_arena_keeps_fixed_scratch_below_183_mib(cpu_cube):
    scratch = cpu_cube.CubeScratch("cpu")
    assert scratch.gathered.data_ptr() == scratch.tiles.data_ptr()
    tensors = []
    for value in vars(scratch).values():
        tensors.extend(value if isinstance(value, tuple) else (value,))
    storage = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors if isinstance(t, torch.Tensor)
    }
    assert sum(storage.values()) <= 183 * 1024**2


def test_same_arena_mask_handles_waves_and_changed_counts(cpu_cube):
    state = cpu_cube.CubeW13State(sparse_weights(1))
    state.mark_loaded(0)
    first = batch([65, 31, 0, 2, 64, 127, 223] + [0] * 25)
    run_and_check(state, *first)
    second = batch([0, 99, 129, 0, 0, 0, 0, 285] + [0] * 24, seed=13)
    run_and_check(state, *second)
    assert state.pack_count == 1


def test_layers_share_arena_without_sharing_packed_weights(cpu_cube):
    weights = sparse_weights(1)
    other_weights = tuple(w.mul(-2) for w in weights)
    states = [cpu_cube.CubeW13State(weights), cpu_cube.CubeW13State(other_weights)]
    assert states[0].scratch is states[1].scratch
    assert states[0].packed.data_ptr() != states[1].packed.data_ptr()
    data = batch([65, 31, 0, 2, 64, 127, 223] + [0] * 25)
    for state in states:
        state.mark_loaded(0)
    first = run_and_check(states[0], *data)
    second = run_and_check(states[1], *data)
    assert all(torch.equal(b, a * -2) for a, b in zip(first, second))
    run_and_check(states[0], *data)
    assert all(state.pack_count == 1 for state in states)


def test_reload_rebuilds_packed_cache_and_changed_slot_clears_tails(cpu_cube):
    state = cpu_cube.CubeW13State(sparse_weights(1))
    state.mark_loaded(0)
    counts = [65, 31, 0, 2, 64, 127, 223] + [0] * 25
    data = batch(counts)
    first = run_and_check(state, *data)
    state.invalidate(0)
    assert state.prepare(data[0], data[1], 1, 0) is None
    for weight in state.weights:
        weight[0].mul_(2)
    state.mark_loaded(0)
    second = run_and_check(state, *data)
    assert all(torch.equal(b, a * 2) for a, b in zip(first, second))
    assert state.pack_count == 2
    state.mark_loaded(1)
    changed = batch([0, 99, 129, 0, 0, 0, 0, 285] + [0] * 24, slot=1, seed=19)
    run_and_check(state, *changed, slot=1)
    assert state.pack_count == 3
