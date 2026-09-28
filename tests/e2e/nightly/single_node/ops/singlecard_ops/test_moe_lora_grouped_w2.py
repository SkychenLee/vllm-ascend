# SPDX-License-Identifier: Apache-2.0
"""Column-tiled grouped W2: native Meta checks and NPU precision gates."""

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    # The offline runner explicitly loads the diagnostic binding first. Normal
    # NPU ST loads the complete production extension, including legacy BGMV.
    if not hasattr(torch.ops._C_ascend, "bgmv_expand_grouped"):
        from vllm_ascend.utils import enable_custom_op

        assert enable_custom_op(), "Build custom operators before running NPU ST."


def inputs(rows, rank, width, *, offset=32, groups=21, dtype=torch.bfloat16, device="meta"):
    return (
        torch.empty(rows, rank, dtype=torch.float32, device=device),
        torch.empty(groups, width, rank, dtype=dtype, device=device),
        torch.empty(rows, dtype=torch.int64, device=device),
        torch.empty(rows, dtype=torch.int64, device=device),
        torch.empty(rows, width + 2 * offset, dtype=dtype, device=device),
        offset,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [8, 16, 32, 64])
@pytest.mark.parametrize("width", [16, 496, 512, 528, 4096, 8192])
@pytest.mark.parametrize("rows", [0, 9, 98304])
def test_grouped_w2_meta_accepts_bounded_column_tiles(dtype, rank, width, rows):
    args = inputs(rows, rank, width, dtype=dtype)
    assert torch.ops._C_ascend.bgmv_expand_grouped(*args) is None
    assert torch.ops._C_ascend.bgmv_expand_grouped_tiled(*args) is None


@pytest.mark.parametrize("rank,width,offset", [(7, 512, 0), (16, 0, 0), (16, 513, 0), (16, 8208, 0), (16, 512, 1)])
def test_grouped_w2_meta_rejects_unsupported_shape(rank, width, offset):
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.bgmv_expand_grouped(*inputs(9, rank, width, offset=offset))


def route(ids, groups):
    valid = (ids >= 0) & (ids < groups)
    order = torch.argsort(torch.where(valid, ids, groups).float())
    return torch.where(valid, ids, -1).index_select(0, order), order


def make_case(rows, rank, width, dtype, mode, offset=32, groups=768):
    torch.manual_seed(20260928 + rows + rank + width)
    x = torch.randn(rows, rank, device="npu") * 0.3
    w = torch.randn(groups, width, rank, dtype=dtype, device="npu") * 0.05
    # Interleaved slots exercise the full expert/adapter ID, not expert alone.
    ids = torch.arange(rows, device="npu", dtype=torch.int64) * 17 % groups
    if mode == "disabled":
        ids.fill_(-1)
    elif mode == "single":
        ids = groups - 256 + ids % 256
    if mode != "disabled" and rows > 1:
        ids[::11] = -1
    base = torch.randn(rows, width + 2 * offset, dtype=dtype, device="npu")
    return x, w, ids, base, offset


def assert_candidate(case, actual, expected):
    x, w, ids, base, offset = case
    width = w.shape[1]
    # Preserve the legacy tree reduction, casting, and add exactly.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if offset:
        torch.testing.assert_close(actual[:, :offset], base[:, :offset], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, offset + width :], base[:, offset + width :], atol=0, rtol=0)
    torch.testing.assert_close(actual[ids < 0], base[ids < 0], atol=0, rtol=0)
    # An independent high precision check covers every output column for rows
    # around task boundaries and the final partial unit.
    rows = x.shape[0]
    for row in sorted({r for r in (0, 1, 7, 8, 9, rows // 2, rows - 1) if 0 <= r < rows}):
        slot = int(ids[row].cpu())
        if slot < 0:
            continue
        ref = (base[row, offset : offset + width].cpu().double() + w[slot].cpu().double() @ x[row].cpu().double()).to(
            base.dtype
        )
        torch.testing.assert_close(actual[row, offset : offset + width].cpu(), ref, atol=2e-5, rtol=8e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rank,width",
    [(8, 1040), (16, 496), (16, 512), (16, 528), (16, 4096), (32, 272), (32, 4096), (64, 144), (64, 4096), (64, 8192)],
)
@pytest.mark.parametrize(
    "rows,mode,offset",
    [(0, "mixed", 32), (1, "single", 0), (7, "mixed", 32), (8, "mixed", 32), (9, "disabled", 32), (33, "mixed", 32)],
)
@torch.inference_mode()
def test_grouped_w2_npu_tiles_and_offsets(dtype, rank, width, rows, mode, offset):
    case = make_case(rows, rank, width, dtype, mode, offset)
    x, w, ids, base, offset = case
    sorted_ids, order = route(ids, w.shape[0])
    actual, expected = base.clone(), base.clone()
    torch.ops._C_ascend.bgmv_expand_grouped_tiled(x, w, sorted_ids, order, actual, offset)
    if rows:
        torch.ops._C_ascend.bgmv_expand(x, w, ids, expected, offset, width)
    assert_candidate(case, actual, expected)


@pytest.mark.parametrize("rows", [49152, 98304])
@torch.inference_mode()
def test_grouped_w2_npu_target_prefill(rows):
    case = make_case(rows, 16, 4096, torch.bfloat16, "mixed")
    x, w, ids, base, offset = case
    sorted_ids, order = route(ids, w.shape[0])
    actual, expected = base.clone(), base.clone()
    torch.ops._C_ascend.bgmv_expand_grouped_tiled(x, w, sorted_ids, order, actual, offset)
    torch.ops._C_ascend.bgmv_expand(x, w, ids, expected, offset, 4096)
    assert_candidate(case, actual, expected)


@torch.inference_mode()
def test_grouped_w2_npu_graph_replay_updates_inputs_routes_and_weights():
    case = make_case(33, 16, 4096, torch.bfloat16, "mixed")
    x, w, ids, base, offset = case
    actual = base.clone()

    def run():
        actual.copy_(base)
        sorted_ids, order = route(ids, w.shape[0])
        torch.ops._C_ascend.bgmv_expand_grouped_tiled(x, w, sorted_ids, order, actual, offset)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for mode in ("single", "disabled", "mixed"):
        updated = make_case(33, 16, 4096, torch.bfloat16, mode)
        x.copy_(updated[0] + 0.1)
        w.copy_(updated[1] * 0.7)
        ids.copy_(updated[2].roll(3))
        base.copy_(updated[3] * 0.9)
        graph.replay()
        torch.npu.synchronize()
        expected = base.clone()
        torch.ops._C_ascend.bgmv_expand(x, w, ids, expected, offset, 4096)
        assert_candidate(case, actual, expected)
