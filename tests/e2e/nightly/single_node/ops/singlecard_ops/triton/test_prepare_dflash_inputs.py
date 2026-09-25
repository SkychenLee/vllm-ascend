import gc
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel_ascend,
)


def test_prepare_dflash_inputs_clamps_seq_len_to_max_model_len():
    init_device_properties_triton()
    device = "npu"
    max_model_len = 16
    num_query_per_req = 3
    num_speculative_steps = 2

    outputs = {
        "out_input_ids_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_query_positions_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_query_start_loc_ptr": torch.empty(2, dtype=torch.int32, device=device),
        "out_seq_lens_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_query_slot_mapping_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_context_positions_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_context_slot_mapping_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_sample_indices_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_sample_pos_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_sample_idx_mapping_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_temperature_ptr": torch.empty(1, dtype=torch.float32, device=device),
        "out_seeds_ptr": torch.empty(1, dtype=torch.int64, device=device),
    }
    inputs = {
        "target_positions_ptr": torch.tensor([max_model_len - 1], dtype=torch.int32, device=device),
        "target_query_start_loc_ptr": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "idx_mapping_ptr": torch.tensor([0], dtype=torch.int32, device=device),
        "last_sampled_ptr": torch.tensor([42], dtype=torch.int32, device=device),
        "next_prefill_tokens_ptr": torch.tensor([43], dtype=torch.int32, device=device),
        "num_sampled_ptr": torch.tensor([1], dtype=torch.int32, device=device),
        "num_rejected_ptr": torch.tensor([0], dtype=torch.int32, device=device),
        "temperature_ptr": torch.tensor([1.0], dtype=torch.float32, device=device),
        "seeds_ptr": torch.tensor([0], dtype=torch.int64, device=device),
        "block_table_ptr": torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device),
    }
    kwargs = {
        **outputs,
        **inputs,
        "block_table_stride": 3,
        "parallel_drafting_token_id": 151643,
        "block_size": 8,
        "num_query_per_req": num_query_per_req,
        "num_speculative_steps": num_speculative_steps,
        "max_num_reqs": 1,
        "max_num_tokens": num_query_per_req,
        "max_model_len": max_model_len,
        "SAMPLE_FROM_ANCHOR": False,
        "PAD_SLOT_ID": -1,
        "BLOCK_SIZE": 1,
    }
    if not vllm_version_is("0.28.0"):
        kwargs.update(cp_rank=0, CP_SIZE=1, CP_INTERLEAVE=1)

    _prepare_dflash_inputs_kernel_ascend[(1, 1)](**kwargs)

    torch.testing.assert_close(
        outputs["out_seq_lens_ptr"],
        torch.tensor([max_model_len], dtype=torch.int32, device=device),
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def make_prepare_case(lengths, anchor=True, cp_size=1, cp_rank=0, interleave=1, capacity=32):
    """Independent CPU oracle, including untouched buffer tails and null blocks."""
    n = len(lengths)
    capacity = max(n, capacity)
    k = 5
    q = k if anchor else k + 1
    max_tokens = max(sum(lengths), 16384, capacity * q)
    starts = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32)
    positions = torch.cat([torch.arange(31 + i * 17, 31 + i * 17 + size) for i, size in enumerate(lengths)])
    mapping = torch.arange(capacity - 1, capacity - n - 1, -1, dtype=torch.int32)
    sampled = torch.tensor([0 if size > k + 1 else 1 for size in lengths], dtype=torch.int32)
    rejected = torch.tensor(
        [min(i % 3, size - 1) if sampled[i] else 0 for i, size in enumerate(lengths)], dtype=torch.int32
    )
    block_size = 32
    max_model_len = int(positions.max()) + 4  # Also exercise query position clipping.
    table_width = triton.cdiv(max_model_len + q, block_size * cp_size)
    table = torch.arange(1, 1 + n * table_width, dtype=torch.int32).reshape(n, table_width)
    table[:, 1::7] = 0
    cpu = dict(
        target_positions_ptr=positions,
        target_query_start_loc_ptr=starts,
        idx_mapping_ptr=mapping,
        last_sampled_ptr=torch.arange(capacity, dtype=torch.int32) + 100,
        next_prefill_tokens_ptr=torch.arange(capacity, dtype=torch.int32) + 200,
        num_sampled_ptr=sampled,
        num_rejected_ptr=rejected,
        temperature_ptr=torch.arange(capacity, dtype=torch.float32) / 16,
        seeds_ptr=torch.arange(capacity, dtype=torch.int64) + 2**40,
        block_table_ptr=table,
    )
    shapes = dict(
        out_input_ids_ptr=max_tokens,
        out_query_positions_ptr=max_tokens,
        out_query_start_loc_ptr=capacity + 1,
        out_seq_lens_ptr=capacity,
        out_query_slot_mapping_ptr=max_tokens,
        out_context_positions_ptr=sum(lengths),
        out_context_slot_mapping_ptr=sum(lengths),
        out_sample_indices_ptr=capacity * k,
        out_sample_pos_ptr=capacity * k,
        out_sample_idx_mapping_ptr=capacity * k,
        out_temperature_ptr=capacity,
        out_seeds_ptr=capacity,
    )
    expected = {
        name: torch.full(
            (size,),
            -77,
            dtype=torch.float32 if "temperature" in name else torch.int64 if "seeds" in name else torch.int32,
        )
        for name, size in shapes.items()
    }
    outputs = {name: value.npu() for name, value in expected.items()}

    def slot(req, pos):
        block = int(table[req, min(pos // (block_size * cp_size), table_width - 1)])
        if block == 0 or pos // interleave % cp_size != cp_rank:
            return -1
        local_pos = pos // (cp_size * interleave) * interleave + pos % interleave
        return block * block_size + local_pos % block_size

    for req, size in enumerate(lengths):
        state = int(mapping[req])
        begin = int(starts[req])
        valid = size - int(rejected[req])
        for j in range(size):
            pos = int(positions[begin + j]) if j < valid else 0
            expected["out_context_positions_ptr"][begin + j] = pos
            expected["out_context_slot_mapping_ptr"][begin + j] = slot(req, pos) if j < valid else -1
        last_pos = int(positions[begin + valid - 1])
        for j in range(q):
            pos = last_pos + 1 + j
            bonus = cpu["last_sampled_ptr" if sampled[req] else "next_prefill_tokens_ptr"][state]
            expected["out_input_ids_ptr"][req * q + j] = bonus if j == 0 else 151643
            expected["out_query_positions_ptr"][req * q + j] = min(pos, max_model_len - 1)
            expected["out_query_slot_mapping_ptr"][req * q + j] = slot(req, pos)
            if j >= (0 if anchor else 1):
                idx = req * k + j - (0 if anchor else 1)
                expected["out_sample_indices_ptr"][idx] = req * q + j
                expected["out_sample_pos_ptr"][idx] = pos + int(anchor)
                expected["out_sample_idx_mapping_ptr"][idx] = state
        expected["out_query_start_loc_ptr"][req] = req * q
        expected["out_seq_lens_ptr"][req] = min(last_pos + 1 + q, max_model_len)
        expected["out_temperature_ptr"][state] = cpu["temperature_ptr"][state]
        expected["out_seeds_ptr"][state] = cpu["seeds_ptr"][state]
    expected["out_query_start_loc_ptr"][n:] = n * q
    expected["out_seq_lens_ptr"][n:] = 0
    expected["out_query_slot_mapping_ptr"][n * q :] = -1
    expected["out_sample_indices_ptr"][n * k :] = 0
    expected["out_sample_pos_ptr"][n * k :] = 0
    expected["out_sample_idx_mapping_ptr"][n * k :] = -1
    kwargs = dict(
        **outputs,
        **{name: value.npu() for name, value in cpu.items()},
        block_table_stride=table_width,
        parallel_drafting_token_id=151643,
        block_size=block_size,
        num_query_per_req=q,
        num_speculative_steps=k,
        max_num_reqs=capacity,
        max_num_tokens=max_tokens,
        max_model_len=max_model_len,
        SAMPLE_FROM_ANCHOR=anchor,
        PAD_SLOT_ID=-1,
        cp_rank=cp_rank,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=interleave,
        BLOCK_SIZE=min(256, triton.next_power_of_2(max(lengths) + q)),
    )
    return kwargs, expected


@pytest.mark.parametrize("lengths", [(1,), (6,) * 8, (8192, 8184), (8192,) * 8, (1, 255, 257, 1031)])
@pytest.mark.parametrize("anchor", [False, True])
def test_prepare_dflash_parallel_matches_reference(lengths, anchor):
    kwargs, expected = make_prepare_case(lengths, anchor)
    grid = (len(lengths), triton.cdiv(max(lengths) + kwargs["num_query_per_req"], kwargs["BLOCK_SIZE"]))
    _prepare_dflash_inputs_kernel_ascend[grid](**kwargs)
    for name, value in expected.items():
        torch.testing.assert_close(kwargs[name].cpu(), value, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("cp_size,cp_rank,interleave", [(2, 0, 1), (2, 1, 2), (4, 3, 4)])
def test_prepare_dflash_parallel_cp(cp_size, cp_rank, interleave):
    kwargs, expected = make_prepare_case((257, 6, 1031), True, cp_size, cp_rank, interleave)
    _prepare_dflash_inputs_kernel_ascend[(3, 5)](**kwargs)
    for name, value in expected.items():
        torch.testing.assert_close(kwargs[name].cpu(), value, rtol=0, atol=0, msg=name)


def test_prepare_dflash_upstream_wrapper():
    """Exercise the actual pre-DCP caller ABI used by this checkout."""
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator

    kwargs, expected = make_prepare_case((8192, 8184), True)
    outputs = {
        name.removeprefix("out_").removesuffix("_ptr"): value
        for name, value in kwargs.items()
        if name.startswith("out_")
    }
    input_buffers = SimpleNamespace(
        **{name: outputs[name] for name in ("input_ids", "query_positions", "query_start_loc", "seq_lens")}
    )
    input_buffers.positions = input_buffers.query_positions
    batch = SimpleNamespace(
        num_reqs=2,
        num_scheduled_tokens=np.array([8192, 8184]),
        positions=kwargs["target_positions_ptr"],
        query_start_loc=kwargs["target_query_start_loc_ptr"],
        idx_mapping=kwargs["idx_mapping_ptr"],
    )
    from unittest.mock import patch

    with patch.object(speculator, "_prepare_dflash_inputs_kernel", _prepare_dflash_inputs_kernel_ascend):
        speculator.prepare_dflash_inputs(
            input_buffers,
            outputs["query_slot_mapping"],
            outputs["context_positions"],
            outputs["context_slot_mapping"],
            outputs["sample_indices"],
            outputs["sample_pos"],
            outputs["sample_idx_mapping"],
            outputs["temperature"],
            outputs["seeds"],
            batch,
            kwargs["num_sampled_ptr"],
            kwargs["num_rejected_ptr"],
            kwargs["last_sampled_ptr"],
            kwargs["next_prefill_tokens_ptr"],
            kwargs["temperature_ptr"],
            kwargs["seeds_ptr"],
            kwargs["block_table_ptr"],
            kwargs["block_size"],
            kwargs["parallel_drafting_token_id"],
            kwargs["num_query_per_req"],
            kwargs["num_speculative_steps"],
            kwargs["max_num_reqs"],
            kwargs["max_num_tokens"],
            kwargs["max_model_len"],
            sample_from_anchor=True,
        )
    for name, value in expected.items():
        torch.testing.assert_close(kwargs[name].cpu(), value, rtol=0, atol=0, msg=name)


def test_prepare_dflash_graph_replay_changes_mapping():
    kwargs, _ = make_prepare_case((6,) * 8, True)
    _prepare_dflash_inputs_kernel_ascend[(8, 1)](**kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _prepare_dflash_inputs_kernel_ascend[(8, 1)](**kwargs)
    for shift in (1, 3):
        kwargs["idx_mapping_ptr"].sub_(shift)
        kwargs["target_positions_ptr"].add_(shift)
        eager = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()}
        _prepare_dflash_inputs_kernel_ascend[(8, 1)](**eager)
        graph.replay()
        torch.npu.synchronize()
        for name in kwargs:
            if name.startswith("out_"):
                torch.testing.assert_close(kwargs[name], eager[name], rtol=0, atol=0, msg=name)
