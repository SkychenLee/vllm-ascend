# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_dflash_inputs_kernel_ascend(
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    temperature_ptr,
    seeds_ptr,
    block_table_ptr,
    block_table_stride,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank=0,
    SAMPLE_FROM_ANCHOR: tl.constexpr = False,
    PAD_SLOT_ID: tl.constexpr = -1,
    CP_SIZE: tl.constexpr = 1,
    CP_INTERLEAVE: tl.constexpr = 1,
    BLOCK_SIZE: tl.constexpr = 256,
):
    # Keep the upstream launch ABI (including pre-DCP v0.28 callers). Each
    # program owns disjoint tiles; no token-dependent host reads are needed.
    req_idx = tl.program_id(0)
    worker = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    num_valid_ctx = num_ctx - tl.load(num_rejected_ptr + req_idx)
    last_valid_pos = tl.load(target_positions_ptr + ctx_start + num_valid_ctx - 1)
    query_base = req_idx * num_query_per_req

    # Triton Ascend cannot lower clamped vector block-table gathers on all
    # supported compiler versions. Partition scalar gathers across workers;
    # contiguous padding below can still use vector stores.
    workers = tl.num_programs(1)
    ctx_per_worker = tl.cdiv(num_ctx, workers)
    for offsets in range(worker * ctx_per_worker, tl.minimum((worker + 1) * ctx_per_worker, num_ctx)):
        valid = offsets < num_valid_ctx
        pos = tl.load(target_positions_ptr + ctx_start + offsets, mask=valid, other=0)
        logical_block = tl.minimum(pos // (block_size * CP_SIZE), block_table_stride - 1)
        block_id = tl.load(block_table_ptr + req_idx * block_table_stride + logical_block, mask=valid, other=0).to(
            tl.int64
        )
        if CP_SIZE == 1:
            slot = block_id * block_size + pos % block_size
        else:
            virtual_offset = pos % (block_size * CP_SIZE)
            local_offset = virtual_offset // (CP_SIZE * CP_INTERLEAVE) * CP_INTERLEAVE
            local_offset += virtual_offset % CP_INTERLEAVE
            valid = valid & ((virtual_offset // CP_INTERLEAVE) % CP_SIZE == cp_rank)
            slot = block_id * block_size + local_offset
        slot = tl.where(valid & (block_id != 0), slot, PAD_SLOT_ID)
        tl.store(
            out_context_positions_ptr + ctx_start + offsets,
            pos,
        )
        tl.store(
            out_context_slot_mapping_ptr + ctx_start + offsets,
            slot,
        )

    query_per_worker = tl.cdiv(num_query_per_req, workers)
    for offsets in range(worker * query_per_worker, tl.minimum((worker + 1) * query_per_worker, num_query_per_req)):
        valid = offsets < num_query_per_req
        pos = last_valid_pos + 1 + offsets
        logical_block = tl.minimum(pos // (block_size * CP_SIZE), block_table_stride - 1)
        block_id = tl.load(block_table_ptr + req_idx * block_table_stride + logical_block, mask=valid, other=0).to(
            tl.int64
        )
        if CP_SIZE == 1:
            slot = block_id * block_size + pos % block_size
            local = valid
        else:
            virtual_offset = pos % (block_size * CP_SIZE)
            local_offset = virtual_offset // (CP_SIZE * CP_INTERLEAVE) * CP_INTERLEAVE
            local_offset += virtual_offset % CP_INTERLEAVE
            local = valid & ((virtual_offset // CP_INTERLEAVE) % CP_SIZE == cp_rank)
            slot = block_id * block_size + local_offset
        slot = tl.where(local & (block_id != 0), slot, PAD_SLOT_ID)
        if tl.load(num_sampled_ptr + req_idx) > 0:
            bonus = tl.load(last_sampled_ptr + req_state_idx)
        else:
            bonus = tl.load(next_prefill_tokens_ptr + req_state_idx)
        token = tl.where(offsets == 0, bonus, parallel_drafting_token_id)
        tl.store(out_input_ids_ptr + query_base + offsets, token, mask=valid)
        tl.store(out_query_positions_ptr + query_base + offsets, tl.minimum(pos, max_model_len - 1), mask=valid)
        tl.store(out_query_slot_mapping_ptr + query_base + offsets, slot, mask=valid)
        sample_off: tl.constexpr = 0 if SAMPLE_FROM_ANCHOR else 1
        sample_idx = req_idx * num_speculative_steps + offsets - sample_off
        sample_mask = valid & (offsets >= sample_off)
        tl.store(out_sample_indices_ptr + sample_idx, query_base + offsets, mask=sample_mask)
        sample_pos = pos + 1 if SAMPLE_FROM_ANCHOR else pos
        tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=sample_mask)
        tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=sample_mask)

    if worker == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(out_seq_lens_ptr + req_idx, tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len))
        tl.store(out_temperature_ptr + req_state_idx, tl.load(temperature_ptr + req_state_idx))
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))

    # Split graph padding across every active program, independently of the
    # small upstream decode tile (often just 8 lanes).
    PAD_BLOCK_SIZE: tl.constexpr = 256
    lanes = tl.arange(0, PAD_BLOCK_SIZE)
    pad_worker = req_idx * workers + worker
    pad_stride = num_reqs * workers * PAD_BLOCK_SIZE
    for tile in range(num_reqs + pad_worker * PAD_BLOCK_SIZE, max_num_reqs + 1, pad_stride):
        pad_offsets = tile + lanes
        tl.store(out_query_start_loc_ptr + pad_offsets, num_reqs * num_query_per_req, mask=pad_offsets <= max_num_reqs)
        tl.store(out_seq_lens_ptr + pad_offsets, 0, mask=pad_offsets < max_num_reqs)
    pad_end = max_num_reqs * num_speculative_steps
    for tile in range(num_reqs * num_speculative_steps + pad_worker * PAD_BLOCK_SIZE, pad_end, pad_stride):
        pad_offsets = tile + lanes
        tl.store(out_sample_indices_ptr + pad_offsets, 0, mask=pad_offsets < pad_end)
        tl.store(out_sample_pos_ptr + pad_offsets, 0, mask=pad_offsets < pad_end)
        tl.store(out_sample_idx_mapping_ptr + pad_offsets, -1, mask=pad_offsets < pad_end)
    for tile in range(num_reqs * num_query_per_req + pad_worker * PAD_BLOCK_SIZE, max_num_tokens, pad_stride):
        pad_offsets = tile + lanes
        tl.store(out_query_slot_mapping_ptr + pad_offsets, PAD_SLOT_ID, mask=pad_offsets < max_num_tokens)
