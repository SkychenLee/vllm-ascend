# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""A2/A3 FA3 draft lengths and graph replay against an FP32 reference."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.pull_request.one_card.test_flash_attn_precision import golden
from vllm_ascend.attention.flash_attention_v3 import (
    AscendFlashAttentionImpl,
    AscendFlashAttentionMetadataBuilder,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 32])
@pytest.mark.parametrize("head_size", [128, 192])
@torch.inference_mode()
def test_dflash2_fa3_rejection_and_padded_graph_replay(dtype, causal, window, head_size):
    pytest.importorskip("flash_attn_npu_3")
    if not get_current_hardware_profile().supports(HardwareCapability.FLASH_ATTN_NPU_3):
        pytest.skip("Device-length ecosystem FA3 is enabled on A2/A3")
    torch.npu.set_device(0)
    torch.manual_seed(42)
    device = "npu"
    block_size, num_reqs, query_len, num_heads = 128, 3, 6, 4
    blocks_per_req = 4
    num_blocks = num_reqs * blocks_per_req
    # Match the runner's interleaved K/V pages, including page padding.
    page_stride = 2 * block_size * head_size + 128
    backing = torch.empty((num_blocks, page_stride), dtype=dtype, device=device)
    key = backing.as_strided((num_blocks, block_size, 1, head_size), (page_stride, head_size, head_size, 1))
    value = backing.as_strided(key.shape, key.stride(), storage_offset=block_size * head_size)
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    query = torch.randn(num_reqs * query_len, num_heads, head_size, dtype=dtype, device=device)
    tables = torch.randperm(num_blocks, device=device).to(torch.int32).reshape(num_reqs, blocks_per_req)
    original_tables = tables.clone()
    lengths = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    offsets_cpu = torch.arange(num_reqs + 1, dtype=torch.int32) * query_len
    offsets = offsets_cpu.to(device)
    common = SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=query.shape[0],
        num_input_tokens=query.shape[0],
        query_start_loc_cpu=offsets_cpu,
        query_start_loc=offsets,
        seq_lens=lengths,
        block_table_tensor=tables,
        slot_mapping=torch.full((query.shape[0],), -1, dtype=torch.int32, device=device),
        max_query_len=query_len,
        causal=causal,
    )
    impl = AscendFlashAttentionImpl(num_heads, head_size, head_size**-0.5, num_kv_heads=1, sliding_window=window)
    builder = AscendFlashAttentionMetadataBuilder.__new__(AscendFlashAttentionMetadataBuilder)
    builder.block_size = block_size
    builder.scheduler_specs = {(num_heads, 1, head_size, dtype, impl.scale, 0.0, window)}
    builder.scheduler_buffers = {}
    builder.use_graph_buffers = True
    layer = SimpleNamespace(layer_name="draft.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    cache = (key, value)
    output = torch.empty_like(query)
    eager_output = torch.empty_like(query)

    lengths.copy_(torch.tensor([100, 200, 300], dtype=torch.int32, device=device))
    capture_metadata = builder.build(0, common)
    impl.forward(layer, query, None, None, cache, capture_metadata, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        impl.forward(layer, query, None, None, cache, capture_metadata, output)

    query_cpu, key_cpu, value_cpu = query.cpu(), key.cpu(), value.cpu()
    # Change rejection counts, request order and active size in one captured bucket.
    for exact_lengths, order in [([198, 99], [1, 0]), ([295], [2]), ([101, 202, 299], [0, 1, 2])]:
        active = len(exact_lengths)
        lengths.zero_()
        lengths[:active].copy_(torch.tensor(exact_lengths, dtype=torch.int32, device=device))
        offsets_cpu.copy_(torch.arange(num_reqs + 1, dtype=torch.int32).clamp(max=active) * query_len)
        offsets.copy_(offsets_cpu)
        tables[:active].copy_(original_tables[order])
        metadata = builder.build(0, common)
        for spec, buffer in metadata.scheduler_metadata.items():
            assert buffer.data_ptr() == capture_metadata.scheduler_metadata[spec].data_ptr()
        impl.forward(layer, query, None, None, cache, metadata, eager_output)
        graph.replay()
        torch.npu.synchronize()
        expected = golden(query_cpu, key_cpu, value_cpu, tables.cpu(), exact_lengths, window, query_len, causal)
        valid_tokens = active * query_len
        torch.testing.assert_close(eager_output[:valid_tokens].cpu(), expected, rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(output[:valid_tokens].cpu(), expected, rtol=2e-2, atol=2e-3)
