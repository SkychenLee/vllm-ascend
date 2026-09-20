# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)


def _mla_spec():
    return AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_get_storage_block_size_and_dcp_memory():
    spec = _mla_spec()
    # On main, storage_block_size is an optional dataclass field and may be
    # None. Ascend derives physical rows from block_size / compression ratio.
    expected = spec.block_size // get_kv_cache_compression_ratio(spec)
    assert get_storage_block_size(spec) == expected

    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    assert get_storage_block_size(uniform) == expected

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    assert spec.max_memory_usage_bytes(vllm_config) > 0


def test_sliding_window_mla_storage_and_page_size():
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    assert spec.storage_block_size == 16
    assert spec.real_page_size_bytes == 16 * 128 * 2


@pytest.mark.parametrize(
    (
        "max_model_len",
        "batch_tokens",
        "in_flight_tokens",
        "retained_tokens",
        "padded_bytes",
        "estimated_blocks",
        "admission_blocks",
    ),
    [
        (4096, 128, 128, 0, None, 13, 13),
        (4096, 128, 256, 0, None, 13, 21),
        (4096, 128, 256, 32, 8192, 13, 23),
        (65, 128, 256, 32, None, 6, 6),
        (4096, 130, 260, 0, None, 14, 22),
    ],
)
def test_sliding_window_mla_legacy_capacity_preserves_runtime_admission(
    max_model_len,
    batch_tokens,
    in_flight_tokens,
    retained_tokens,
    padded_bytes,
    estimated_blocks,
    admission_blocks,
):
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=64,
        extra_retained_tokens=retained_tokens,
        page_size_padded=padded_bytes,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=batch_tokens),
        max_in_flight_tokens=in_flight_tokens,
    )

    assert spec.max_memory_usage_bytes(vllm_config) == estimated_blocks * spec.page_size_bytes
    assert spec.max_admission_blocks_per_request(in_flight_tokens, max_model_len) == admission_blocks
    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    assert uniform.max_memory_usage_pages(vllm_config) == estimated_blocks
