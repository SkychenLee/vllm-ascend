# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
from collections.abc import Sequence

import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
from vllm.v1.core.single_type_kv_cache_manager import (
    BlockHashList,
    BlockPool,
    KVCacheBlock,
    KVCacheSpec,
    MambaManager,
    MambaSpec,
)


class AscendMambaManager(MambaManager):
    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.block_size = kv_cache_spec.block_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        drop_eagle_block: bool = False,
    ) -> tuple[list[KVCacheBlock], ...] | tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(kv_cache_spec, MambaSpec), "AscendMambaManager can only be used for mamba groups"
        if drop_eagle_block and max_length > 0:
            # The last matched Mamba state may have been written over draft
            # tokens that MTP verification later rejects. Unlike attention
            # hits, Mamba hits are null-padded and contain only the rightmost
            # real state, so popping the result would remove the state itself.
            # Exclude that boundary from the lookup instead. Fine-grained
            # partial hits use one hash/alignment unit; regular hits use one
            # Mamba page.
            drop_unit = min(alignment_tokens, kv_cache_spec.block_size)
            max_length = max(0, max_length - drop_unit)

        return super().find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_length,
            kv_cache_group_ids=kv_cache_group_ids,
            block_pool=block_pool,
            kv_cache_spec=kv_cache_spec,
            alignment_tokens=alignment_tokens,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            # Ascend owns the ceiling adjustment above. Keep this false so a
            # future upstream implementation cannot apply the drop twice.
            drop_eagle_block=False,
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int | None = None,
        num_tokens_main_model: int | None = None,
        apply_admission_cap: bool = False,
    ) -> int:
        if num_tokens_main_model is None:
            assert num_local_computed_tokens is not None
            num_tokens_main_model = num_local_computed_tokens
        local_hit_tokens = len(new_computed_blocks) * self.block_size
        num_new_blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
        # When external KV cache is loaded synchronously with new
        # tokens, allocate_new_computed_blocks() allocates one
        # extra block to hold the external cache content. Account
        # for it here so the free-capacity check is accurate.
        # (External tokens exist when total_computed_tokens exceeds
        # what local prefix-cache hits cover; sync loading when
        # num_tokens_main_model exceeds total_computed_tokens.)
        has_external_tokens = total_computed_tokens > local_hit_tokens
        has_new_scheduled_tokens = num_tokens_main_model > total_computed_tokens
        if has_external_tokens and has_new_scheduled_tokens:
            # one more block for external computed tokens
            num_new_blocks += 1
        return num_new_blocks


single_type_kv_cache_manager.MambaManager = AscendMambaManager
