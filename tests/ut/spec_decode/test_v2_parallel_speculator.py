# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode import speculator as base_speculator

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as dspark


@pytest.fixture(params=[(dspark, dspark.AscendDSparkSpeculator), (dflash, dflash.AscendDFlashSpeculator)])
def parallel_speculator(request):
    module, speculator_cls = request.param
    speculator = speculator_cls.__new__(speculator_cls)
    speculator.vllm_config = SimpleNamespace()
    speculator.device = torch.device("cpu")
    speculator.max_num_tokens = 8
    speculator.model = SimpleNamespace()
    return module, speculator


@pytest.mark.parametrize("separate_draft_groups", [False, True])
@pytest.mark.parametrize("filter_draft_layers", [False, True])
@pytest.mark.skipif(not vllm_version_is("0.25.1"), reason="Exercises the real v0.25.1 parent interface")
def test_set_attn_matches_v025_runner(parallel_speculator, separate_draft_groups, filter_draft_layers):
    module, speculator = parallel_speculator
    draft_names = ["draft.0", "draft.1"]
    speculator.draft_attn_layer_names = set(draft_names) if filter_draft_layers else None
    layer_groups = [["target.0"], *([[name] for name in draft_names] if separate_draft_groups else [draft_names])]
    kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=names) for names in layer_groups])
    attn_groups = [([] if filter_draft_layers and names == ["target.0"] else [object()]) for names in layer_groups]
    block_tables = SimpleNamespace(block_sizes=[16 * (i + 1) for i in range(len(layer_groups))])
    model_state = object()
    backends = {name: type(f"Backend_{name.replace('.', '_')}", (), {}) for names in layer_groups for name in names}
    layers = {name: Mock(get_attn_backend=Mock(return_value=backend)) for name, backend in backends.items()}

    def get_layers(config, layer_type, names):
        return {name: layers[name] for name in names}

    with (
        patch.object(base_speculator, "init_attn_backend", return_value=(attn_groups, None, None)) as init_backend,
        patch.object(module, "get_layers_from_vllm_config", side_effect=get_layers),
    ):
        # Exercise the actual upstream set_attn implementation, using the same
        # three arguments as GPUModelRunner.initialize_kv_cache in vLLM v0.25.
        speculator.set_attn(model_state, kv_cache_config, block_tables)

    init_backend.assert_called_once_with(
        kv_cache_config,
        speculator.vllm_config,
        speculator.device,
        active_layer_names=speculator.draft_attn_layer_names,
    )
    group_ids = [i for i, groups in enumerate(attn_groups) if groups]
    assert speculator.model_state is model_state
    assert speculator.kv_cache_config is kv_cache_config
    assert speculator.block_tables is block_tables
    assert speculator.attn_groups is attn_groups
    assert speculator.draft_kv_cache_group_ids == group_ids
    assert speculator.draft_block_size == block_tables.block_sizes[group_ids[0]]
    assert speculator._context_slot_mappings.dtype == torch.int32
    assert speculator._context_slot_mappings.shape == (len(group_ids), speculator.max_num_tokens)
    assert torch.count_nonzero(speculator._context_slot_mappings) == 0
    expected_names = set(draft_names) if filter_draft_layers else set(backends)
    assert speculator.attn_backends == {name: backends[name] for name in expected_names}


@pytest.mark.skipif(not vllm_version_is("0.25.1"), reason="Exercises the real v0.25.1 parent interface")
def test_set_attn_rejects_missing_draft_groups(parallel_speculator):
    _, speculator = parallel_speculator
    speculator.draft_attn_layer_names = {"draft.0"}
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    with (
        patch.object(base_speculator, "init_attn_backend", return_value=([[]], None, None)),
        pytest.raises(AssertionError, match="No draft attention groups found"),
    ):
        speculator.set_attn(object(), kv_cache_config, SimpleNamespace(block_sizes=[16]))


def test_set_attn_forwards_target_attention_on_newer_vllm(parallel_speculator):
    module, speculator = parallel_speculator
    model_state, block_tables, target_buffers, target_groups = (object() for _ in range(4))
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    speculator.draft_attn_layer_names = set()

    def parent_set_attn(instance, state, cache_config, tables, buffers, groups):
        assert instance is speculator
        assert (state, cache_config, tables, buffers, groups) == (
            model_state,
            kv_cache_config,
            block_tables,
            target_buffers,
            target_groups,
        )
        instance.draft_kv_cache_group_ids = [0]
        instance._context_slot_mappings = torch.zeros((1, instance.max_num_tokens), dtype=torch.int64)

    with (
        patch.object(module, "vllm_version_is", return_value=False),
        patch.object(type(speculator).__base__, "set_attn", parent_set_attn),
    ):
        speculator.set_attn(model_state, kv_cache_config, block_tables, target_buffers, target_groups)

    assert speculator._context_slot_mappings.dtype == torch.int32
    assert speculator.attn_backends == {}


@pytest.mark.parametrize("is_v025", [True, False])
@pytest.mark.parametrize("causal", [True, False])
def test_graph_metadata_uses_version_specific_causality(parallel_speculator, is_v025, causal):
    module, speculator = parallel_speculator
    speculator.num_query_per_req = 3
    # Only expose the attribute available in the corresponding upstream version.
    expected_causal = causal if is_v025 else {0: causal}
    if is_v025:
        speculator.dflash_causal = expected_causal
    else:
        speculator._group_causal = expected_causal
    metadata = object()
    with (
        patch.object(module, "vllm_version_is", return_value=is_v025),
        patch.object(module, "build_attn_metadata_wrapper", return_value=nullcontext()),
        patch.object(speculator, "_build_draft_attn_metadata", return_value=metadata) as build_metadata,
    ):
        assert speculator.build_draft_attn_metadatas(2) == [metadata]

    build_metadata.assert_called_once_with(num_reqs=2, num_reqs_padded=2, num_tokens_padded=6, causal=expected_causal)
