# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.compilation.acl_graph import GraphParams, _GraphParamStore, weak_ref_workspaces


@dataclass(frozen=True)
class _Descriptor:
    num_tokens: int
    has_lora: bool


def _forward_context(descriptor: _Descriptor) -> SimpleNamespace:
    return SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=descriptor,
    )


def test_graph_param_store_isolates_base_and_lora_descriptors() -> None:
    store = _GraphParamStore([4], list)
    base_descriptor = _Descriptor(num_tokens=4, has_lora=False)
    lora_descriptor = _Descriptor(num_tokens=4, has_lora=True)

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(base_descriptor),
    ):
        store[4].append("base")

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(lora_descriptor),
    ):
        store[4].append("lora")

    assert dict.__getitem__(store, base_descriptor) == ["base"]
    assert dict.__getitem__(store, lora_descriptor) == ["lora"]
    assert dict.__getitem__(store, 4) == []


def test_graph_param_store_keeps_integer_key_outside_full_graph_context() -> None:
    store = _GraphParamStore([8], list)

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        side_effect=AssertionError,
    ):
        store[8].append("eager")

    assert store.get(8) == ["eager"]
    assert 8 in store


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_graph_param_store_keeps_integer_key_without_full_replay(mode):
    store = _GraphParamStore([4], list)
    context = _forward_context(_Descriptor(4, True))
    context.cudagraph_runtime_mode = mode
    with patch("vllm_ascend.compilation.acl_graph.get_forward_context", return_value=context):
        store[4].append("non-full")
    assert dict.__getitem__(store, 4) == ["non-full"]
    assert len(store) == 1


def test_graph_param_store_does_not_alias_another_token_count():
    store = _GraphParamStore([4, 8], list)
    context = _forward_context(_Descriptor(4, True))
    with patch("vllm_ascend.compilation.acl_graph.get_forward_context", return_value=context):
        store[8].append("other-size")
        assert store.get(4) is None
        assert 4 not in store
        store[4].append("lora")
        assert 4 in store
    assert dict.__getitem__(store, 8) == ["other-size"]
    assert dict.__getitem__(store, 4) == []


def test_weak_ref_workspaces_preserves_all_stored_graph_keys():
    store = _GraphParamStore([4], lambda: None)
    base = _Descriptor(4, False)
    lora = _Descriptor(4, True)
    dict.__setitem__(store, 4, "integer-workspace")
    store[base] = "base-workspace"
    store[lora] = "lora-workspace"
    params = GraphParams({}, store, {}, {})
    with (
        patch("vllm_ascend.compilation.acl_graph.get_forward_context", return_value=_forward_context(lora)),
        patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors", side_effect=lambda value: ("weak", value)),
    ):
        weak_ref_workspaces(params)
    assert dict.__getitem__(store, 4) == ("weak", "integer-workspace")
    assert store[base] == ("weak", "base-workspace")
    assert store[lora] == ("weak", "lora-workspace")
