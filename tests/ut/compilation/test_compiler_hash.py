# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch_npu
import vllm.envs as vllm_envs
from vllm.compilation.backends import CompilerManager
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.config import SpeculativeConfig
from vllm.config.utils import Range

from vllm_ascend.compilation.compiler_interface import AscendCompiler
from vllm_ascend.lora import grouped_prefill
from vllm_ascend.patch.platform import patch_compile_cache

DYNAMIC_BRANCH_FLAGS = (
    "VLLM_ASCEND_MOE_LORA_FUSED_ROUTING",
    "VLLM_ASCEND_MOE_LORA_FUSED_ACTIVATION",
    "VLLM_ASCEND_MOE_LORA_FUSED_SHRINK_PAIR",
    "VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY",
)


@pytest.mark.parametrize("flag_name", ("VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL", *DYNAMIC_BRANCH_FLAGS))
@pytest.mark.parametrize("cached_enabled", [False, True])
def test_graph_branch_flag_isolates_persisted_compiler_cache(tmp_path, monkeypatch, cached_enabled, flag_name):
    """Same mode reloads its artifact; toggling the LoRA path misses that cache."""
    ascend_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(
            enable_npugraph_ex=True,
            enable_static_kernel=False,
        )
    )
    config = SimpleNamespace()
    compile_range = Range(start=1, end=8192)
    graph = Mock(name="fx_graph")
    handle = ("artifact_compile_range_1_8192_subgraph_0", str(tmp_path / "graph.py"))

    def initialize(enabled):
        if flag_name == "VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL":
            monkeypatch.setattr(grouped_prefill, "ENABLE_GROUPED_PREFILL", enabled)
        else:
            monkeypatch.setenv(flag_name, str(int(enabled)))
        manager = CompilerManager(SimpleNamespace())
        compiler_hash = manager.compute_hash(config)
        cache_dir = tmp_path / compiler_hash
        cache_dir.mkdir(exist_ok=True)
        manager.initialize_cache(str(cache_dir))
        return manager, compiler_hash

    with (
        patch("vllm.compilation.backends.make_compiler", side_effect=lambda _: AscendCompiler()),
        patch("vllm_ascend.compilation.compiler_interface.get_ascend_config", return_value=ascend_config),
        patch.object(torch_npu, "__version__", "test-version", create=True),
    ):
        original, original_hash = initialize(cached_enabled)
        original.cache[(compile_range, 0, AscendCompiler.name)] = {
            "graph_handle": handle,
            "cache_key": "saved-graph",
        }
        original.is_cache_updated = True
        original.save_to_file()

        changed, changed_hash = initialize(not cached_enabled)
        assert changed_hash != original_hash
        with patch.object(changed.compiler, "load") as load:
            assert changed.load(graph, [], 0, compile_range) is None
            load.assert_not_called()

        restored, restored_hash = initialize(cached_enabled)
        assert restored_hash == original_hash
        artifact = Mock(name="cached_artifact")
        with patch.object(restored.compiler, "load", return_value=artifact) as load:
            assert restored.load(graph, [], 0, compile_range) is artifact
            load.assert_called_once_with(handle, graph, [], 0, compile_range)


def test_compiler_hash_uses_effective_grouped_prefill_flag(monkeypatch):
    """Changing the environment after import must not mislabel the active path."""
    ascend_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(
            enable_npugraph_ex=True,
            enable_static_kernel=False,
        )
    )
    monkeypatch.setattr(grouped_prefill, "ENABLE_GROUPED_PREFILL", False)
    compiler = AscendCompiler()
    with (
        patch("vllm_ascend.compilation.compiler_interface.get_ascend_config", return_value=ascend_config),
        patch.object(torch_npu, "__version__", "test-version", create=True),
    ):
        monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL", "0")
        initial_hash = compiler.compute_hash(SimpleNamespace())
        monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL", "1")
        assert compiler.compute_hash(SimpleNamespace()) == initial_hash
        monkeypatch.setattr(grouped_prefill, "ENABLE_GROUPED_PREFILL", True)
        assert compiler.compute_hash(SimpleNamespace()) != initial_hash


def test_aot_cache_factors_include_effective_grouped_prefill_flag(monkeypatch):
    """The outer AOT key must separate modes before loading backend artifacts."""
    monkeypatch.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "0")
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL", "0")
    monkeypatch.setattr(grouped_prefill, "ENABLE_GROUPED_PREFILL", False)
    config = SimpleNamespace(compute_hash=lambda: "same-vllm-config")
    original_env = vllm_envs.compile_factors()
    original_aot = aot_compile_hash_factors(config)

    monkeypatch.setattr(grouped_prefill, "ENABLE_GROUPED_PREFILL", True)
    changed_env = vllm_envs.compile_factors()
    changed_aot = aot_compile_hash_factors(config)
    assert changed_env.pop("VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL") is True
    assert original_env.pop("VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL") is False
    assert changed_env == original_env
    assert changed_aot != original_aot

    # Environment mutation alone does not change the implementation already
    # selected at module import, and must not change its cache identity.
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_GROUPED_PREFILL", "1")
    assert aot_compile_hash_factors(config) == changed_aot
    monkeypatch.setattr(grouped_prefill, "ENABLE_GROUPED_PREFILL", False)
    assert aot_compile_hash_factors(config) == original_aot


@pytest.mark.parametrize("flag_name", DYNAMIC_BRANCH_FLAGS)
def test_aot_cache_factors_isolate_dynamic_decode_branches(monkeypatch, flag_name):
    """The outer AOT key must split modes before loading a backend graph."""
    monkeypatch.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "0")
    monkeypatch.setenv(flag_name, "0")
    config = SimpleNamespace(compute_hash=lambda: "same-vllm-config")
    original_env = vllm_envs.compile_factors()
    original_aot = aot_compile_hash_factors(config)

    monkeypatch.setenv(flag_name, "1")
    changed_env = vllm_envs.compile_factors()
    changed_aot = aot_compile_hash_factors(config)
    assert original_env.pop(flag_name) is False
    assert changed_env.pop(flag_name) is True
    assert changed_env == original_env
    assert changed_aot != original_aot

    monkeypatch.setenv(flag_name, "0")
    assert aot_compile_hash_factors(config) == original_aot


def test_all_decode_branch_combinations_isolate_both_cache_layers(monkeypatch):
    """Every combination gets its own key; default/unset agrees with zero."""
    ascend_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=True, enable_static_kernel=False)
    )
    compiler = AscendCompiler()
    config = SimpleNamespace(compute_hash=lambda: "same-vllm-config")
    monkeypatch.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "0")
    for flag in DYNAMIC_BRANCH_FLAGS:
        monkeypatch.delenv(flag, raising=False)

    with (
        patch("vllm_ascend.compilation.compiler_interface.get_ascend_config", return_value=ascend_config),
        patch.object(torch_npu, "__version__", "test-version", create=True),
    ):
        default_compiler = compiler.compute_hash(config)
        default_aot = aot_compile_hash_factors(config)
        compiler_keys, aot_keys = set(), set()
        for flags in product(("0", "1"), repeat=len(DYNAMIC_BRANCH_FLAGS)):
            for name, value in zip(DYNAMIC_BRANCH_FLAGS, flags):
                monkeypatch.setenv(name, value)
            compiler_key = compiler.compute_hash(config)
            aot_key = str(aot_compile_hash_factors(config))
            assert compiler_key not in compiler_keys
            assert aot_key not in aot_keys
            compiler_keys.add(compiler_key)
            aot_keys.add(aot_key)
            if all(value == "0" for value in flags):
                assert compiler_key == default_compiler
                assert aot_compile_hash_factors(config) == default_aot
        assert len(compiler_keys) == len(aot_keys) == 16


def fake_speculative_config(method):
    # Hashing only requires the normalized method and the draft HF config;
    # bypass model/config construction and never load weights.
    config = object.__new__(SpeculativeConfig)
    config.method = method
    config.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=(1, 2, 3)))
    return config


def test_aux_methods_isolate_aot_only_when_mtp_optimization_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    dspark = fake_speculative_config("dspark")
    eagle3 = fake_speculative_config("eagle3")
    original = SpeculativeConfig.compute_hash.__wrapped__
    # Both upstream hashes use the same aux-output flag and layer IDs.
    assert original(dspark) == original(eagle3)
    dspark_config = SimpleNamespace(compute_hash=dspark.compute_hash)
    eagle3_config = SimpleNamespace(compute_hash=eagle3.compute_hash)

    monkeypatch.setenv("VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY", "0")
    assert dspark.compute_hash() == original(dspark)
    assert eagle3.compute_hash() == original(eagle3)
    original_aot = aot_compile_hash_factors(dspark_config)
    assert original_aot == aot_compile_hash_factors(eagle3_config)

    monkeypatch.setenv("VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY", "1")
    assert dspark.compute_hash() != eagle3.compute_hash()
    assert aot_compile_hash_factors(dspark_config) != aot_compile_hash_factors(eagle3_config)
    monkeypatch.setenv("VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY", "0")
    assert dspark.compute_hash() == original(dspark)
    assert aot_compile_hash_factors(dspark_config) == original_aot


@pytest.mark.parametrize("runner_v2", ["0", "1", None])
def test_mtp_skip_decision_isolates_inner_cache_by_method(monkeypatch, runner_v2):
    monkeypatch.setenv("VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY", "1")
    if runner_v2 is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner_v2)
    compiler = AscendCompiler()
    ascend_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=True, enable_static_kernel=False)
    )
    dspark = SimpleNamespace(speculative_config=fake_speculative_config("dspark"))
    ordinary = SimpleNamespace(speculative_config=fake_speculative_config("mtp"))
    no_spec = SimpleNamespace(speculative_config=None)
    with (
        patch("vllm_ascend.compilation.compiler_interface.get_ascend_config", return_value=ascend_config),
        patch.object(torch_npu, "__version__", "test-version", create=True),
    ):
        normal_hash = compiler.compute_hash(ordinary)
        assert compiler.compute_hash(no_spec) == normal_hash
        if runner_v2 == "0":
            assert compiler.compute_hash(dspark) != normal_hash
        else:
            # V2 and automatic selection retain the ordinary buffer.
            assert compiler.compute_hash(dspark) == normal_hash


def test_official_runner_factor_isolates_aot_without_virtual_env(monkeypatch):
    monkeypatch.setenv("VLLM_USE_MEGA_AOT_ARTIFACT", "0")
    config = SimpleNamespace(compute_hash=lambda: "same-vllm-config")
    keys = set()
    for value in (None, "0", "1"):
        if value is None:
            monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
        else:
            monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
        assert vllm_envs.compile_factors()["VLLM_USE_V2_MODEL_RUNNER"] is (None if value is None else value == "1")
        key = str(aot_compile_hash_factors(config))
        assert key not in keys
        keys.add(key)


def test_speculative_hash_patch_is_idempotent_under_reload(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_OPTIMIZE_MTP_BUFFER_COPY", "1")
    config = fake_speculative_config("dspark")
    patched = SpeculativeConfig.compute_hash
    expected = config.compute_hash()
    assert patch_compile_cache._with_mtp_method_compile_factor(patched) is patched
    for _ in range(2):
        importlib.reload(patch_compile_cache)
        assert SpeculativeConfig.compute_hash is patched
        assert config.compute_hash() == expected
