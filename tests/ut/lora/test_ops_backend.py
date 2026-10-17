# SPDX-License-Identifier: Apache-2.0
"""Backend selection must not hide a broken standalone LoRA installation."""

import importlib.util
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.lora import lora_ops, ops_backend


@pytest.fixture(autouse=True)
def clear_backend_cache(monkeypatch):
    if hasattr(torch, "npu"):
        monkeypatch.setattr(
            torch.npu,
            "_lazy_init",
            Mock(side_effect=AssertionError("backend tests must not initialize NPU hardware")),
        )
    ops_backend._load_lora_ops.cache_clear()
    yield
    ops_backend._load_lora_ops.cache_clear()


def _standalone_backend():
    backend = ModuleType("ops_repo")
    backend.__spec__ = importlib.util.spec_from_loader("ops_repo", loader=None)
    for name in ops_backend._LORA_OP_NAMES:
        setattr(backend, name, Mock(name=name))
    return backend


def test_import_does_not_resolve_optional_library(monkeypatch):
    find_spec = Mock(side_effect=AssertionError("optional library resolved during import"))
    import_module = Mock(side_effect=AssertionError("optional library loaded during import"))
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr("importlib.import_module", import_module)
    spec = importlib.util.spec_from_file_location("isolated_lora_backend", ops_backend.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    find_spec.assert_not_called()
    import_module.assert_not_called()


def test_absent_library_uses_existing_namespace(monkeypatch):
    monkeypatch.setattr(ops_backend, "find_spec", Mock(return_value=None))
    load = Mock(side_effect=AssertionError("missing library must not be imported"))
    monkeypatch.setattr(ops_backend, "import_module", load)
    assert ops_backend.get_lora_ops() is torch.ops._C_ascend
    load.assert_not_called()


def test_standalone_library_is_preferred_and_resolved_once(monkeypatch):
    backend = _standalone_backend()
    monkeypatch.setitem(sys.modules, "ops_repo", backend)
    backend.__file__ = "/test/site-packages/ops_repo/__init__.py"
    logger = Mock()
    monkeypatch.setattr(ops_backend, "logger", logger)
    discover = Mock(return_value=object())
    load = Mock(return_value=backend)
    monkeypatch.setattr(ops_backend, "find_spec", discover)
    monkeypatch.setattr(ops_backend, "import_module", load)
    assert ops_backend.get_lora_ops() is backend
    assert ops_backend.get_lora_ops() is backend
    discover.assert_called_once_with("ops_repo")
    load.assert_called_once_with("ops_repo")
    logger.info.assert_called_once_with("Using ops_repo LoRA operators from %s (torch.ops.ops_repo).", backend.__file__)


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("missing dependency", name="ops_repo._C"),
        OSError("undefined symbol in native extension"),
        RuntimeError("operator registration failed"),
    ],
)
def test_broken_installation_is_not_silently_bypassed(monkeypatch, error):
    monkeypatch.setattr(ops_backend, "find_spec", Mock(return_value=object()))
    monkeypatch.setattr(ops_backend, "import_module", Mock(side_effect=error))
    with pytest.raises(ImportError, match="installed but could not be imported") as caught:
        ops_backend.get_lora_ops()
    assert caught.value.__cause__ is error


def test_old_or_partial_api_reports_missing_operators(monkeypatch):
    backend = _standalone_backend()
    monkeypatch.setitem(sys.modules, "ops_repo", backend)
    del backend.add_lora_swiglu_quant
    backend.add_lora_expand_delta = None
    monkeypatch.setattr(ops_backend, "find_spec", Mock(return_value=object()))
    monkeypatch.setattr(ops_backend, "import_module", Mock(return_value=backend))
    with pytest.raises(ImportError, match="add_lora_expand_delta, add_lora_swiglu_quant"):
        ops_backend.get_lora_ops()


@pytest.mark.parametrize(
    ("wrapper", "operator", "segmented", "sliced", "shrink"),
    [
        ("bgmv_shrink", "bgmv_shrink", False, False, True),
        ("bgmv_expand", "bgmv_expand", False, False, False),
        ("bgmv_expand_slice", "bgmv_expand", False, True, False),
        ("sgmv_shrink", "sgmv_shrink", True, False, True),
        ("sgmv_expand", "sgmv_expand", True, False, False),
        ("sgmv_expand_slice", "sgmv_expand", True, True, False),
    ],
)
def test_dense_wrappers_preserve_native_argument_order(monkeypatch, wrapper, operator, segmented, sliced, shrink):
    backend = _standalone_backend()
    monkeypatch.setitem(sys.modules, "ops_repo", backend)
    monkeypatch.setattr(ops_backend, "find_spec", Mock(return_value=object()))
    monkeypatch.setattr(ops_backend, "import_module", Mock(return_value=backend))
    inputs, weights, output = torch.empty(2, 16), torch.empty(1, 16, 16), torch.empty(2, 32)
    indices = torch.tensor([0, -1])
    seq_lens = torch.tensor([1, 1])
    args = [inputs, weights, output]
    expected = [inputs, weights, indices]
    if segmented:
        args += [torch.tensor([0, 1]), seq_lens, indices, 2, 1, 2]
        expected.append(seq_lens)
    else:
        args.append(indices)
    expected.append(output)
    if shrink:
        args.append(0.5)
        expected.append(0.5)
    elif sliced:
        args += [8, 16]
        expected += [8, 16]
    else:
        expected += [0, 32]
    result = getattr(lora_ops, wrapper)(*args)
    selected_op = getattr(backend, operator)
    selected_op.assert_called_once_with(*expected)
    assert result is selected_op.return_value


def test_fake_tensors_reach_selected_operator_without_materialization(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    backend = _standalone_backend()
    monkeypatch.setitem(sys.modules, "ops_repo", backend)
    monkeypatch.setattr(ops_backend, "find_spec", Mock(return_value=object()))
    monkeypatch.setattr(ops_backend, "import_module", Mock(return_value=backend))
    with FakeTensorMode():
        inputs = torch.empty(2, 16)
        weights = torch.empty(1, 8, 16)
        output = torch.empty(2, 8)
        indices = torch.empty(2, dtype=torch.int64)
        lora_ops.bgmv_shrink(inputs, weights, output, indices)
    backend.bgmv_shrink.assert_called_once_with(inputs, weights, indices, output, 1.0)


@pytest.fixture(scope="module")
def compile_test_op():
    # A CPU custom op with mutation and Meta behavior exercises the same
    # dispatcher/graph boundary as the native LoRA op without launching NPU work.
    library = torch.library.Library("ascend_lora_backend_test", "DEF")
    library.define("shrink(Tensor x, Tensor w, Tensor idx, Tensor(a!) y, float scale) -> ()")

    def cpu_impl(x, w, idx, y, scale):
        y.copy_(x.matmul(w[0].T) * scale)

    library.impl("shrink", cpu_impl, "CPU")
    library.impl("shrink", lambda x, w, idx, y, scale: None, "Meta")
    yield torch.ops.ascend_lora_backend_test.shrink


def _compile_test_standalone_shrink(x, w, idx, y, scale):
    return torch.ops.ascend_lora_backend_test.shrink(x, w, idx, y, scale)


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_backend_selection_preserves_fullgraph_compile(monkeypatch, compile_test_op, standalone, warm):
    from torch._dynamo.device_interface import get_registered_device_interfaces

    # Inductor probes every available device even when compiling CPU graphs.
    # Suppress that probe and keep _lazy_init blocked by the fixture above.
    for device, interface in get_registered_device_interfaces():
        if device.split(":")[0] == "npu":
            monkeypatch.setattr(interface, "is_available", staticmethod(lambda: False))
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    torch._dynamo.reset()
    if standalone:
        backend = _standalone_backend()
        backend.bgmv_shrink = _compile_test_standalone_shrink
        monkeypatch.setitem(sys.modules, "ops_repo", backend)
        monkeypatch.setattr(ops_backend, "find_spec", Mock(return_value=object()))
        monkeypatch.setattr(ops_backend, "import_module", Mock(return_value=backend))
    else:
        monkeypatch.setattr(ops_backend, "find_spec", Mock(return_value=None))
        monkeypatch.setattr(torch.ops._C_ascend, "bgmv_shrink", compile_test_op, raising=False)
    if warm:
        ops_backend.get_lora_ops()

    def apply(x, w, y, idx):
        lora_ops.bgmv_shrink(x, w, y, idx, 0.5)
        return y

    x = torch.randn(2, 16)
    w = torch.randn(1, 8, 16)
    y = torch.empty(2, 8)
    idx = torch.zeros(2, dtype=torch.int64)
    compiled = torch.compile(apply, backend="eager", fullgraph=True)
    try:
        torch.testing.assert_close(compiled(x, w, y, idx), x.matmul(w[0].T) * 0.5)
    finally:
        torch._dynamo.reset()
