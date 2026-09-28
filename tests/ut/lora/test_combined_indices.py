# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.lora.combined_indices import combined_moe_lora_indices


@pytest.mark.parametrize("rows", [0, 1, 7, 8, 9, 36, 288, 512, 513, 49152, 98304])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cpu_reference_and_no_native(rows, dtype, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", "1")
    experts = torch.arange(rows, dtype=dtype) % 256
    slots = torch.arange(rows, dtype=torch.int64) % 5 - 1
    mask = torch.tensor([0, 1, -7, 0], dtype=torch.int32)
    native = Mock()
    with patch.object(torch.ops, "_C_ascend", SimpleNamespace(moe_lora_combined_indices=native)):
        result = combined_moe_lora_indices(experts, slots, mask, 256)
    expected = [int(s) * 256 + int(e) if s >= 0 and mask[s] != 0 else -1 for e, s in zip(experts, slots)]
    assert torch.equal(result, torch.tensor(expected, dtype=torch.int64))
    native.assert_not_called()


@pytest.mark.parametrize(
    "rows,flag,has_op,expect_native",
    [
        (36, "1", True, True),
        (288, "1", True, True),
        (512, "1", True, True),
        (513, "1", True, True),
        (98304, "1", True, True),
        (98305, "1", True, False),
        (36, "0", True, False),
        (36, "1", False, False),
    ],
)
def test_dispatch_guards_and_old_binary(rows, flag, has_op, expect_native, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", flag)
    experts = torch.zeros(rows, dtype=torch.int64)
    slots = torch.zeros(rows, dtype=torch.int64)
    mask = torch.ones(4, dtype=torch.int32)
    expected = torch.zeros(rows, dtype=torch.int64)
    native = Mock(return_value=expected)
    namespace = SimpleNamespace(moe_lora_combined_indices=native) if has_op else SimpleNamespace()
    # Metadata-only device shim exercises the real helper without NPU execution.
    device = SimpleNamespace(type="npu")
    with (
        patch.object(torch.Tensor, "device", property(lambda self: device)),
        patch.object(torch.ops, "_C_ascend", namespace),
    ):
        result = combined_moe_lora_indices(experts, slots, mask, 256)
    assert torch.equal(result, expected)
    assert native.call_count == int(expect_native)


def test_int64_fallback_and_strict_flag(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", raising=False)
    assert envs.VLLM_ASCEND_MOE_LORA_FUSED_ROUTING is False
    experts = torch.tensor([2**60 + 17, -(2**63), 2**63 - 1], dtype=torch.int64)
    slots = torch.tensor([1, -1, 0], dtype=torch.int64)
    mask = torch.tensor([-2, 1], dtype=torch.int32)
    assert combined_moe_lora_indices(experts, slots, mask, 256).tolist() == [2**60 + 273, -1, 2**63 - 1]
    monkeypatch.setenv("VLLM_ASCEND_MOE_LORA_FUSED_ROUTING", "invalid")
    with pytest.raises(ValueError):
        _ = envs.VLLM_ASCEND_MOE_LORA_FUSED_ROUTING


@pytest.fixture(scope="module")
def native_meta():
    library = os.environ.get("MOE_LORA_COMBINED_TEST_LIBRARY")
    if not library:
        pytest.skip("Set MOE_LORA_COMBINED_TEST_LIBRARY to the diagnostic .so for native Meta tests")
    import torch_npu  # noqa: F401

    torch.ops.load_library(library)
    return torch.ops._C_ascend.moe_lora_combined_indices


@pytest.mark.parametrize("rows", [0, 36, 288, 98304])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_native_meta_output(native_meta, rows, dtype):
    experts = torch.empty(rows, device="meta", dtype=dtype)
    slots = torch.empty(rows, device="meta", dtype=torch.int64)
    mask = torch.empty(4, device="meta", dtype=torch.int32)
    result = native_meta(experts, slots, mask, 256)
    assert result.shape == experts.shape and result.dtype == torch.int64 and result.device.type == "meta"
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::moe_lora_combined_indices", "PrivateUse1")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::moe_lora_combined_indices", "Meta")


@pytest.mark.parametrize("error", ["dtype", "rows", "mask", "contiguous", "num_experts"])
def test_native_meta_contract_errors(native_meta, error):
    experts = torch.empty(36, device="meta", dtype=torch.int64)
    slots = torch.empty(36, device="meta", dtype=torch.int64)
    mask = torch.empty(4, device="meta", dtype=torch.int32)
    count = 256
    if error == "dtype":
        experts = experts.float()
    elif error == "rows":
        slots = slots[:1]
    elif error == "mask":
        mask = torch.empty(65, device="meta", dtype=torch.int32)
    elif error == "contiguous":
        experts = torch.empty(72, device="meta", dtype=torch.int64)[::2]
    else:
        count = 0
    with pytest.raises(RuntimeError):
        native_meta(experts, slots, mask, count)
