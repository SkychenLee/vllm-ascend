from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.lora.fused_moe import _recover_moe_lora_routing_allgather


@pytest.mark.parametrize("use_ep", [False, None, 0])
@pytest.mark.parametrize("tokens,top_k,slot_count", [(0, 3, 0), (7, 3, 2), (513, 1, 7)])
def test_cpu_routing_recovery_never_dispatches_native(use_ep, tokens, top_k, slot_count):
    rows = tokens * top_k
    expanded = torch.arange(rows, dtype=torch.int64).flip(0)
    expanded[::2].neg_()
    experts = torch.arange(rows, dtype=torch.int64).reshape(tokens, top_k) + 2**40
    slots = torch.full((slot_count,), -(2**63), dtype=torch.int64)
    if slot_count:
        slots[-1] = 2**63 - 1
    context = SimpleNamespace(top_k=top_k, punica_wrapper=SimpleNamespace(token_lora_indices=slots))
    if use_ep is not None:
        context.use_ep = use_ep
    snapshots = [value.clone() for value in (expanded, experts, slots)]
    with patch.object(torch.ops._C_ascend, "moe_lora_recover", create=True) as native:
        outputs = _recover_moe_lora_routing_allgather(context, expanded, experts)
        native.assert_not_called()
    assert isinstance(outputs, (tuple, list)) and len(outputs) == 2
    expected_experts = [0] * rows
    expected_slots = [0] * rows
    for original, destination in enumerate(expanded.tolist()):
        expected_experts[abs(destination)] = 2**40 + original
        expected_slots[abs(destination)] = slots[min(original // top_k, slot_count - 1)].item()
    for actual, expected in zip(outputs, (expected_experts, expected_slots)):
        assert actual.device.type == "cpu"
        assert actual.dtype == torch.int64
        assert torch.equal(actual, torch.tensor(expected, dtype=torch.int64))
    for value, snapshot in zip((expanded, experts, slots), snapshots):
        assert torch.equal(value, snapshot)


def test_ep_routing_recovery_filters_remote_experts_and_uses_local_ids():
    expanded = torch.tensor([0, -1, 2, -1, 1, -1], dtype=torch.int32)
    experts = torch.tensor([[5, 2], [3, 1], [4, 0]], dtype=torch.int64)
    slots = torch.tensor([7, -1, 19], dtype=torch.int64)
    context = SimpleNamespace(top_k=2, use_ep=True, punica_wrapper=SimpleNamespace(token_lora_indices=slots))
    result = _recover_moe_lora_routing_allgather(context, expanded, experts, expert_start=3, num_local_experts=3)
    assert torch.equal(result[0], torch.tensor([2, 1, 0, -1, -1, -1]))
    assert torch.equal(result[1], torch.tensor([7, 19, -1, -1, -1, -1]))
