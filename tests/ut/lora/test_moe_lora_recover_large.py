# SPDX-License-Identifier: Apache-2.0
"""Large producer contracts remain separate for complete routing and EP."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.lora.fused_moe import _recover_moe_lora_routing_allgather


@pytest.mark.parametrize("top_k", [6, 8])
@pytest.mark.parametrize("use_ep", [False, None, 0])
def test_large_signed_complete_routing_cpu_fallback(top_k, use_ep):
    tokens = 8192
    rows = tokens * top_k
    expanded = torch.arange(rows, dtype=torch.int32).flip(0)
    expanded[1::2].neg_()
    experts = (torch.arange(rows, dtype=torch.int64) + (1 << 40)).reshape(tokens, top_k)
    slots = torch.tensor([-(1 << 63), -1, 0, (1 << 63) - 1], dtype=torch.int64)
    context = SimpleNamespace(top_k=top_k, fully_sharded=True, punica_wrapper=SimpleNamespace(token_lora_indices=slots))
    if use_ep is not None:
        context.use_ep = use_ep
    with patch.object(torch.ops._C_ascend, "moe_lora_recover", create=True) as native:
        outputs = _recover_moe_lora_routing_allgather(context, expanded, experts)
        native.assert_not_called()
    assert torch.equal(outputs[0], experts.reshape(-1).flip(0))
    source_tokens = list(range(tokens - 1, -1, -1))
    expected_slots = [slots[min(token, len(slots) - 1)].item() for token in source_tokens for _ in range(top_k)]
    assert torch.equal(outputs[1], torch.tensor(expected_slots, dtype=torch.int64))
    assert outputs[0].dtype == outputs[1].dtype == torch.int64
