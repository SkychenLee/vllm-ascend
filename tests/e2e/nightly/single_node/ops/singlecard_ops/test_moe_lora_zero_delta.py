# SPDX-License-Identifier: Apache-2.0
"""An active zero-B adapter must not change the same BGMV base boundary."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

import vllm_ascend.ops  # noqa: F401 -- initializes Ascend/Punica integration
from vllm_ascend.lora.lora_ops import bgmv_expand_slice, bgmv_shrink
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("rows", [1, 8, 128])
@pytest.mark.parametrize("rank", [16, 32])
@torch.inference_mode()
def test_zero_moe_lora_delta_preserves_base(rows, rank):
    enable_custom_op()
    hidden_size, output_size, experts, slots = 4096, 2048, 4, 2
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(rows, hidden_size, generator=generator).bfloat16().npu()
    a = (torch.randn(slots, experts, rank, hidden_size, generator=generator) * 0.01).bfloat16().npu()
    b = torch.zeros(slots, experts, output_size, rank, dtype=torch.bfloat16, device="npu")
    indices = (torch.arange(rows, device="npu") % (slots * experts)).long()
    if rows > 1:
        indices[-1] = -1
    base = torch.randn(rows, 2 * output_size, generator=generator).bfloat16().npu()
    context = SimpleNamespace(
        token_lora_indices=indices,
        bgmv_shrink=bgmv_shrink,
        bgmv_expand_slice=bgmv_expand_slice,
    )

    def apply(output):
        PunicaWrapperNPU.add_lora_fused_moe(
            context,
            output,
            x,
            (a, a),
            (b, b),
            expert_ids=None,
            adapter_enabled=torch.ones(slots, dtype=torch.int32, device="npu"),
            bgmv_lora_indices=indices,
        )

    actual = base.clone()
    apply(actual)
    torch.testing.assert_close(actual.cpu(), base.cpu(), rtol=0, atol=0)
    # A nonzero control proves this check did not accidentally bypass BGMV.
    b.fill_(0.125)
    apply(actual)
    assert not torch.equal(actual[0].cpu(), base[0].cpu())
    if rows > 1:
        torch.testing.assert_close(actual[-1].cpu(), base[-1].cpu(), rtol=0, atol=0)
