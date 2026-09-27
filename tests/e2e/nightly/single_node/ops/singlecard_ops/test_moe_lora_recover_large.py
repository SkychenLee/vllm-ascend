# SPDX-License-Identifier: Apache-2.0
"""Exact large complete-permutation routing recovery and replay gates."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.lora.fused_moe import _recover_moe_lora_routing_allgather
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    assert enable_custom_op(), "Build the Ascend custom operators before running this test."


def make_case(tokens, top_k, slot_count, index_dtype, expert_dtype, *, seed=0, owners=0):
    rows = tokens * top_k
    if owners:
        # Adjacent destinations belong to different contiguous source chunks.
        # A naive source-partitioned scalar GM scatter shares output cachelines.
        chunk = (rows + owners - 1) // owners
        sources = [owner * chunk + offset for offset in range(chunk) for owner in range(owners)]
        sources = [source for source in sources if source < rows]
        destinations = [0] * rows
        for destination, source in enumerate(sources):
            destinations[source] = destination
        expanded = torch.tensor(destinations, dtype=index_dtype)
    else:
        generator = torch.Generator().manual_seed(193 + seed)
        expanded = torch.randperm(rows, generator=generator).to(index_dtype)
    expanded[seed % 3 :: 3].neg_()
    # A negative destination of -1 is valid in the signed permutation contract.
    expanded[expanded == 1] = -1
    experts = torch.arange(rows, dtype=torch.int64) * 104729 + (1 << 40)
    experts[::3].neg_()
    experts = experts.to(expert_dtype)
    limits = torch.iinfo(expert_dtype)
    if rows >= 4:
        experts[:4] = torch.tensor([limits.min, limits.max, -1, 0], dtype=expert_dtype)
    experts = experts.roll(seed).reshape(tokens, top_k)
    slot_values = torch.tensor([-(1 << 63), (1 << 63) - 1, -1, 0, 1 << 40], dtype=torch.int64)
    slots = slot_values.repeat((slot_count + 4) // 5)[:slot_count].roll(seed)
    return expanded, experts, slots


def integer_reference(inputs, top_k):
    """Python integer destination assignment, independent of sorting kernels."""
    expanded, experts, slots = inputs
    source_experts, source_slots = experts.reshape(-1).tolist(), slots.tolist()
    expected_experts, expected_slots = [None] * expanded.numel(), [None] * expanded.numel()
    for source, signed_destination in enumerate(expanded.tolist()):
        destination = abs(signed_destination)
        assert expected_experts[destination] is None, "test input must be a complete permutation"
        expected_experts[destination] = source_experts[source]
        expected_slots[destination] = source_slots[min(source // top_k, len(source_slots) - 1)]
    return torch.tensor(expected_experts, dtype=torch.int64), torch.tensor(expected_slots, dtype=torch.int64)


def sort_reference(inputs, top_k):
    expanded, experts, slots = inputs
    inverse = expanded.abs().float().argsort()
    return (
        experts.reshape(-1).index_select(0, inverse).long(),
        slots.index_select(0, (inverse // top_k).clamp_max(slots.numel() - 1)),
    )


def assert_exact(outputs, inputs, cpu_inputs, top_k):
    integer = integer_reference(cpu_inputs, top_k)
    established = sort_reference(inputs, top_k)
    for actual, expected, old in zip(outputs, integer, established):
        assert actual.dtype == torch.int64
        assert actual.shape == cpu_inputs[0].shape
        assert actual.is_contiguous() and actual.device == inputs[0].device
        assert torch.equal(actual.cpu(), expected)
        assert torch.equal(actual, old)
        assert not any(torch._C._is_alias_of(actual, source) for source in inputs)
    assert not torch._C._is_alias_of(*outputs)
    for actual, original in zip(inputs, cpu_inputs):
        assert torch.equal(actual.cpu(), original)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "tokens,top_k,slot_count",
    [
        (8192, 6, 8192),  # DeepSeek 49152 routed rows.
        (8192, 8, 8192),  # Qwen 65536 routed rows.
        (8193, 6, 13),  # Tail and short slot table use exact floor/clamp.
        (8193, 8, 1),
        (8160, 1, 8160),
        (8161, 1, 7),
        (2731, 3, 2731),  # 8193 rows, odd top-k and odd tail.
    ],
)
@torch.inference_mode()
def test_large_recover_exact_signed_permutation(index_dtype, expert_dtype, tokens, top_k, slot_count):
    cpu_inputs = make_case(tokens, top_k, slot_count, index_dtype, expert_dtype)
    inputs = tuple(value.npu() for value in cpu_inputs)
    outputs = torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
    assert_exact(outputs, inputs, cpu_inputs, top_k)


@pytest.mark.parametrize("rows", [8160, 8161, 8162, 262144, 262145])
@torch.inference_mode()
def test_large_recover_dispatch_and_workspace_boundaries(rows):
    # The candidate interval is [8161, 262144], with 64B scratch per row.
    # Both adjacent fallbacks must preserve the identical integer contract;
    # 8162 exercises a partial 512-row tile; the 8193-row case has one tail row.
    cpu_inputs = make_case(rows, 1, 13, torch.int32, torch.int64)
    inputs = tuple(value.npu() for value in cpu_inputs)
    outputs = torch.ops._C_ascend.moe_lora_recover(*inputs, 1)
    assert_exact(outputs, inputs, cpu_inputs, 1)


@pytest.mark.parametrize("top_k", [6, 8])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_large_recover_graph_reads_updated_permutation_experts_and_slots(top_k, index_dtype):
    # Both main shapes lie inside the native large-row interval. Tensor data
    # changes in place while the captured shapes and addresses stay constant.
    tokens, slot_count = 8192, 19
    cpu_inputs = make_case(tokens, top_k, slot_count, index_dtype, torch.int64)
    inputs = tuple(value.npu() for value in cpu_inputs)
    for _ in range(3):
        torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
    for seed in (1, 2, 3):
        current = make_case(tokens, top_k, slot_count, index_dtype, torch.int64, seed=seed)
        if seed == 2:
            current[2].fill_(2)  # Single adapter in the final slot.
        elif seed == 3:
            current[2].fill_(-1)  # Every adapter inactive.
        for destination, source in zip(inputs, current):
            destination.copy_(source)
        graph.replay()
        assert_exact(outputs, inputs, current, top_k)


@pytest.mark.parametrize("top_k,owners", [(6, 16), (6, 48), (8, 16), (8, 48)])
@torch.inference_mode()
def test_large_recover_interleaved_destination_cachelines(top_k, owners):
    cpu_inputs = make_case(8192, top_k, 8192, torch.int32, torch.int64, owners=owners)
    inputs = tuple(value.npu() for value in cpu_inputs)
    expected = integer_reference(cpu_inputs, top_k)
    # Repeated launches exercise lost updates which one random permutation can
    # miss; payload bits deliberately vary even within one destination line.
    for _ in range(24):
        actual = torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
        for output, reference in zip(actual, expected):
            assert torch.equal(output.cpu(), reference)


@pytest.mark.parametrize("experts,top_k", [(256, 6), (128, 8)])
@pytest.mark.parametrize("quant_mode", [-1, 1])
@pytest.mark.parametrize("distribution", ["balanced", "skewed"])
@torch.inference_mode()
def test_large_recover_real_routing_producer(experts, top_k, quant_mode, distribution):
    tokens = 8192
    rows = tokens * top_k
    # Skew includes repeated top-k experts and many empty experts.
    topk = torch.arange(rows, dtype=torch.int32).reshape(tokens, top_k)
    topk.remainder_(experts if distribution == "balanced" else 5)
    slots = torch.arange(tokens, dtype=torch.int64).remainder(4) - 1
    x = torch.randn(tokens, 64, dtype=torch.bfloat16, device="npu")
    topk_npu, slots_npu = topk.npu(), slots.npu()
    _, expanded, counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        topk_npu,
        active_num=rows,
        expert_num=experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, experts],
        quant_mode=quant_mode,
    )
    cpu_expanded = expanded.cpu()
    assert torch.equal(cpu_expanded.abs().sort().values.long(), torch.arange(rows))
    assert counts.sum().item() == rows
    context = SimpleNamespace(
        use_ep=False, fully_sharded=True, top_k=top_k, punica_wrapper=SimpleNamespace(token_lora_indices=slots_npu)
    )
    outputs = _recover_moe_lora_routing_allgather(context, expanded, topk_npu)
    assert_exact(outputs, (expanded, topk_npu, slots_npu), (cpu_expanded, topk, slots), top_k)
