from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.lora.fused_moe import _recover_moe_lora_routing_allgather
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def make_inputs(tokens, top_k, slot_count, index_dtype, expert_dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rows = tokens * top_k
    expanded = torch.randperm(rows, generator=generator).to(index_dtype)
    expanded[seed % 2 :: 2].neg_()
    limits = torch.iinfo(expert_dtype)
    pattern = torch.tensor([limits.min, limits.max, -1, 0, 17], dtype=expert_dtype)
    experts = pattern.repeat((rows + pattern.numel() - 1) // pattern.numel())[:rows].roll(seed)
    slot_pattern = torch.tensor([-(2**63), 2**63 - 1, -1, 0, 2**40], dtype=torch.int64)
    slots = slot_pattern.repeat((slot_count + 4) // 5)[:slot_count].roll(seed)
    return expanded, experts.reshape(tokens, top_k), slots


def integer_reference(inputs, top_k):
    expanded, experts, slots = inputs
    expected_experts = torch.empty(expanded.numel(), dtype=torch.int64)
    expected_slots = torch.empty_like(expected_experts)
    # Independent inverse-permutation construction in Python integer arithmetic.
    for original, destination in enumerate(expanded.tolist()):
        expected_experts[abs(destination)] = experts.flatten()[original]
        expected_slots[abs(destination)] = slots[min(original // top_k, slots.numel() - 1)]
    return expected_experts, expected_slots


def original_chain(inputs, top_k):
    expanded, experts, slots = inputs
    inverse = expanded.abs().float().argsort()
    return experts.flatten()[inverse].long(), slots[(inverse // top_k).clamp_(max=slots.numel() - 1)]


def assert_outputs(outputs, inputs, cpu_inputs, top_k):
    expected = integer_reference(cpu_inputs, top_k)
    original = original_chain(inputs, top_k)
    for output, reference, old in zip(outputs, expected, original):
        assert output.dtype == torch.int64
        assert output.shape == cpu_inputs[0].shape
        assert output.is_contiguous()
        assert output.device == inputs[0].device
        assert torch.equal(output.cpu(), reference)
        assert torch.equal(output, old)
        assert not any(torch._C._is_alias_of(output, source) for source in inputs)
    assert not torch._C._is_alias_of(*outputs)
    for source, original_input in zip(inputs, cpu_inputs):
        assert torch.equal(source.cpu(), original_input)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "tokens,top_k,slot_count",
    [
        pytest.param(0, 8, 0, id="empty"),
        pytest.param(1, 8, 1, id="decode"),
        pytest.param(8, 8, 8, id="batch-decode"),
        pytest.param(7, 3, 2, id="odd-topk-clamped-slots"),
        pytest.param(16, 5, 131, id="unused-slot-tail"),
        pytest.param(64, 8, 64, id="medium"),
        pytest.param(513, 1, 7, id="sort-cost-boundary"),
        pytest.param(205, 5, 17, id="sort-odd-tail"),
        pytest.param(256, 8, 256, id="sort-2048"),
        pytest.param(1024, 3, 1024, id="sort-non-power-of-two-topk"),
        pytest.param(580, 8, 580, id="prefill"),
        pytest.param(768, 8, 11, id="sort-clamped-slots"),
        pytest.param(250, 32, 3, id="sort-topk-32"),
        pytest.param(127, 63, 17, id="sort-odd-topk-63"),
        pytest.param(15, 511, 17, id="sort-large-topk"),
        pytest.param(1, 7999, 1, id="sort-single-token-large-topk"),
        pytest.param(7999, 1, 8192, id="sort-ub-tail"),
        pytest.param(8000, 1, 8000, id="sort-ub-boundary"),
        pytest.param(8001, 1, 7, id="sort-resource-fallback"),
        pytest.param(1024, 16, 1024, id="larger-prefill-fallback"),
        pytest.param(8192, 8, 8192, id="large-resource-fallback"),
    ],
)
@torch.inference_mode()
def test_moe_lora_recover(index_dtype, expert_dtype, tokens, top_k, slot_count):
    cpu_inputs = make_inputs(tokens, top_k, slot_count, index_dtype, expert_dtype)
    inputs = tuple(value.npu() for value in cpu_inputs)
    outputs = torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
    assert_outputs(outputs, inputs, cpu_inputs, top_k)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "tokens,top_k,slot_count",
    [(13, 5, 7), (205, 5, 7), (580, 8, 580), (7999, 1, 3), (8001, 1, 5), (1024, 16, 1024)],
)
@torch.inference_mode()
def test_moe_lora_recover_changed_graph_inputs(index_dtype, expert_dtype, tokens, top_k, slot_count):
    cpu_inputs = make_inputs(tokens, top_k, slot_count, index_dtype, expert_dtype)
    inputs = tuple(value.npu() for value in cpu_inputs)
    for _ in range(3):
        torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
    for seed in (1, 2, 0):
        current = make_inputs(tokens, top_k, slot_count, index_dtype, expert_dtype, seed)
        for destination, source in zip(inputs, current):
            destination.copy_(source)
        graph.replay()
        assert_outputs(outputs, inputs, current, top_k)


@pytest.mark.parametrize("use_ep", [None, 0])
@torch.inference_mode()
def test_moe_lora_recover_filtered_context_retains_sort(use_ep):
    expanded = torch.tensor([0, -1, 2, -1, 1, -1], dtype=torch.int32, device="npu")
    experts = torch.tensor([[5, 2], [3, 1], [4, 0]], dtype=torch.int64, device="npu")
    slots = torch.tensor([7, -1, 19], dtype=torch.int64, device="npu")
    context = SimpleNamespace(top_k=2, punica_wrapper=SimpleNamespace(token_lora_indices=slots))
    if use_ep is not None:
        context.use_ep = use_ep
    with patch.object(torch.ops._C_ascend, "moe_lora_recover") as fused:
        outputs = _recover_moe_lora_routing_allgather(context, expanded, experts)
        fused.assert_not_called()
    assert isinstance(outputs, (tuple, list)) and len(outputs) == 2
    expected = original_chain((expanded, experts, slots), 2)
    for actual, reference in zip(outputs, expected):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize(
    "tokens,top_k,expert_start,num_local_experts",
    [
        (1, 6, 0, 32),
        (8, 6, 96, 32),
        (64, 8, 16, 16),
        (512, 6, 0, 32),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_moe_lora_recover_ep_matches_fixed_shape_reference(
    tokens, top_k, expert_start, num_local_experts, index_dtype, expert_dtype
):
    rows = tokens * top_k
    experts = torch.arange(rows, dtype=torch.int32).remainder(num_local_experts * 2)
    experts = (experts + expert_start - num_local_experts // 2).reshape(tokens, top_k)
    expanded = torch.full((rows,), -1, dtype=torch.int32)
    destinations = 0
    for source, expert in enumerate(experts.flatten().tolist()):
        if expert_start <= expert < expert_start + num_local_experts:
            expanded[source] = destinations
            destinations += 1
    slots = (torch.arange(tokens, dtype=torch.int64) % 3) - 1
    expected_experts = torch.full((rows,), -1, dtype=torch.int64)
    expected_slots = torch.full((rows,), -1, dtype=torch.int64)
    for source, destination in enumerate(expanded.tolist()):
        if destination >= 0:
            expected_experts[destination] = int(experts.flatten()[source]) - expert_start
            expected_slots[destination] = slots[source // top_k]
    npu_inputs = (expanded.to(index_dtype).npu(), experts.to(expert_dtype).npu(), slots.npu())
    actual = torch.ops._C_ascend.moe_lora_recover_ep(*npu_inputs, top_k, expert_start, num_local_experts)
    assert torch.equal(actual[0].cpu(), expected_experts)
    assert torch.equal(actual[1].cpu(), expected_slots)


@pytest.mark.parametrize("expert_start,num_local_experts", [(0, 32), (96, 32), (16, 16)])
@torch.inference_mode()
def test_moe_lora_recover_ep_graph_replay_updates_mapping(expert_start, num_local_experts):
    tokens, top_k = 8, 6
    rows = tokens * top_k
    expanded = torch.empty(rows, dtype=torch.int32, device="npu")
    experts = torch.empty((tokens, top_k), dtype=torch.int32, device="npu")
    slots = torch.empty(tokens, dtype=torch.int64, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        local_experts, local_slots = torch.ops._C_ascend.moe_lora_recover_ep(
            expanded, experts, slots, top_k, expert_start, num_local_experts
        )
    for seed in (0, 1, 2):
        cpu_experts = (
            torch.arange(rows, dtype=torch.int32).roll(seed * 3).remainder(num_local_experts * 2)
            + expert_start
            - num_local_experts // 2
        ).view(tokens, top_k)
        cpu_expanded = torch.full((rows,), -1, dtype=torch.int32)
        valid_sources = [
            i
            for i, expert in enumerate(cpu_experts.flatten().tolist())
            if expert_start <= expert < expert_start + num_local_experts
        ]
        for destination, source in enumerate(reversed(valid_sources)):
            cpu_expanded[source] = destination
        cpu_slots = torch.arange(tokens, dtype=torch.int64).roll(seed).remainder(3) - 1
        expanded.copy_(cpu_expanded)
        experts.copy_(cpu_experts)
        slots.copy_(cpu_slots)
        graph.replay()
        expected_experts = torch.full((rows,), -1, dtype=torch.int64)
        expected_slots = torch.full((rows,), -1, dtype=torch.int64)
        for source, destination in enumerate(cpu_expanded.tolist()):
            if destination >= 0:
                expected_experts[destination] = int(cpu_experts.flatten()[source]) - expert_start
                expected_slots[destination] = cpu_slots[source // top_k]
        assert torch.equal(local_experts.cpu(), expected_experts)
        assert torch.equal(local_slots.cpu(), expected_slots)


@pytest.mark.parametrize("num_experts,top_k", [(256, 6), (128, 8)])
@pytest.mark.parametrize("ep_size,ep_rank", [(2, 0), (2, 1), (8, 0), (8, 3), (8, 7)])
@torch.inference_mode()
def test_moe_lora_recover_ep_with_npu_routing(ep_size, ep_rank, num_experts, top_k):
    tokens = 8
    num_local = num_experts // ep_size
    first = ep_rank * num_local
    experts = torch.tensor(
        [[(token * 17 + choice * 37) % num_experts for choice in range(top_k)] for token in range(tokens)],
        dtype=torch.int32,
    )
    # Ensure both local and remote rows even at the final EP rank.
    experts[:, 0] = first + torch.arange(tokens, dtype=torch.int32) % num_local
    x = torch.randn(tokens, 32, dtype=torch.bfloat16, device="npu")
    _, expanded, counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        experts.npu(),
        active_num=tokens * top_k,
        expert_num=num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[first, first + num_local],
        quant_mode=-1,
    )
    slots = torch.tensor([0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.int64)
    context = SimpleNamespace(
        use_ep=True,
        top_k=top_k,
        punica_wrapper=SimpleNamespace(token_lora_indices=slots.npu()),
    )
    local_experts, local_slots = _recover_moe_lora_routing_allgather(
        context, expanded, experts.npu(), expert_start=first, num_local_experts=num_local
    )
    with patch("vllm_ascend.lora.fused_moe._MOE_LORA_EP_RECOVER_FUSED", False):
        fallback_experts, fallback_slots = _recover_moe_lora_routing_allgather(
            context, expanded, experts.npu(), expert_start=first, num_local_experts=num_local
        )
    expected_experts = torch.full((tokens * top_k,), -1, dtype=torch.int64)
    expected_slots = torch.full_like(expected_experts, -1)
    for source, destination in enumerate(expanded.cpu().tolist()):
        if destination >= 0:
            expected_experts[destination] = int(experts.flatten()[source]) - first
            expected_slots[destination] = slots[source // top_k]
    assert int(counts.sum().item()) == int((expanded >= 0).sum().item())
    assert torch.equal(local_experts.cpu(), expected_experts)
    assert torch.equal(local_slots.cpu(), expected_slots)
    assert torch.equal(fallback_experts.cpu(), expected_experts)
    assert torch.equal(fallback_slots.cpu(), expected_slots)


@pytest.mark.parametrize("invalid", ["top-k", "dtype", "slots-empty", "noncontiguous", "rows"])
def test_moe_lora_recover_rejects_invalid_metadata(invalid):
    expanded = torch.empty(8, dtype=torch.int32, device="meta")
    experts = torch.empty((2, 4), dtype=torch.int64, device="meta")
    slots = torch.empty(2, dtype=torch.int64, device="meta")
    top_k = 4
    if invalid == "top-k":
        top_k = 0
    elif invalid == "dtype":
        slots = slots.float()
    elif invalid == "slots-empty":
        slots = slots[:0]
    elif invalid == "noncontiguous":
        expanded = torch.empty(16, dtype=torch.int32, device="meta")[::2]
    elif invalid == "rows":
        experts = experts[:1]
    with pytest.raises(RuntimeError, match="moe_lora_recover"):
        torch.ops._C_ascend.moe_lora_recover(expanded, experts, slots, top_k)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens,expected_calls", [(113, 1), (114, 1)])
@torch.inference_mode()
def test_moe_lora_recover_helper_cost_boundary(index_dtype, expert_dtype, tokens, expected_calls):
    cpu_inputs = make_inputs(tokens, 8, 7, index_dtype, expert_dtype)
    inputs = tuple(value.npu() for value in cpu_inputs)
    context = SimpleNamespace(top_k=8, use_ep=False, punica_wrapper=SimpleNamespace(token_lora_indices=inputs[2]))
    original_op = torch.ops._C_ascend.moe_lora_recover
    with patch.object(torch.ops._C_ascend, "moe_lora_recover", wraps=original_op) as fused:
        outputs = _recover_moe_lora_routing_allgather(context, inputs[0], inputs[1])
        assert fused.call_count == expected_calls
    assert_outputs(outputs, inputs, cpu_inputs, 8)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided_input", [0, 1, 2])
@torch.inference_mode()
def test_moe_lora_recover_materializes_strided_graph_inputs(index_dtype, expert_dtype, strided_input):
    top_k = 5
    cpu_inputs = make_inputs(9, top_k, 7, index_dtype, expert_dtype)
    inputs = [value.npu() for value in cpu_inputs]
    source = cpu_inputs[strided_input]
    holder = torch.full((source.numel() * 2 + 32,), -73, dtype=source.dtype, device="npu")
    inputs[strided_input] = holder[16 : 16 + source.numel() * 2 : 2].view(source.shape)
    inputs[strided_input].copy_(source)
    assert not inputs[strided_input].is_contiguous()
    context = SimpleNamespace(top_k=top_k, use_ep=False, punica_wrapper=SimpleNamespace(token_lora_indices=inputs[2]))
    original_op = torch.ops._C_ascend.moe_lora_recover
    with patch.object(torch.ops._C_ascend, "moe_lora_recover", wraps=original_op) as wrapped:
        outputs = _recover_moe_lora_routing_allgather(context, inputs[0], inputs[1])
        assert wrapped.call_count == 1
        assert all(value.is_contiguous() for value in wrapped.call_args.args[:3])
    assert_outputs(outputs, inputs, cpu_inputs, top_k)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = _recover_moe_lora_routing_allgather(context, inputs[0], inputs[1])
    for seed in (1, 2, 0):
        current = make_inputs(9, top_k, 7, index_dtype, expert_dtype, seed)
        for destination, value in zip(inputs, current):
            destination.copy_(value)
        before = holder.cpu()
        graph.replay()
        assert_outputs(outputs, inputs, current, top_k)
        assert torch.equal(holder.cpu(), before)
