from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

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
        pytest.param(580, 8, 580, id="prefill"),
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
@torch.inference_mode()
def test_moe_lora_recover_changed_graph_inputs(index_dtype, expert_dtype):
    top_k = 5
    cpu_inputs = make_inputs(13, top_k, 7, index_dtype, expert_dtype)
    inputs = tuple(value.npu() for value in cpu_inputs)
    for _ in range(3):
        torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = torch.ops._C_ascend.moe_lora_recover(*inputs, top_k)
    for seed in (1, 2, 0):
        current = make_inputs(13, top_k, 7, index_dtype, expert_dtype, seed)
        for destination, source in zip(inputs, current):
            destination.copy_(source)
        graph.replay()
        assert_outputs(outputs, inputs, current, top_k)


@pytest.mark.parametrize("use_ep", [True, None, 0])
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
