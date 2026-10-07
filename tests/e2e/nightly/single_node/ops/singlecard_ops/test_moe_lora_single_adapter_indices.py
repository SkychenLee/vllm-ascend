# SPDX-License-Identifier: Apache-2.0
"""Run only after an owner explicitly assigns ASCEND_RT_VISIBLE_DEVICES.

No adapter identity may be frozen by graph capture. The operation also handles
mixed adapters and inactive padding with the generic device-side calculation.
"""

import pytest
import torch
import torch_npu


def make_inputs(tokens, top_k, index_dtype, expert_dtype, pattern="single-0", seed=0, short_slots=False):
    generator = torch.Generator().manual_seed(seed)
    rows = tokens * top_k
    expanded = torch.randperm(rows, generator=generator).to(index_dtype)
    expanded[seed % 2 :: 2].neg_()
    experts = torch.randint(0, 256, (tokens, top_k), generator=generator, dtype=expert_dtype)
    slot_count = min(tokens, 3) if short_slots else tokens
    slots = torch.zeros(slot_count, dtype=torch.int64)
    enabled = torch.tensor([1, -7, 1], dtype=torch.int32)
    if pattern.startswith("single-"):
        slots.fill_(int(pattern[-1]))
    elif pattern == "negative-padding":
        slots.fill_(1)
        slots[1::3] = -1
    elif pattern == "mixed":
        slots.copy_(torch.arange(slot_count, dtype=torch.int64) % 3)
    elif pattern == "disabled":
        slots.fill_(1)
        enabled[1] = 0
    elif pattern == "all-negative":
        slots.fill_(-(2**63))
    elif pattern == "invalid-positive":
        slots.fill_(3)
        slots[1::2] = 2**63 - 1
    else:
        raise ValueError(pattern)
    return expanded, experts, slots, enabled


def integer_reference(inputs, top_k, num_experts=256):
    """Independent inverse permutation, with explicit INT64 wrapping."""
    expanded, experts, slots, enabled = (value.tolist() for value in (inputs[0], inputs[1].flatten(), *inputs[2:]))
    result = [-1] * len(expanded)
    for original, destination in enumerate(expanded):
        slot = slots[min(original // top_k, len(slots) - 1)]
        if 0 <= slot < len(enabled) and enabled[slot] != 0:
            bits = (slot * num_experts + experts[original]) & ((1 << 64) - 1)
            result[abs(destination)] = bits if bits < 1 << 63 else bits - (1 << 64)
    return torch.tensor(result, dtype=torch.int64)


def old_native_chain(inputs, top_k, num_experts=256):
    expanded, experts, slots, enabled = inputs
    recovered_experts, recovered_slots = torch.ops._C_ascend.moe_lora_recover(expanded, experts, slots, top_k)
    return torch.ops._C_ascend.moe_lora_combined_indices(recovered_experts, recovered_slots, enabled, num_experts)


def assert_output(output, inputs, cpu_inputs, top_k, num_experts=256):
    assert output.shape == inputs[0].shape
    assert output.dtype == torch.int64
    assert output.device == inputs[0].device
    assert output.is_contiguous()
    assert torch.equal(output.cpu(), integer_reference(cpu_inputs, top_k, num_experts))
    assert torch.equal(output, old_native_chain(inputs, top_k, num_experts))
    for tensor, snapshot in zip(inputs, cpu_inputs):
        assert torch.equal(tensor.cpu(), snapshot)
        assert not torch._C._is_alias_of(output, tensor)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "tokens,top_k",
    [(0, 6), (1, 1), (6, 6), (8, 6), (24, 6), (48, 6), (64, 6), (146, 6), (8161, 1), (1366, 6), (16384, 6)],
)
@pytest.mark.parametrize(
    "pattern",
    ["single-0", "single-1", "single-2", "negative-padding", "mixed", "disabled", "all-negative", "invalid-positive"],
)
@torch.inference_mode()
def test_single_adapter_exact_metadata(index_dtype, expert_dtype, tokens, top_k, pattern):
    cpu_inputs = make_inputs(tokens, top_k, index_dtype, expert_dtype, pattern)
    inputs = tuple(value.npu() for value in cpu_inputs)
    output = torch.ops._C_ascend.moe_lora_single_adapter_indices(*inputs, top_k, 256)
    assert_output(output, inputs, cpu_inputs, top_k)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens", [6, 64, 1366, 16384])
@torch.inference_mode()
def test_single_adapter_short_slots_and_unused_tail(index_dtype, expert_dtype, tokens):
    for short_slots in (False, True):
        cpu_inputs = make_inputs(tokens, 6, index_dtype, expert_dtype, "mixed", short_slots=short_slots)
        if not short_slots:
            cpu_inputs = (*cpu_inputs[:2], torch.cat((cpu_inputs[2], torch.tensor([2, -1, 1]))), cpu_inputs[3])
        inputs = tuple(value.npu() for value in cpu_inputs)
        output = torch.ops._C_ascend.moe_lora_single_adapter_indices(*inputs, 6, 256)
        assert_output(output, inputs, cpu_inputs, 6)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens", [6, 48, 1366, 16384])
@torch.inference_mode()
def test_single_adapter_reads_slot_and_enable_updates_on_graph_replay(index_dtype, expert_dtype, tokens):
    cpu_inputs = make_inputs(tokens, 6, index_dtype, expert_dtype)
    inputs = tuple(value.npu() for value in cpu_inputs)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            torch.ops._C_ascend.moe_lora_single_adapter_indices(*inputs, 6, 256)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = torch.ops._C_ascend.moe_lora_single_adapter_indices(*inputs, 6, 256)
    # Change slot, disable it, re-enable it, then include mixed/padded rows.
    for seed, pattern in enumerate(("single-1", "disabled", "single-2", "mixed", "negative-padding", "single-0"), 1):
        current = make_inputs(tokens, 6, index_dtype, expert_dtype, pattern, seed)
        for destination, source in zip(inputs, current):
            destination.copy_(source)
        graph.replay()
        assert_output(output, inputs, current, 6)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expert_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_single_adapter_preserves_integer_expert_bits_and_wrap(index_dtype, expert_dtype):
    top_k = 6
    expanded = torch.tensor([5, -2, 0, -3, 1, -4], dtype=index_dtype)
    limits = torch.iinfo(expert_dtype)
    large = 2 ** (limits.bits - 4)
    experts = torch.tensor([limits.max, limits.min, large + 3, -large + 5, -1, 0], dtype=expert_dtype).reshape(1, 6)
    slots = torch.tensor([2], dtype=torch.int64)
    enabled = torch.tensor([0, 1, -2], dtype=torch.int32)
    cpu_inputs = expanded, experts, slots, enabled
    inputs = tuple(value.npu() for value in cpu_inputs)
    for count in (256, 2**62 + 1):
        output = torch.ops._C_ascend.moe_lora_single_adapter_indices(*inputs, top_k, count)
        assert_output(output, inputs, cpu_inputs, top_k, count)


@pytest.mark.parametrize("tokens,top_k", [(0, 6), (6, 6), (1366, 6), (16384, 6)])
def test_single_adapter_meta(tokens, top_k):
    inputs = (
        torch.empty(tokens * top_k, dtype=torch.int32, device="meta"),
        torch.empty((tokens, top_k), dtype=torch.int64, device="meta"),
        torch.empty(tokens, dtype=torch.int64, device="meta"),
        torch.empty(3, dtype=torch.int32, device="meta"),
    )
    output = torch.ops._C_ascend.moe_lora_single_adapter_indices(*inputs, top_k, 256)
    assert output.shape == inputs[0].shape and output.dtype == torch.int64 and output.device.type == "meta"
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::moe_lora_single_adapter_indices", "PrivateUse1")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::moe_lora_single_adapter_indices", "Meta")


@pytest.mark.parametrize(
    "invalid",
    [
        "top-k",
        "rows",
        "index-dtype",
        "expert-dtype",
        "slot-dtype",
        "mask-dtype",
        "slots-empty",
        "mask-empty",
        "mask-too-long",
        "noncontiguous",
        "num-experts",
        "unsupported-middle",
        "over-limit",
    ],
)
def test_single_adapter_rejects_invalid_meta(invalid):
    inputs = list(make_inputs(6, 6, torch.int32, torch.int64))
    top_k, count = 6, 256
    if invalid == "top-k":
        top_k = 0
    elif invalid == "rows":
        inputs[0] = inputs[0][:-1]
    elif invalid.endswith("-dtype"):
        index = {"index-dtype": 0, "expert-dtype": 1, "slot-dtype": 2, "mask-dtype": 3}[invalid]
        inputs[index] = inputs[index].float()
    elif invalid == "slots-empty":
        inputs[2] = inputs[2][:0]
    elif invalid == "mask-empty":
        inputs[3] = inputs[3][:0]
    elif invalid == "mask-too-long":
        inputs[3] = torch.ones(65, dtype=torch.int32)
    elif invalid == "noncontiguous":
        inputs[0] = torch.empty(72, dtype=torch.int32)[::2]
    elif invalid == "num-experts":
        count = 0
    elif invalid == "unsupported-middle":
        inputs = list(make_inputs(147, 6, torch.int32, torch.int64))
    elif invalid == "over-limit":
        inputs = list(make_inputs(16385, 6, torch.int32, torch.int64))
    inputs = [
        torch.empty(value.shape, device="meta", dtype=value.dtype)
        if value.is_contiguous()
        else torch.empty(72, device="meta", dtype=value.dtype)[::2]
        for value in inputs
    ]
    with pytest.raises(RuntimeError, match="moe_lora_single_adapter_indices"):
        torch.ops._C_ascend.moe_lora_single_adapter_indices(*inputs, top_k, count)


@torch.inference_mode()
def test_single_adapter_rejects_blocked_storage():
    cpu_inputs = make_inputs(6, 6, torch.int32, torch.int32)
    inputs = [value.npu() for value in cpu_inputs]
    # FRACTAL_NZ has physical padding despite a contiguous logical view.
    allow_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    assert isinstance(allow_internal_format, bytes), "cannot preserve the internal-format option state"
    try:
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": "enable"})
        inputs[1] = torch_npu.npu_format_cast(inputs[1], 29)
    finally:
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": allow_internal_format.decode()})
    assert torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == allow_internal_format
    assert torch_npu.get_npu_format(inputs[1]) == 29
    with pytest.raises(RuntimeError, match="moe_lora_single_adapter_indices"):
        torch.ops._C_ascend.moe_lora_single_adapter_indices(*inputs, 6, 256)
