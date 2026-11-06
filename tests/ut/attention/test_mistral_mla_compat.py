# SPDX-License-Identifier: Apache-2.0
"""CPU numerical contracts for Mistral's padded MLA path.

These tests execute production methods with only the NPU kernel boundary
substituted. Run without the NPU-wide conftest using ``pytest --noconftest``.
They do not establish CANN kernel or NPU graph-replay support.
"""

import ast
import importlib.util
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def _load_production_nodes(relative_path, names, namespace, methods=None):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            if methods and isinstance(node, ast.ClassDef) and node.name in methods:
                node.bases = []
                node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods[node.name]]
            selected.append(node)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def production():
    spec = importlib.util.spec_from_file_location("mla_compat_cpu", ROOT / "vllm_ascend/attention/mla_compat.py")
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    namespace = dict(
        torch=torch,
        F=F,
        NamedTuple=NamedTuple,
        pad_mla_head=compat.pad_mla_head,
        scale_mla_query=compat.scale_mla_query,
        MLAPO_MAX_SUPPORTED_TOKENS=100,
        _EXTRA_CTX=SimpleNamespace(num_tokens=3, capturing=False, is_draft_model=False),
        maybe_save_kv_layer_to_connector=lambda *args: None,
        get_graph_params=lambda: None,
        AscendAttentionState=SimpleNamespace(SpecDecoding=0, ChunkedPrefill=1, DecodeOnly=2, PrefillNoCache=3),
    )
    return _load_production_nodes(
        "vllm_ascend/attention/mla_v1.py",
        {"AscendMLAImpl", "DecodeMLAPreprocessResult", "PrefillMLAPreprocessResult"},
        namespace,
        methods={
            "AscendMLAImpl": {
                "_validate_mla_compatibility",
                "_pad_prefill_qk",
                "_exec_kv_unfused",
                "_forward_decode",
                "_v_up_proj",
                "_q_proj_and_k_up_proj",
                "forward",
                "_compute_prefill_context",
            }
        },
    )


def _impl(production):
    impl = production["AscendMLAImpl"]()
    impl.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
    )
    impl.kv_lora_rank = 256
    impl.kv_cache_lora_rank = 512
    impl.qk_nope_head_dim = 64
    impl.qk_rope_head_dim = 64
    impl.v_head_dim = 128
    impl.num_heads = impl.num_heads_padded = 2
    impl.num_kv_heads = 1
    impl.head_padding = 0
    impl.enable_kv_nz = impl.fa_quant_layer = impl.enable_mlapo = False
    return impl


@pytest.mark.parametrize("semantic,physical", [(64, 128), (256, 512)])
def test_zero_padding_preserves_attention_numerically(production, semantic, physical):
    torch.manual_seed(8)
    q, k, value = torch.randn(3, 2, semantic), torch.randn(7, 2, semantic), torch.randn(7, 2, 128)
    qr, kr = torch.randn(3, 2, 64), torch.randn(7, 2, 64)

    def attention(query, key):
        scores = (torch.einsum("thd,shd->hts", query, key) + torch.einsum("thd,shd->hts", qr, kr)) / 128**0.5
        return torch.einsum("hts,shv->thv", scores.softmax(-1), value)

    pad = production["pad_mla_head"]
    torch.testing.assert_close(attention(pad(q, semantic, physical), pad(k, semantic, physical)), attention(q, k))
    padded = pad(q, semantic, physical)
    assert pad(padded, semantic, physical) is padded
    assert torch.count_nonzero(padded[..., semantic:]) == 0


@pytest.mark.parametrize("num_decode", [0, 1, 3])
def test_forward_scales_both_query_parts_with_current_positions(production, num_decode):
    impl = _impl(production)
    impl.use_output_gate = impl.is_pcp_decode_sharded = False
    impl.get_num_actual_tokens = lambda metadata: 3
    impl._decode_requires_current_kv = lambda metadata: False
    impl.o_proj = lambda x, **kwargs: (x,)
    decode_cls, prefill_cls = production["DecodeMLAPreprocessResult"], production["PrefillMLAPreprocessResult"]
    decode = decode_cls(torch.ones(num_decode, 2, 256), torch.full((num_decode, 2, 64), 2.0)) if num_decode else None
    prefill = (
        prefill_cls(torch.ones(3 - num_decode, 2, 64), torch.full((3 - num_decode, 2, 64), 2.0))
        if num_decode < 3
        else None
    )

    def preprocess(*args, llama_4_scaling=None):
        # Decode NoPE is scaled before K projection; its production projection
        # and BF16 rounding are checked separately below.
        scaled = (
            decode._replace(ql_nope=production["scale_mla_query"](decode.ql_nope, llama_4_scaling)) if decode else None
        )
        return scaled, prefill

    impl._mla_preprocess = preprocess
    seen = []

    def observe(q_nope, q_pe):
        seen.append((q_nope.clone(), q_pe.clone()))
        return (q_nope.mean((1, 2)) + q_pe.mean((1, 2))).unsqueeze(-1).expand(-1, 256)

    impl._forward_decode = lambda result, *args: observe(result.ql_nope, result.q_pe)
    impl._forward_prefill = lambda q, qr, *args: observe(q, qr)
    meta = SimpleNamespace(
        num_decode_tokens=num_decode, num_decodes=int(num_decode > 0), num_prefills=int(num_decode < 3)
    )
    positions = torch.tensor([8191, 8192, 16384])
    factors = (1 + 0.1 * torch.log1p(positions // 8192)).view(3, 1, 1)
    hidden, output = torch.zeros(3, 256), torch.empty(3, 256)
    impl.forward("layer", hidden, (torch.empty(1, 4, 1, 512),), meta, output, llama_4_scaling=factors)
    torch.testing.assert_close(output[:, 0], 3 * factors.flatten())
    assert len(seen) == int(num_decode > 0) + int(num_decode < 3)
    # Reuse the same tensor allocation with changed values, as replay inputs do.
    factors.mul_(2)
    impl.forward("layer", hidden, (torch.empty(1, 4, 1, 512),), meta, output, llama_4_scaling=factors)
    torch.testing.assert_close(output[:, 0], 3 * factors.flatten())


def test_query_scaling_is_dynamic_in_cpu_graph(production):
    scale = torch.ones(3, 1, 1)
    query = torch.randn(3, 2, 64, dtype=torch.bfloat16)
    compiled = torch.compile(production["scale_mla_query"], backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(query, scale), query)
    scale.copy_((1 + 0.1 * torch.log1p(torch.tensor([8191, 8192, 16384]) // 8192)).view(3, 1, 1))
    expected = query.clone()
    expected *= scale
    torch.testing.assert_close(compiled(query, scale), expected, rtol=0, atol=0)


def test_unfused_kv_normalizes_semantic_rank_then_writes_padded_cache(production):
    torch.manual_seed(4)
    impl = _impl(production)
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.randn(256), variance_epsilon=1e-5)
    raw = torch.randn(2, 1, 1, 320)
    cos, sin = torch.randn(2, 1, 1, 64), torch.randn(2, 1, 1, 64)
    caches = (torch.zeros(1, 4, 1, 512), torch.zeros(1, 4, 1, 64))
    slots = torch.tensor([2, 0])

    def rms_norm(x, weight, epsilon):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight, None

    def rope(x, cos, sin):
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
        return x * cos + rotated * sin

    def write_cache(*, key, value, key_cache, value_cache, slot_mapping):
        key_cache.flatten(0, 1)[slot_mapping.long()] = key
        value_cache.flatten(0, 1)[slot_mapping.long()] = value

    production["torch_npu"] = SimpleNamespace(npu_rms_norm=rms_norm, npu_interleave_rope=rope)
    production["DeviceOperator"] = SimpleNamespace(reshape_and_cache=write_cache)
    kr, latent = impl._exec_kv_unfused(raw, cos, sin, caches, slots)
    expected, _ = rms_norm(raw[..., :256], impl.kv_a_layernorm.weight, 1e-5)
    torch.testing.assert_close(latent, expected)
    torch.testing.assert_close(kr, rope(raw[..., 256:], cos, sin))
    torch.testing.assert_close(caches[0].flatten(0, 1)[slots, :, :256], expected.view(2, 1, 256))
    assert torch.count_nonzero(caches[0][..., 256:]) == 0
    torch.testing.assert_close(caches[1].flatten(0, 1)[slots], kr.view(2, 1, 64))


@pytest.mark.parametrize("speculative", [False, True])
def test_decode_uses_512_kernel_and_256_value_projection(production, speculative):
    torch.manual_seed(5)
    impl = _impl(production)
    impl.speculative_config = object() if speculative else None
    impl.scale = 128**-0.5
    impl.W_UV = torch.randn(2, 256, 128)
    q, qr = torch.randn(3, 2, 256), torch.randn(3, 2, 64)
    k, kr = torch.randn(5, 1, 256), torch.randn(5, 1, 64)
    padded = production["pad_mla_head"](k, 256, 512).view(1, 5, 1, 512)

    def attention(query, key, value, **kwargs):
        assert query.shape[-1] == key.shape[-1] == value.shape[-1] == 512
        query, key = query.reshape(3, 2, 512), key.reshape(5, 1, 512)
        query_rope, key_rope = kwargs["query_rope"].reshape(3, 2, 64), kwargs["key_rope"].reshape(5, 1, 64)
        scores = (
            torch.einsum("thd,shd->hts", query, key.expand(-1, 2, -1))
            + torch.einsum("thd,shd->hts", query_rope, key_rope.expand(-1, 2, -1))
        ) * kwargs["softmax_scale"]
        result = torch.einsum("hts,shd->htd", scores.softmax(-1), value.reshape(5, 1, 512).expand(-1, 2, -1))
        return (result if speculative else result.unsqueeze(2)), None

    def batchmatmul(x, weight, *, perm_y):
        assert x.shape[-1] == 256 and x.is_contiguous()
        return torch.bmm(x, weight).permute(perm_y)

    production["torch_npu"] = SimpleNamespace(
        npu_fused_infer_attention_score_v2=attention, npu_transpose_batchmatmul=batchmatmul
    )
    meta = SimpleNamespace(
        attn_state=1,
        causal=True,
        decode=SimpleNamespace(
            attn_mask=None, actual_seq_lengths_q=[3], seq_lens_list=[5], block_table=torch.tensor([[0]])
        ),
    )
    result = impl._forward_decode(production["DecodeMLAPreprocessResult"](q, qr, padded, kr.view(1, 5, 1, 64)), 5, meta)
    scores = (
        torch.einsum("thd,shd->hts", q, k.expand(-1, 2, -1)) + torch.einsum("thd,shd->hts", qr, kr.expand(-1, 2, -1))
    ) * impl.scale
    latent = torch.einsum("hts,shd->htd", scores.softmax(-1), k.expand(-1, 2, -1))
    expected = torch.bmm(latent, impl.W_UV).transpose(0, 1).reshape(3, 256)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("enable_kv_nz", True),
        ("fa_quant_layer", True),
    ],
)
def test_unsupported_layout_fails_before_execution(production, field, value):
    impl = _impl(production)
    target = impl.vllm_config.parallel_config if field.endswith("parallel_size") else impl
    setattr(target, field, value)
    with pytest.raises(NotImplementedError):
        impl._validate_mla_compatibility()


def test_prefix_context_slices_physical_cache_before_projection(production):
    torch.manual_seed(6)
    impl = _impl(production)
    impl.prefill_qk_head_dim = 128
    impl.scale = 128**-0.5
    impl.get_context_block_table = lambda _: None
    impl.get_context_seq_len_npu = lambda *args: None
    impl._reorg_kvcache = lambda k, kr, **kwargs: (k, kr)
    weight = torch.randn(256, 2 * (64 + 128)) / 256**0.5

    def projection(x):
        assert x.shape[-1] == 256
        return (x @ weight,)

    impl.kv_b_proj = projection
    latent = torch.randn(5, 1, 256)
    cache = (production["pad_mla_head"](latent, 256, 512).view(1, 5, 1, 512), torch.randn(1, 5, 1, 64))
    q, qr = torch.randn(3, 2, 64), torch.randn(3, 2, 64)
    prefix_output, prefix_lse = torch.randn(3, 2, 128), torch.randn(2, 3)
    meta = SimpleNamespace(
        prefill=SimpleNamespace(
            actual_seq_lengths_q=[3],
            chunked_context=SimpleNamespace(seq_tot=[5], starts=[0], chunk_actual_seq_lengths_kv_list=[[5]]),
        )
    )

    def load_cache(k, kr, *args, key, value):
        key.copy_(k.flatten(0, 1))
        value.copy_(kr.flatten(0, 1))

    def attention(query, key, value, **kwargs):
        assert query.shape[-1] == key.shape[-1] == 128
        scores = (
            torch.einsum("thd,shd->hts", query, key)
            + torch.einsum("thd,shd->hts", kwargs["query_rope"], kwargs["key_rope"])
        ) * kwargs["scale"]
        return torch.einsum("hts,shv->thv", scores.softmax(-1), value), scores.logsumexp(-1)

    def update(lses, outs, _):
        weights = torch.stack(lses).softmax(0)
        return (torch.stack(outs) * weights.unsqueeze(-1)).sum(0), None

    production["DeviceOperator"] = SimpleNamespace(kv_cache_load=load_cache)
    production["torch_npu"] = SimpleNamespace(npu_fused_infer_attention_score=attention, npu_attention_update=update)
    result, _ = impl._compute_prefill_context(q, qr, cache, 64, meta, prefix_output, prefix_lse)
    key, value = (latent.squeeze(1) @ weight).view(5, 2, 192).split([64, 128], -1)
    scores = (
        torch.einsum("thd,shd->hts", q, key)
        + torch.einsum("thd,shd->hts", qr, cache[1].view(5, 1, 64).expand(-1, 2, -1))
    ) * impl.scale
    chunk = torch.einsum("hts,shv->thv", scores.softmax(-1), value)
    probabilities = torch.stack((prefix_lse, scores.logsumexp(-1))).softmax(0).permute(0, 2, 1).unsqueeze(-1)
    expected = (torch.stack((prefix_output, chunk)) * probabilities).sum(0)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_eagle_markers_and_both_runner_cache_dimensions(production):
    @dataclass
    class CacheSpec:
        head_size: int = 320

    class Layer:
        kv_lora_rank = 256
        qk_rope_head_dim = 64
        qk_nope_head_dim = 64
        v_head_dim = 128

    layer = Layer()
    layer.impl = _impl(production)
    del layer.impl.kv_cache_lora_rank
    layer.impl.enable_mlapo = True
    namespace = dict(
        replace=replace,
        AscendMLAAttention=Layer,
        AscendMLAImpl=production["AscendMLAImpl"],
        current_platform=SimpleNamespace(device_name="npu"),
        MISTRAL_KV_RANK=256,
        MISTRAL_ROPE_DIM=64,
        ASCEND_MLA_CACHE_RANK=512,
        _ORIGINAL_EAGLE_INIT=lambda *args, **kwargs: None,
        _ORIGINAL_GET_CACHE_SPEC=lambda *args: CacheSpec(),
    )
    _load_production_nodes(
        "vllm_ascend/patch/worker/patch_mistral4_mla.py", {"_patched_eagle_init", "_patched_get_cache_spec"}, namespace
    )
    config = SimpleNamespace(kv_transfer_config=None)
    namespace["_patched_eagle_init"](
        SimpleNamespace(model=SimpleNamespace(modules=lambda: [layer])), vllm_config=config
    )
    assert layer.kv_lora_rank == 256
    assert layer.impl.kv_cache_lora_rank == 512 and layer.impl.prefill_qk_head_dim == 128
    assert not layer.impl.enable_mlapo
    spec = namespace["_patched_get_cache_spec"](layer, config)
    assert spec.head_size == 576
    namespace.update(
        AscendMLAAttentionSpec=CacheSpec,
        MLAAttention=Layer,
        AttentionLayerBase=object,
        get_layers_from_vllm_config=lambda *args: {"layer": layer},
        get_current_vllm_config=lambda: config,
    )
    _load_production_nodes(
        "vllm_ascend/worker/model_runner_v1.py",
        {"NPUModelRunner"},
        namespace,
        methods={"NPUModelRunner": {"_get_attention_kv_cache_dims"}},
    )
    runner = namespace["NPUModelRunner"]()
    runner.vllm_config = config
    assert runner._get_attention_kv_cache_dims("layer", spec) == (512, 64)
    _load_production_nodes("vllm_ascend/worker/v2/attn_utils.py", {"_get_attention_kv_cache_dims"}, namespace)
    assert namespace["_get_attention_kv_cache_dims"]("layer", spec) == (512, 64)


def test_wrapper_third_argument_reaches_backend_as_scaling():
    captured = {}

    def backend(*args, **kwargs):
        captured.update(kwargs)
        args[4].zero_()

    attention = SimpleNamespace(layer_name="layer.attn", kv_cache=(), impl=SimpleNamespace(forward=backend))
    context = SimpleNamespace(
        no_compile_layers={"layer": SimpleNamespace(mla_attn=attention)}, attn_metadata={"layer.attn": None}
    )
    namespace = dict(torch=SimpleNamespace(empty=torch.empty), get_forward_context=lambda: context)
    _load_production_nodes(
        "vllm_ascend/ops/mla.py",
        {"AscendMultiHeadLatentAttention", "mla_forward"},
        namespace,
        methods={"AscendMultiHeadLatentAttention": {"forward"}},
    )
    namespace["torch"].ops = SimpleNamespace(vllm=SimpleNamespace(mla_forward=namespace["mla_forward"]))
    wrapper = namespace["AscendMultiHeadLatentAttention"]()
    wrapper.output_token_shard_size = 1
    wrapper.hidden_size = 16
    wrapper.prefix = "layer"
    scaling = torch.tensor([1, 1.1, 1.2]).view(3, 1, 1)
    output = wrapper.forward(torch.arange(3), torch.ones(3, 16), scaling)
    assert captured["llama_4_scaling"] is scaling
    assert output.shape == (3, 16)


def test_decode_scales_before_absorbed_projection_with_upstream_bf16_rounding(production):
    torch.manual_seed(10)
    impl = _impl(production)
    impl.qk_head_dim = 128
    projected = torch.randn(3, 2, 128, dtype=torch.bfloat16)
    impl.q_proj = lambda _: (projected.flatten(1),)
    impl.W_UK_T = torch.randn(2, 64, 256, dtype=torch.bfloat16)
    scaling = (1 + 0.1 * torch.log1p(torch.tensor([8191, 8192, 16384]) // 8192)).view(3, 1, 1)
    actual, rope_query = impl._q_proj_and_k_up_proj(torch.empty(3, 16), scaling)
    semantic_query = projected[..., :64].clone()
    semantic_query *= scaling
    expected = torch.bmm(semantic_query.transpose(0, 1), impl.W_UK_T).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(rope_query, projected[..., 64:], rtol=0, atol=0)


@pytest.mark.parametrize("attention_factor", [None, 1.0, 0.75])
def test_ascend_yarn_attention_factor_overrides_legacy_mscale(attention_factor):
    class RotaryBase:
        def __init__(self, *args):
            pass

    class DeepseekBase(RotaryBase):
        pass

    namespace = dict(
        torch=torch,
        math=math,
        DeepseekScalingRotaryEmbedding=DeepseekBase,
        NPUPlatform=SimpleNamespace(device_type="cpu"),
    )
    _load_production_nodes(
        "vllm_ascend/ops/rotary_embedding.py",
        {"AscendDeepseekScalingRotaryEmbedding", "_uses_mla_rope_buffer"},
        namespace,
    )
    cls = namespace["AscendDeepseekScalingRotaryEmbedding"]
    cls._set_cos_sin_cache = lambda *args, **kwargs: None
    rope = cls(
        128,
        64,
        8192,
        1000000,
        False,
        128,
        torch.bfloat16,
        mscale=1,
        mscale_all_dim=0,
        attention_factor=attention_factor,
    )
    expected = 1 + 0.1 * math.log(128) if attention_factor is None else attention_factor
    assert rope.mscale == pytest.approx(expected)
    config = SimpleNamespace(
        use_mla=False,
        hf_config=SimpleNamespace(model_type="mistral3"),
        hf_text_config=SimpleNamespace(model_type="mistral4"),
    )
    assert namespace["_uses_mla_rope_buffer"](config)
    config.hf_text_config.model_type = "mistral"
    assert not namespace["_uses_mla_rope_buffer"](config)
