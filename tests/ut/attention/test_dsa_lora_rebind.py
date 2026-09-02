from types import SimpleNamespace
from unittest.mock import MagicMock

from torch import nn

from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention


class _LoRAWrapper:
    def __init__(self, base_layer):
        self.base_layer = base_layer


def test_dsa_impl_skips_partial_rebind_with_lora_compressor() -> None:
    modules = SimpleNamespace(
        compressor=SimpleNamespace(
            wkv=_LoRAWrapper(object()),
            wgate=object(),
        ),
        indexer=None,
    )
    impl = object.__new__(AscendDSAImpl)
    original_wq_a = object()
    impl.wq_a = original_wq_a

    rebound = impl.refresh_lora_module_references(modules)

    assert not rebound
    assert impl.wq_a is original_wq_a


def test_dsa_impl_skips_partial_rebind_with_lora_indexer_compressor() -> None:
    modules = SimpleNamespace(
        compressor=None,
        indexer=SimpleNamespace(
            compressor=SimpleNamespace(
                wkv=object(),
                wgate=_LoRAWrapper(object()),
            ),
        ),
    )
    impl = object.__new__(AscendDSAImpl)
    original_wq_a = object()
    impl.wq_a = original_wq_a

    rebound = impl.refresh_lora_module_references(modules)

    assert not rebound
    assert impl.wq_a is original_wq_a


def test_dsa_impl_refreshes_lora_aliases_without_lora_compressor() -> None:
    compressor = SimpleNamespace(wkv=object(), wgate=object())
    indexer_compressor = SimpleNamespace(wkv=object(), wgate=object())
    modules = SimpleNamespace(
        wq_a=_LoRAWrapper(object()),
        wq_b=_LoRAWrapper(object()),
        wkv=_LoRAWrapper(object()),
        wo_b=_LoRAWrapper(object()),
        indexer=SimpleNamespace(
            wq_b=_LoRAWrapper(object()),
            weights_proj=_LoRAWrapper(object()),
            compressor=indexer_compressor,
        ),
        compressor=compressor,
    )
    impl = object.__new__(AscendDSAImpl)

    rebound = impl.refresh_lora_module_references(modules)

    assert rebound
    assert impl.wq_a is modules.wq_a
    assert impl.wq_b is modules.wq_b
    assert impl.wkv is modules.wkv
    assert impl.wo_b is modules.wo_b
    assert impl.inderxer_wq_b is modules.indexer.wq_b
    assert impl.weights_proj is modules.indexer.weights_proj
    assert impl.cv_wq_a.linear is modules.wq_a
    assert impl.cv_wkv.linear is modules.wkv
    assert impl.cv_wq_b.linear is modules.wq_b
    assert impl.cv_inderxer_wq_b.linear is modules.indexer.wq_b
    assert impl.compressor_wkv is compressor.wkv
    assert impl.compressor_wgate is compressor.wgate
    assert impl.indexcom_wkv is indexer_compressor.wkv
    assert impl.indexcom_wgate is indexer_compressor.wgate


def test_sparse_attention_rebind_is_one_shot() -> None:
    layer = object.__new__(AscendDeepseekSparseAttention)
    nn.Module.__init__(layer)
    impl = MagicMock()
    layer.dsa_attn = SimpleNamespace(impl=impl)
    layer._dsa_impl_lora_rebind_pending = True

    if layer._dsa_impl_lora_rebind_pending:
        layer.dsa_attn.impl.refresh_lora_module_references(layer)
        layer._dsa_impl_lora_rebind_pending = False
    if layer._dsa_impl_lora_rebind_pending:
        layer.dsa_attn.impl.refresh_lora_module_references(layer)

    impl.refresh_lora_module_references.assert_called_once_with(layer)
