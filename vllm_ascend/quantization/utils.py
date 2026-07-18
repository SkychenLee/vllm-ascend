#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import json
from pathlib import Path

import torch
from vllm import envs
from vllm.logger import logger

from vllm_ascend.utils import (
    ASCEND_QUANTIZATION_METHOD,
    COMPRESSED_TENSORS_METHOD,
    FP8_METHOD,
    AscendDeviceType,
    get_ascend_device_type,
)

HIF4_QUANT_TYPE = "W4A4_HIFP4"
MXFP4_BLOCK_SIZE = 32
_MXFP4_EPSILON = 1.17e-38
_MXFP4_MAX_NORM = 6.0
_MXFP4_MIN_EXP = 0.0
_MXFP4_SCALE_FACTOR = 2.0
_MXFP4_INV_SCALE_FACTOR = 0.5
_HIF4_BLOCK_SIZE = 64
_HIF4_SCALE_MIN_EXP = -48
_HIF4_SEPARATE_QKV_WEIGHT_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
)
_HIF4_FUSED_QKV_WEIGHT_SUFFIXES = (
    ".qkv_proj.weight",
    ".query_key_value.weight",
)


def get_model_file(
    model: str | Path,
    filename: str,
    revision: str | None = None,
) -> Path | None:
    """Get a file from local model directory or download from remote repo.

    This function handles both local paths and remote repository IDs,
    automatically downloading files from HuggingFace Hub or ModelScope
    if they are not already cached.

    Args:
        model: Local directory path or HuggingFace/ModelScope repo id.
        filename: Name of the file to retrieve (e.g., "config.json").
        revision: Optional revision (branch, tag, or commit hash) for remote repos.

    Returns:
        Path to the file if found, None otherwise.
    """
    # Check if it's a local path
    model_path = Path(model) if isinstance(model, str) else model
    if model_path.exists():
        file_path = model_path / filename
        return file_path if file_path.exists() else None

    # Remote repo: try to download from HF Hub or ModelScope
    try:
        if envs.VLLM_USE_MODELSCOPE:
            from modelscope.hub.file_download import model_file_download  # type: ignore[import-untyped]

            downloaded_path = model_file_download(
                model_id=str(model),
                file_path=filename,
                revision=revision,
            )
            return Path(downloaded_path)
        else:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=str(model),
                filename=filename,
                revision=revision,
            )
            return Path(downloaded_path)
    except Exception as e:
        logger.warning("Could not download %s from %s: %s", filename, model, e)
        return None


def detect_quantization_method(model: str, revision: str | None = None) -> str | None:
    """Auto-detect the quantization method from model files.

    This function performs a lightweight check (JSON files only — no
    .safetensors or .bin inspection) to determine which quantization
    method was used to produce the weights in *model*.

    Works with both local directories (``/path/to/model``) and remote
    repository identifiers (``org/model-name``).  For remote repos the
    lookup goes through the HuggingFace / ModelScope cache, downloading
    config files if not already cached.

    Detection priority:
        1. **ModelSlim (Ascend)** – ``quant_model_description.json`` exists.
        2. **LLM-Compressor (compressed-tensors)** – ``config.json`` contains
           a ``quantization_config`` section with
           ``"quant_method": "compressed-tensors"``.
        3. **None** – neither condition is met; the caller should fall back to
           the default (float) behaviour.

    Args:
        model: Local directory path **or** HuggingFace / ModelScope repo id.
        revision: Optional model revision (branch, tag, or commit id).

    Returns:
        ``"ascend"`` for ModelSlim models,
        ``"compressed-tensors"`` for LLM-Compressor models,
        or ``None`` if no quantization signature is found.
    """
    from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME

    # Case 1: ModelSlim — look for quant_model_description.json
    modelslim_path = get_model_file(model, MODELSLIM_CONFIG_FILENAME, revision=revision)
    if modelslim_path is not None:
        return ASCEND_QUANTIZATION_METHOD

    # Case 2: LLM-Compressor — look for compressed-tensors in config.json
    config_path = get_model_file(model, "config.json", revision=revision)
    if config_path is not None:
        try:
            with open(config_path) as f:
                config = json.load(f)
            quant_cfg = config.get("quantization_config")
            if isinstance(quant_cfg, dict):
                quant_method = quant_cfg.get("quant_method", "")
                if quant_method == COMPRESSED_TENSORS_METHOD:
                    return COMPRESSED_TENSORS_METHOD
            if isinstance(quant_cfg, dict):
                quant_method = quant_cfg.get("quant_method", "")
                if quant_method == FP8_METHOD:
                    return FP8_METHOD
        except (json.JSONDecodeError, OSError):
            pass

    # Case 3: No quantization signature found.
    return None


def maybe_auto_detect_quantization(vllm_config) -> None:
    """Auto-detect and apply the quantization method on *vllm_config*.

    This should be called during engine initialisation (from
    ``NPUPlatform.check_and_update_config``) **after** ``VllmConfig`` has been
    created but **before** heavy weights are loaded.

    Because ``check_and_update_config`` runs *after*
    ``VllmConfig.__post_init__`` has already evaluated
    ``_get_quantization_config`` (which returned ``None`` when
    ``model_config.quantization`` was not set), we must:

    1. Set ``model_config.quantization`` to the detected value.
    2. Recreate ``vllm_config.quant_config`` so that the quantization
       pipeline (``get_quant_config`` → ``QuantizationConfig`` →
       ``get_quant_method`` for every layer) is properly initialised.

    Rules:
        * If the user explicitly set ``--quantization``, that value is
          respected.  A warning is emitted when the detected method differs.
        * If no ``--quantization`` was given, the detected method (if any) is
          applied automatically.

    Args:
        vllm_config: A ``vllm.config.VllmConfig`` instance (mutable).
    """
    model_config = vllm_config.model_config
    model = model_config.model
    revision = model_config.revision
    user_quant = model_config.quantization
    detected = detect_quantization_method(model, revision=revision)

    if detected is None:
        logger.info(
            'No quantization signature detected from model files for "%s". '
            "The model will be loaded as float. "
            'To force a quantization method, pass "--quantization <method>" explicitly.',
            model,
        )
        return

    if user_quant is not None:
        # User explicitly specified a quantization method.
        if user_quant != detected:
            logger.warning(
                "Auto-detected quantization method '%s' from model "
                "files for '%s', but user explicitly specified "
                "'--quantization %s'. Respecting the user-specified "
                "value. If you encounter errors during model loading, "
                "consider using '--quantization %s' instead.",
                detected,
                model,
                user_quant,
                detected,
            )
        return

    # No user-specified quantization — apply auto-detected value.
    model_config.quantization = detected
    logger.info(
        "Auto-detected quantization method '%s' from model files "
        "for '%s'. To override, pass '--quantization <method>' explicitly.",
        detected,
        model,
    )

    # Recreate quant_config on VllmConfig.  The original __post_init__
    # already ran _get_quantization_config(), but at that point
    # model_config.quantization was None so it returned None.  Now that
    # we've set it, we need to build the actual QuantizationConfig so the
    # downstream model-loading code can use it.
    from vllm.config import VllmConfig as _VllmConfig

    vllm_config.quant_config = _VllmConfig._get_quantization_config(model_config, vllm_config.load_config)


def enable_fa_quant(vllm_config, layer_name=None) -> bool:
    is_kv_consumer = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_consumer
    if not is_kv_consumer and get_ascend_device_type() != AscendDeviceType.A5:
        return False
    if vllm_config.quant_config is not None and getattr(vllm_config.quant_config, "enable_fa_quant", False):
        if layer_name is not None:
            return vllm_config.quant_config.enabling_fa_quant(vllm_config, layer_name)
        else:
            return True
    return False


def enable_hif4_qkv_quant(vllm_config) -> bool:
    """Return whether the checkpoint uses HiF4 for a QKV projection.

    Attention QKV pseudo quantization is tied to the ModelSlim quantization
    description instead of a global switch so non-HiF4 models keep their
    original attention path.
    """
    quant_config = getattr(vllm_config, "quant_config", None)
    quant_description = getattr(quant_config, "quant_description", None)
    if not isinstance(quant_description, dict):
        return False

    hif4_weight_names = {
        weight_name for weight_name, quant_type in quant_description.items() if quant_type == HIF4_QUANT_TYPE
    }
    if any(weight_name.endswith(_HIF4_FUSED_QKV_WEIGHT_SUFFIXES) for weight_name in hif4_weight_names):
        return True
    qkv_prefixes = [
        {weight_name[: -len(suffix)] for weight_name in hif4_weight_names if weight_name.endswith(suffix)}
        for suffix in _HIF4_SEPARATE_QKV_WEIGHT_SUFFIXES
    ]
    return bool(set.intersection(*qkv_prefixes))


def quant_dequant_mxfp4(
    tensor: torch.Tensor,
    qdim: int,
    blocksize: int = MXFP4_BLOCK_SIZE,
    stochastic_rounding: bool = False,
) -> torch.Tensor:
    """MXFP4 pseudo-quantization copied from the attention reference."""
    orig_shape = tensor.shape
    tensor = tensor.unflatten(qdim, (-1, blocksize))

    max_val = torch.amax(tensor.abs(), qdim, keepdim=True)
    inv_constant = 1 / 7
    shared_exp = torch.ceil(torch.log2(max_val.clamp(min=_MXFP4_EPSILON) * inv_constant))
    shared_exp = shared_exp.clamp(-127, 127)

    tensor = tensor * torch.exp2(-shared_exp)

    private_exp = torch.floor(torch.log2(tensor.abs().clamp(min=_MXFP4_EPSILON))).clamp(min=_MXFP4_MIN_EXP)
    tensor = tensor * torch.exp2(-private_exp) * _MXFP4_SCALE_FACTOR

    tensor_sign = torch.sign(tensor)
    tensor = tensor_sign * torch.floor_(tensor.abs() + 0.5)

    tensor = (tensor * _MXFP4_INV_SCALE_FACTOR * torch.exp2(private_exp)).clamp(-_MXFP4_MAX_NORM, _MXFP4_MAX_NORM)

    recovered_tensor = tensor * torch.exp2(shared_exp)
    return recovered_tensor.reshape(orig_shape)


def quant_dequant_mxfp4_grouped(tensor: torch.Tensor) -> torch.Tensor:
    """MXFP4 pseudo-quantization for ``[groups, 32, heads, size]``."""
    orig_shape = tensor.shape
    max_val = torch.amax(tensor.abs(), 1, keepdim=True)
    inv_constant = 1 / 7
    shared_exp = torch.ceil(torch.log2(max_val.clamp(min=_MXFP4_EPSILON) * inv_constant))
    shared_exp = shared_exp.clamp(-127, 127)

    tensor = tensor * torch.exp2(-shared_exp)

    private_exp = torch.floor(torch.log2(tensor.abs().clamp(min=_MXFP4_EPSILON))).clamp(min=_MXFP4_MIN_EXP)
    tensor = tensor * torch.exp2(-private_exp) * _MXFP4_SCALE_FACTOR

    tensor_sign = torch.sign(tensor)
    tensor = tensor_sign * torch.floor_(tensor.abs() + 0.5)

    tensor = (tensor * _MXFP4_INV_SCALE_FACTOR * torch.exp2(private_exp)).clamp(-_MXFP4_MAX_NORM, _MXFP4_MAX_NORM)

    recovered_tensor = tensor * torch.exp2(shared_exp)
    return recovered_tensor.reshape(orig_shape)


def quant_dequant_hif4(x: torch.Tensor, quant_type: str = "hifx4", axe: int = -1) -> torch.Tensor:
    """Simulate a HiF4 quantize-dequantize round trip along ``axe``.

    HiF4 uses one three-level scale hierarchy per 64 values. The selected
    dimension is padded to a complete group for quantization and cropped back
    afterwards. Padding is deliberately handled here rather than in the core
    kernel, which keeps the kernel layout simple and supports non-multiple
    attention head sizes.
    """
    del quant_type  # Kept for compatibility with existing callers.
    if x.ndim == 0:
        raise ValueError("HiF4 pseudo quantization requires a tensor with at least one dimension")
    if not -x.ndim <= axe < x.ndim:
        raise IndexError(f"HiF4 quantization axis {axe} is out of range for a {x.ndim}D tensor")
    if not x.is_floating_point():
        raise TypeError(f"HiF4 pseudo quantization requires a floating-point tensor, got {x.dtype}")
    if x.numel() == 0:
        return x

    axis = axe % x.ndim
    original_size = x.shape[axis]
    x_last = x.movedim(axis, -1) if axis != x.ndim - 1 else x
    padding = (-original_size) % _HIF4_BLOCK_SIZE
    if padding:
        pad_shape = (*x_last.shape[:-1], padding)
        x_last = torch.cat((x_last, x_last.new_zeros(pad_shape)), dim=-1)

    qdq_out = _quantize_hif4_kernel(x_last, -1)[..., :original_size]
    if axis != x.ndim - 1:
        qdq_out = qdq_out.movedim(-1, axis)
    return qdq_out.to(x.dtype)


def _quantize_hif4_kernel(x: torch.Tensor, qdim: int):
    if x.shape[qdim] % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(f"HiF4 quantization dimension must be divisible by {_HIF4_BLOCK_SIZE}, got {x.shape[qdim]}")
    x = x.unflatten(qdim, (-1, 8, 2, 4))  # head_size -> [16, 8, 2, 4] for 1024
    man_bits = 3
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    # Three-level max: innermost 4 / middle 8 / outer 64 channels
    max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
    max_lv2 = torch.max(max_lv3, dim=qdim - 1, keepdim=True)[0]
    max_lv1 = torch.max(max_lv2, dim=qdim - 2, keepdim=True)[0]

    # div7 = 1/7: allows L2/L3 to each shrink by up to 2x so the largest
    # value lands in [0, 2) after all scaling, ready for mantissa quant.
    div7 = torch.ones_like(max_lv1) / 7.0
    div7 = div7.to(torch.bfloat16).to(x.dtype)
    # Keep scale arithmetic in FP32. In float16, the E6M2 rounding expression
    # can otherwise overflow at exp2(16), even when the final scale is valid.
    scale_factor = (max_lv1 * div7).float()  # base scale
    # A zero block otherwise produces log2(0), followed by inf * 0 and NaN.
    # E6M2 uses exponent bias 48 in the HiF4 checkpoint format. Float16 cannot
    # represent 2**-48, so use its smallest normal value as a safe fallback.
    min_scale = max(2.0**_HIF4_SCALE_MIN_EXP, torch.finfo(x.dtype).tiny)
    scale_factor = scale_factor.clamp_min(min_scale)

    # round scale_factor to bf16 mantissa (simulate HW behavior)
    e_sf = torch.floor(torch.log2(scale_factor))
    mant_sf = scale_factor / 2**e_sf * 2**7
    scale_factor = torch.round(mant_sf) / 2**7 * 2**e_sf

    # round scale_factor to e6m2 (8-bit: 1 sign, 6 exp, 2 mant)
    e_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)

    # per-sub-block dynamic shift
    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)
    # L2 sub-block: scale_lv2 = 2 if max_lv2 >= 4*scale_factor else 1
    scale_lv2 = max_lv2 * rec_sf
    scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())
    # L3 sub-block: scale_lv3 = 2 if max_lv3 >= 2*scale_factor else 1
    scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())

    # mantissa quant (3-bit)
    mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf
    mant = torch.floor(mant * 2 ** (man_bits - 1) + 0.5) / 2 ** (man_bits - 1)
    upper_bound = 2 - 2 ** (-man_bits + 1)
    mant = torch.clamp_max(mant, max=upper_bound)

    # dequant: sign * mant * three-level scale (S1E2M1 carries sign)
    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor
    out = out.flatten(qdim - 3, qdim)

    return out


def _unpack_uint8_to_bits_graph(u8_tensor: torch.Tensor) -> torch.Tensor:
    """
    uint8 解包为 8 个比特位，输出 2.0 或 1.0
    """
    # 使用位操作替代除法和取模，图模式友好
    bits = (
        u8_tensor.unsqueeze(-1)
        .bitwise_and(torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=u8_tensor.device))
        .ne(0)
    )  # 非零即为 bit=1

    # 直接映射：True->2.0, False->1.0 (避免 torch.where)
    return bits.to(torch.float32) + 1.0


def _decode_e6m2_bits_graph(f_u8: torch.Tensor, bias=48) -> torch.Tensor:
    """
    E6M2 比特解析
    """
    # 使用位操作提取指数和尾数（已经是图模式最优）
    exp = ((f_u8 >> 2) & 0x3F).float() - bias
    mant = (f_u8 & 0x03).float() * 0.25 + 1.0

    return mant * torch.exp2(exp)


def unpack_hif4_scale_from_fp32_graph(
    weight: torch.Tensor,
    packed_scale_fp32: torch.Tensor,
    bias: int = 48,
):
    """
    Unpack HiF4 scale stored in FP32.

    Supports:
        packed_scale_fp32: [N, G]
        packed_scale_fp32: [E, N, G]
        or any shape [..., G]
    """

    weight_ori = weight.shape
    weight = weight.unflatten(-1, (-1, 8, 2, 4))

    # 保留所有前导维度
    leading_shape = packed_scale_fp32.shape[:-1]
    G = packed_scale_fp32.shape[-1]
    packed_int32 = packed_scale_fp32.view(torch.int32)

    f_u8 = (packed_int32 >> 24) & 0xFF
    l2_u8 = (packed_int32 >> 16) & 0xFF
    l3_u16 = packed_int32 & 0xFFFF

    # uint16 -> two uint8
    l3_u8 = torch.stack(
        (
            l3_u16 & 0xFF,
            (l3_u16 >> 8) & 0xFF,
        ),
        dim=-1,
    )
    del packed_int32, l3_u16
    # decode
    scale_factor_raw = _decode_e6m2_bits_graph(
        f_u8,
        bias=bias,
    )
    scale_lv2_raw = _unpack_uint8_to_bits_graph(
        l2_u8,
    )
    scale_lv3_raw = _unpack_uint8_to_bits_graph(
        l3_u8,
    )
    del f_u8, l2_u8, l3_u8
    # broadcast
    scale_factor = scale_factor_raw.reshape(*scale_factor_raw.shape, 1, 1, 1)
    scale_lv2 = scale_lv2_raw.reshape(*scale_lv2_raw.shape, 1, 1)

    scale_lv3 = scale_lv3_raw.reshape(
        *leading_shape,
        G,
        8,
        2,
        1,
    )
    del scale_factor_raw, scale_lv2_raw, scale_lv3_raw

    # recover weight
    weight = weight * scale_factor * scale_lv2 * scale_lv3
    weight = weight.reshape(weight_ori)
    return weight.to(torch.bfloat16)


def unpack_dynamic_hif4_tensor(packed_tensor: torch.Tensor) -> torch.Tensor:
    """
    支持任意动态维度的 uint8 HIF4 权重解包（TorchCompile / 图模式极致优化版）
    """
    init_shape = packed_tensor.shape
    device = packed_tensor.device

    # 1. 动态计算恢复后的全维度 Shape（最后一维 * 2）
    target_shape = init_shape[:-1] + (init_shape[-1] * 2,)
    # 2. 极致省显存：直接按目标形状开辟 float32 空间
    result = torch.empty(target_shape, dtype=torch.float32, device=device)
    # 3. 将输入一次性转为 int32，避免后续多次强转
    packed_int8 = packed_tensor.to(torch.uint8)

    # --- 处理低 4 位（写入偶数索引位置 0::2） ---
    low_indices = packed_int8 & 0x0F
    low_sign = (low_indices >> 3) & 0x01

    # 核心优化：用数学运算 * 0.25 代替查表，对编译器极度友好
    low_abs = (low_indices & 0x07).to(torch.float32) * 0.25
    low_sign_multiplier = 1.0 - 2.0 * low_sign.to(torch.float32)
    result[..., 0::2] = low_abs * low_sign_multiplier
    # --- 处理高 4 位（写入奇数索引位置 1::2） ---
    high_indices = (packed_int8 >> 4) & 0x0F
    high_sign = (high_indices >> 3) & 0x01

    # 同理，算术代替查表
    high_abs = (high_indices & 0x07).to(torch.float32) * 0.25
    high_sign_multiplier = 1.0 - 2.0 * high_sign.to(torch.float32)
    result[..., 1::2] = high_abs * high_sign_multiplier

    return result
