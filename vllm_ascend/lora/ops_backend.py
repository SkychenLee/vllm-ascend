# SPDX-License-Identifier: Apache-2.0
"""Resolve the optional standalone LoRA kernels on their first use."""

from functools import cache
from importlib import import_module
from importlib.util import find_spec

import torch
from vllm.logger import logger

_LORA_OP_NAMES = (
    "bgmv_shrink",
    "bgmv_expand",
    "sgmv_shrink",
    "sgmv_expand",
    "build_combined_lora_idx",
    "add_lora",
    "add_lora_shrink",
    "add_lora_expand",
    "add_lora_expand_delta",
    "add_lora_swiglu_quant",
)


def get_lora_ops():
    if _use_ops_repo():
        # Import statements are traceable by Dynamo; returning a module from
        # assume_constant_result would leave its operator attributes unguarded.
        import ops_repo

        return ops_repo
    return torch.ops._C_ascend


@torch.compiler.assume_constant_result
def _use_ops_repo():
    return _load_lora_ops() is not None


@cache
def _load_lora_ops():
    """Prefer ops_repo; retain the in-tree backend when it is not installed.

    Importing the MoE execution modules must not load an optional native
    library in workers that do not use LoRA. Resolve only on the first LoRA
    operation and keep the selected backend for the lifetime of the worker.
    A broken or incompatible installation is an error, never a fallback.
    """
    if find_spec("ops_repo") is None:
        return None

    try:
        backend = import_module("ops_repo")
    except Exception as exc:
        raise ImportError(
            "ops_repo is installed but could not be imported for the Ascend "
            "LoRA operators. Rebuild or reinstall ops_repo for this PyTorch, "
            "torch-npu and CANN environment."
        ) from exc

    missing = [name for name in _LORA_OP_NAMES if not callable(getattr(backend, name, None))]
    if missing:
        raise ImportError(
            "The installed ops_repo does not provide the required Ascend LoRA "
            f"API: {', '.join(missing)}. Install an ops_repo build with LoRA support."
        )
    logger.info(
        "Using ops_repo LoRA operators from %s (torch.ops.ops_repo).",
        getattr(backend, "__file__", "<unknown>"),
    )
    return backend
