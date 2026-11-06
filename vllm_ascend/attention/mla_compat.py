# SPDX-License-Identifier: Apache-2.0
"""Tensor operations for MLA models with a smaller semantic latent rank.

Keep these operations independent of the NPU runtime so their numerical
contracts can also be checked on CPU. Physical padding never changes the
model's projection sizes, RMSNorm dimension, or attention temperature.
"""

import torch
import torch.nn.functional as F


def pad_mla_head(tensor: torch.Tensor, semantic_dim: int, physical_dim: int) -> torch.Tensor:
    """Zero-pad a head, accepting an already padded tensor on prefix reuse."""
    actual_dim = tensor.shape[-1]
    if not semantic_dim <= actual_dim <= physical_dim:
        raise ValueError(
            f"Invalid MLA head dimensions: semantic={semantic_dim}, input={actual_dim}, physical={physical_dim}"
        )
    if actual_dim == physical_dim:
        return tensor
    return F.pad(tensor, (0, physical_dim - actual_dim))


def scale_mla_query(query: torch.Tensor, scaling: torch.Tensor, token_start: int = 0) -> torch.Tensor:
    """Apply the per-token Llama-4 multiplier to either part of a query.

    The scaling tensor is an explicit forward/custom-op input. Do not cache it
    on the layer: graph replay must use the current token positions.
    """
    if query.ndim != 3 or scaling.ndim != 3 or scaling.shape[1:] != (1, 1):
        raise ValueError("MLA query scaling expects query [T, H, D] and scaling [T, 1, 1].")
    token_end = token_start + query.shape[0]
    if token_start < 0 or token_end > scaling.shape[0]:
        raise ValueError("MLA query scaling does not cover the current token range.")
    # Match upstream's q *= scaling, including rounding back to q's dtype.
    return (query * scaling[token_start:token_end]).to(query.dtype)
