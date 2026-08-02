"""Q/K LayerNorm reference matching the configurable SeedVR2 qk_norm call site."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def apply_qk_layer_norm(q: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor,
                        q_bias: torch.Tensor, k_weight: torch.Tensor, k_bias: torch.Tensor,
                        eps: float = 1.0e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the final head_dim independently for Q and K."""
    normalized_shape = (q.shape[-1],)
    return (
        F.layer_norm(q, normalized_shape, q_weight, q_bias, eps),
        F.layer_norm(k, normalized_shape, k_weight, k_bias, eps),
    )
