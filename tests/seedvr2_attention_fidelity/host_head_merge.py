"""Reference host-side merge contract for four contiguous attention heads."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def merge_heads(heads: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(heads) != 4:
        raise ValueError("exactly four heads are required")
    if any(head.ndim != 3 for head in heads):
        raise ValueError("heads must be [windows,tokens,head_dim]")
    if any(tuple(head.shape) != tuple(heads[0].shape) for head in heads[1:]):
        raise ValueError("all heads must have the same shape")
    if any(not torch.isfinite(head).all() for head in heads):
        raise ValueError("heads must be finite")
    return torch.cat(tuple(heads), dim=-1)
