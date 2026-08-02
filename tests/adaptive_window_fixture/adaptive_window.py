"""Weight-free fixed-window attention reference implementation."""

from __future__ import annotations

import math

import torch


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition [B,T,H,W,C] into [B*nw, window_tokens, C]."""
    if x.ndim != 5:
        raise ValueError(f"expected [B,T,H,W,C], got {tuple(x.shape)}")
    b, t, h, w, c = x.shape
    wt = min(t, window_size)
    if t % wt or h % window_size or w % window_size:
        raise ValueError("T must be divisible by min(T, window_size); H/W must be divisible by window_size")
    nt, nh, nw = t // wt, h // window_size, w // window_size
    y = x.reshape(b, nt, wt, nh, window_size, nw, window_size, c)
    y = y.permute(0, 1, 3, 5, 2, 4, 6, 7)
    return y.reshape(b * nt * nh * nw, wt * window_size * window_size, c)


def window_reverse(
    windows: torch.Tensor,
    original_shape: tuple[int, int, int, int, int],
    window_size: int,
) -> torch.Tensor:
    """Reverse [B*nw, window_tokens, C] into [B,T,H,W,C]."""
    b, t, h, w, c = original_shape
    wt = min(t, window_size)
    if windows.ndim != 3 or windows.shape[1:] != (wt * window_size * window_size, c):
        raise ValueError("windows shape does not match original shape/window size")
    nt, nh, nw = t // wt, h // window_size, w // window_size
    if windows.shape[0] != b * nt * nh * nw:
        raise ValueError("window batch count does not match original shape")
    y = windows.reshape(b, nt, nh, nw, wt, window_size, window_size, c)
    y = y.permute(0, 1, 4, 2, 5, 3, 6, 7)
    return y.reshape(b, t, h, w, c)


def reference_attention(windows: torch.Tensor) -> torch.Tensor:
    """Reference self-attention: softmax(QK^T/sqrt(C))V, with Q=K=V."""
    if windows.ndim != 3:
        raise ValueError("expected [num_windows, tokens, channels]")
    scale = 1.0 / math.sqrt(windows.shape[-1])
    scores = torch.matmul(windows, windows.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, windows)
