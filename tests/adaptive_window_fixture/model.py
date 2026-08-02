"""Standalone fixed-shape Adaptive Window Attention migration fixture."""

from __future__ import annotations

import math

import torch
from torch import nn

from adaptive_window import window_partition, window_reverse
from attention import ExplicitMultiHeadAttention


class WindowAttentionCore(nn.Module):
    """Exportable attention core operating on pre-partitioned [W,N,C] tensors."""

    def __init__(self, channels: int = 32, num_heads: int = 4, head_dim: int = 8) -> None:
        super().__init__()
        self.attention = ExplicitMultiHeadAttention(channels, num_heads, head_dim)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.attention(windows)


class ExportAttentionCore(WindowAttentionCore):
    """Equivalent 4-head core that keeps pnnx attention intermediates 3D."""

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        q = self.attention.wq(windows)
        k = self.attention.wk(windows)
        v = self.attention.wv(windows)
        q_heads = torch.split(q, self.attention.head_dim, dim=-1)
        k_heads = torch.split(k, self.attention.head_dim, dim=-1)
        v_heads = torch.split(v, self.attention.head_dim, dim=-1)
        head_outputs = []
        for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads):
            scores = torch.matmul(q_head, k_head.transpose(-2, -1)) * (1.0 / math.sqrt(self.attention.head_dim))
            head_outputs.append(torch.matmul(torch.softmax(scores, dim=-1), v_head))
        return self.attention.wo(torch.cat(head_outputs, dim=-1))


class AdaptiveWindowAttentionFixture(nn.Module):
    """Maps [B,T,H,W,C] to the same layout through local window attention."""

    def __init__(self, channels: int = 32, num_heads: int = 4, head_dim: int = 8, window_size: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.attention = ExplicitMultiHeadAttention(channels, num_heads, head_dim)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        windows = window_partition(x, self.window_size)
        if return_details:
            attended, attention_details = self.attention(windows, return_details=True)
            output = window_reverse(attended, tuple(x.shape), self.window_size)
            return output, {"windows": windows, **attention_details}
        attended = self.attention(windows)
        return window_reverse(attended, tuple(x.shape), self.window_size)
