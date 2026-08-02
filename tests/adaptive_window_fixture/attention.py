"""Explicit attention operators chosen to map to basic export graph nodes."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn


class ExplicitMultiHeadAttention(nn.Module):
    """QKV -> scaled matmul -> softmax -> V, without MultiheadAttention."""

    def __init__(self, channels: int, num_heads: int, head_dim: int, bias: bool = True) -> None:
        super().__init__()
        if channels != num_heads * head_dim:
            raise ValueError("channels must equal num_heads * head_dim")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.wq = nn.Linear(channels, channels, bias=bias)
        self.wk = nn.Linear(channels, channels, bias=bias)
        self.wv = nn.Linear(channels, channels, bias=bias)
        self.wo = nn.Linear(channels, channels, bias=bias)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        num_windows, tokens, _ = x.shape
        q = self.wq(x).reshape(num_windows, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(num_windows, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(num_windows, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # Merge windows and heads so exported MatMul/Softmax tensors are 3D.
        # This preserves the logical 4-head computation but avoids 4D MatMul.
        q_flat = q.reshape(num_windows * self.num_heads, tokens, self.head_dim)
        k_flat = k.reshape(num_windows * self.num_heads, tokens, self.head_dim)
        v_flat = v.reshape(num_windows * self.num_heads, tokens, self.head_dim)
        scores_flat = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attention_flat = torch.softmax(scores_flat, dim=-1)
        attended_flat = torch.matmul(attention_flat, v_flat)
        attention = attention_flat.reshape(num_windows, self.num_heads, tokens, tokens)
        merged = attended_flat.reshape(num_windows, self.num_heads, tokens, self.head_dim).transpose(1, 2).reshape(num_windows, tokens, self.channels)
        output = self.wo(merged)
        if return_details:
            return output, {"q": q, "k": k, "v": v, "attention": attention, "attention_flat": attention_flat}
        return output
