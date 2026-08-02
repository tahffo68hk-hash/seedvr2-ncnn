"""FP32 reference for padded dynamic-window attention with an additive key mask."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

_DYNAMIC_FIXTURE = Path(__file__).resolve().parents[1] / "dynamic_window_fixture"
if str(_DYNAMIC_FIXTURE) not in sys.path:
    sys.path.insert(0, str(_DYNAMIC_FIXTURE))

from dynamic_window import dynamic_window_partition, dynamic_window_reverse


class DynamicMaskAttentionCore(nn.Module):
    """[windows,tokens,C] plus additive [windows,tokens,tokens] key mask."""

    def __init__(self, channels: int = 32, num_heads: int = 4, head_dim: int = 8) -> None:
        super().__init__()
        if channels != num_heads * head_dim:
            raise ValueError("channels must equal num_heads * head_dim")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.qkv = nn.Linear(channels, channels * 3)
        self.output = nn.Linear(channels, channels)

    def forward(self, windows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        flat = windows.reshape(-1, self.channels)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], self.channels * 3)
        q, k, v = torch.split(qkv, self.channels, dim=-1)
        q_heads = torch.split(q, self.head_dim, dim=-1)
        k_heads = torch.split(k, self.head_dim, dim=-1)
        v_heads = torch.split(v, self.head_dim, dim=-1)
        heads = []
        for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads):
            scores = torch.matmul(q_head, k_head.transpose(-2, -1)) * self.scale
            heads.append(torch.matmul(torch.softmax(scores + mask, dim=-1), v_head))
        merged = torch.cat(heads, dim=-1)
        return self.output(merged.reshape(-1, self.channels)).reshape_as(windows)


class DynamicMaskAttention(nn.Module):
    """BTHWC wrapper supplying a per-window additive mask for right-edge padding."""

    def __init__(self, channels: int = 32, base_window: tuple[int, int, int] = (4, 4, 4)) -> None:
        super().__init__()
        self.channels = channels
        self.base_window = base_window
        self.core = DynamicMaskAttentionCore(channels=channels)

    def partition_with_mask(self, x: torch.Tensor):
        windows, metadata = dynamic_window_partition(x, self.base_window)
        valid_bthwc = x.new_ones((*x.shape[:-1], 1))
        valid_windows, _ = dynamic_window_partition(valid_bthwc, self.base_window)
        valid = valid_windows[..., 0] > 0.5
        invalid_key = (~valid).to(dtype=x.dtype).unsqueeze(1)
        mask = invalid_key.expand(-1, valid.shape[1], -1) * -10000.0
        return windows, mask, valid, metadata

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows, mask, _, metadata = self.partition_with_mask(x)
        return dynamic_window_reverse(self.core(windows, mask), metadata)
