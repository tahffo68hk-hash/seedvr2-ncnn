"""Fixed-window attention whose Linear layers cross ncnn as 2D Gemm nodes."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "adaptive_window_fixture"
if str(_FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIXTURE_ROOT))

from adaptive_window import window_partition, window_reverse


class FixedWindowAttentionRewrite(nn.Module):
    """BTHWC fixed attention with flattened QKV and output projection boundaries."""

    def __init__(self, channels: int = 32, num_heads: int = 4, head_dim: int = 8, window_size: int = 4) -> None:
        super().__init__()
        if channels != num_heads * head_dim:
            raise ValueError("channels must equal num_heads * head_dim")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(head_dim)
        self.qkv = nn.Linear(channels, channels * 3)
        self.output = nn.Linear(channels, channels)

    def _attention(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = torch.split(qkv, self.channels, dim=-1)
        q_heads = torch.split(q, self.head_dim, dim=-1)
        k_heads = torch.split(k, self.head_dim, dim=-1)
        v_heads = torch.split(v, self.head_dim, dim=-1)
        outputs = []
        for q_head, k_head, v_head in zip(q_heads, k_heads, v_heads):
            scores = torch.matmul(q_head, k_head.transpose(-2, -1)) * self.scale
            outputs.append(torch.matmul(torch.softmax(scores, dim=-1), v_head))
        return torch.cat(outputs, dim=-1)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return restored [windows,tokens,3C] QKV plus its partitioned input."""
        windows = window_partition(x, self.window_size)
        flat = windows.reshape(-1, self.channels)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], self.channels * 3)
        return qkv, windows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv, windows = self.project_qkv(x)
        merged = self._attention(qkv)
        projected = self.output(merged.reshape(-1, self.channels)).reshape_as(windows)
        return window_reverse(projected, tuple(x.shape), self.window_size)

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same weights and operators, with direct 3D Linears as the reference path."""
        windows = window_partition(x, self.window_size)
        qkv = self.qkv(windows)
        merged = self._attention(qkv)
        return window_reverse(self.output(merged), tuple(x.shape), self.window_size)


class FixedWindowAttentionCoreRewrite(FixedWindowAttentionRewrite):
    """Export unit: [windows,tokens,C] -> [windows,tokens,C], no BTHWC reshape graph."""

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        flat = windows.reshape(-1, self.channels)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], self.channels * 3)
        merged = self._attention(qkv)
        return self.output(merged.reshape(-1, self.channels)).reshape_as(windows)
