"""Small fixed-window attention path derived from SeedVR2 NaSwinAttention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from qk_norm_reference import apply_qk_layer_norm
from rope3d_reference import apply_mmrope3d, mmrope3d_frequencies


@dataclass(frozen=True)
class PartitionMetadata:
    original_shape: tuple[int, int, int, int, int]
    padded_shape: tuple[int, int, int, int, int]
    window: tuple[int, int, int]


def partition_bthwc(x: torch.Tensor, window: tuple[int, int, int]) -> tuple[torch.Tensor, PartitionMetadata]:
    batch, time, height, width, channels = x.shape
    wt, wh, ww = window
    padded = (math.ceil(time / wt) * wt, math.ceil(height / wh) * wh, math.ceil(width / ww) * ww)
    padded_x = F.pad(x, (0, 0, 0, padded[2] - width, 0, padded[1] - height, 0, padded[0] - time))
    windows = padded_x.reshape(batch, padded[0] // wt, wt, padded[1] // wh, wh, padded[2] // ww, ww, channels)
    windows = windows.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wt * wh * ww, channels)
    return windows, PartitionMetadata(tuple(x.shape), tuple(padded_x.shape), window)


def reverse_bthwc(windows: torch.Tensor, metadata: PartitionMetadata) -> torch.Tensor:
    batch, _, _, _, channels = metadata.padded_shape
    pt, ph, pw = metadata.padded_shape[1:4]
    wt, wh, ww = metadata.window
    x = windows.reshape(batch, pt // wt, ph // wh, pw // ww, wt, wh, ww, channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(metadata.padded_shape)
    _, time, height, width, _ = metadata.original_shape
    return x[:, :time, :height, :width, :]


class FidelityAttention(nn.Module):
    """QKV flatten/Gemm + separate Q/K LN + host 3-D RoPE + SDPA math."""
    def __init__(self, channels: int = 32, heads: int = 4, head_dim: int = 8, window: tuple[int, int, int] = (4, 4, 4)) -> None:
        super().__init__()
        if channels != heads * head_dim:
            raise ValueError("channels must equal heads * head_dim")
        self.channels, self.heads, self.head_dim, self.window = channels, heads, head_dim, window
        self.qkv = nn.Linear(channels, channels * 3)
        self.output = nn.Linear(channels, channels)
        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_k = nn.LayerNorm(head_dim)

    def _qkv(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = windows.reshape(-1, self.channels)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], 3, self.heads, self.head_dim)
        return qkv.unbind(2)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) / math.sqrt(self.head_dim)
        return torch.matmul(torch.softmax(scores, dim=-1), v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    def _forward(self, x: torch.Tensor, source_style: bool) -> torch.Tensor:
        windows, metadata = partition_bthwc(x, self.window)
        q, k, v = self._qkv(windows)
        if source_style:
            q, k = apply_qk_layer_norm(q, k, self.norm_q.weight, self.norm_q.bias, self.norm_k.weight, self.norm_k.bias, self.norm_q.eps)
        else:
            q, k = self.norm_q(q), self.norm_k(k)
        phases = mmrope3d_frequencies(self.window, self.head_dim).to(x.device)
        q = apply_mmrope3d(q.reshape(-1, self.heads, self.head_dim), phases.repeat(q.shape[0], 1)).reshape_as(q)
        k = apply_mmrope3d(k.reshape(-1, self.heads, self.head_dim), phases.repeat(k.shape[0], 1)).reshape_as(k)
        merged = self._attend(q, k, v).reshape(windows.shape[0], windows.shape[1], self.channels)
        projected = self.output(merged.reshape(-1, self.channels)).reshape_as(windows)
        return reverse_bthwc(projected, metadata)

    def forward_independent(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, source_style=False)

    def forward_source_equivalent(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, source_style=True)
