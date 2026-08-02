import torch
from torch import nn


class RewriteQkvProjection(nn.Module):
    """2D-Gemm rewrite of QKV projection while preserving [W,N,3C] output."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(32, 96)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows, tokens, channels = x.shape
        flat = x.reshape(windows * tokens, channels)
        projected = self.proj(flat)
        return projected.reshape(windows, tokens, 96)
