"""Readable mathematical RMSNorm reference matching SeedVR2's affine RMS path."""
import torch
from torch import nn


class RMSNormReference(nn.Module):
    def __init__(self, dim: int, eps: float, weight: torch.Tensor) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(weight.detach().clone().to(torch.float32))
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return normalized * self.weight
