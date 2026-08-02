"""Four export candidates for mathematical RMSNorm."""
import torch
from torch import nn


class _Base(nn.Module):
    def __init__(self, dim: int, eps: float, weight: torch.Tensor) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(weight.detach().clone().to(torch.float32))
        self.bias = None

    def inverse_rms(self, mean_square: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        return x * self.inverse_rms(mean_square) * self.weight


class RMSNormRewriteA(_Base):
    def inverse_rms(self, mean_square: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(mean_square + self.eps)


class RMSNormRewriteB(_Base):
    def inverse_rms(self, mean_square: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(mean_square + self.eps)


class RMSNormRewriteC(_Base):
    def inverse_rms(self, mean_square: torch.Tensor) -> torch.Tensor:
        return torch.pow(mean_square + self.eps, -0.5)


class RMSNormRewriteD(_Base):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square = x * x
        mean_square = torch.mean(square, dim=-1, keepdim=True)
        stabilized = mean_square + self.eps
        root = torch.sqrt(stabilized)
        inverse = torch.reciprocal(root)
        normalized = x * inverse
        return normalized * self.weight
