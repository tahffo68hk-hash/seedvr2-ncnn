import torch
from torch import nn


class OriginalQkvProjection(nn.Module):
    """The failing direct 3D Linear reference: [W,N,C] -> [W,N,3C]."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(32, 96)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
