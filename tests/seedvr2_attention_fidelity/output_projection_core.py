"""ncnn stage 2: full output projection with a strict 2-D Gemm boundary."""

from __future__ import annotations

from torch import nn


class OutputProjectionCore(nn.Module):
    def __init__(self, projection: nn.Linear) -> None:
        super().__init__()
        self.projection = projection

    def forward(self, flattened):
        return self.projection(flattened)
