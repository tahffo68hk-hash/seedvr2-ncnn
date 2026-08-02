"""CPU-only exports that isolate attention operators one by one."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


class MatMulOnly(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, x.transpose(-2, -1)) * (1.0 / math.sqrt(x.shape[-1]))


class MatMulSoftmax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, x.transpose(-2, -1)) * (1.0 / math.sqrt(x.shape[-1]))
        return torch.softmax(scores, dim=-1)


class Attention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, x.transpose(-2, -1)) * (1.0 / math.sqrt(x.shape[-1]))
        return torch.matmul(torch.softmax(scores, dim=-1), x)


def write_raw(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().numpy().tofile(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.manual_seed(20250803)
    x = torch.randn((4, 32, 8), dtype=torch.float32)
    write_raw(x, artifacts / "input.bin")
    torch.save(x, artifacts / "input.pt")
    for name, model in (("matmul", MatMulOnly()), ("softmax", MatMulSoftmax()), ("attention", Attention())):
        model.eval()
        with torch.no_grad():
            output = model(x)
        torch.jit.trace(model, x, check_trace=True).save(str(artifacts / f"{name}.pt"))
        torch.save(output, artifacts / f"{name}_reference.pt")
        write_raw(output, artifacts / f"{name}_reference.bin")
        print(f"{name}: input={tuple(x.shape)} output={tuple(output.shape)}")


if __name__ == "__main__":
    main()
