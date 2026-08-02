"""Export minimal QKV projection graph prefixes on CPU only."""

from pathlib import Path

import torch
from torch import nn


class _Projection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(32, 96)

    def project(self, x: torch.Tensor):
        return torch.chunk(self.proj(x), 3, dim=-1)


class LinearOnly(_Projection):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Linear2dOnly(_Projection):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LinearSplit(_Projection):
    def forward(self, x: torch.Tensor):
        return self.project(x)


class LinearReshape(_Projection):
    def forward(self, x: torch.Tensor):
        return tuple(t.reshape(4, 32, 4, 8) for t in self.project(x))


class LinearReshapeTranspose(_Projection):
    def forward(self, x: torch.Tensor):
        return tuple(t.reshape(4, 32, 4, 8).transpose(1, 2) for t in self.project(x))


def write_raw(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().numpy().tofile(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.manual_seed(20250803)
    x = torch.randn((4, 32, 32), dtype=torch.float32)
    torch.save(x, artifacts / "input.pt")
    write_raw(x, artifacts / "input.bin")
    x2d = x[0]
    torch.save(x2d, artifacts / "linear2d_input.pt")
    write_raw(x2d, artifacts / "linear2d_input.bin")
    models = (("linear", LinearOnly()), ("split", LinearSplit()), ("reshape", LinearReshape()), ("transpose", LinearReshapeTranspose()))
    for name, model in models:
        model.eval()
        with torch.no_grad():
            output = model(x)
        torch.jit.trace(model, x, check_trace=True).save(str(artifacts / f"{name}.pt"))
        outputs = output if isinstance(output, tuple) else (output,)
        for index, tensor in enumerate(outputs):
            torch.save(tensor, artifacts / f"{name}_out{index}_reference.pt")
            write_raw(tensor, artifacts / f"{name}_out{index}_reference.bin")
        print(f"{name}: {[tuple(t.shape) for t in outputs]}")
    model2d = Linear2dOnly().eval()
    with torch.no_grad():
        output2d = model2d(x2d)
    torch.jit.trace(model2d, x2d, check_trace=True).save(str(artifacts / "linear2d.pt"))
    torch.save(output2d, artifacts / "linear2d_out0_reference.pt")
    write_raw(output2d, artifacts / "linear2d_out0_reference.bin")
    print(f"linear2d: {[tuple(output2d.shape)]}")


if __name__ == "__main__":
    main()
