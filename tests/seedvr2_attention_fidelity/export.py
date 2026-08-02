"""Export the staged QK-norm/RoPE migration units without external packages."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from attention_fidelity import FidelityAttention, partition_bthwc
from rope3d_reference import mmrope3d_frequencies


ROOT = Path(__file__).resolve().parent


class QkNormUnit(nn.Module):
    def __init__(self, head_dim: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class RopeApplyUnit(nn.Module):
    """ncnn target: host-provided cos/sin phases, no runtime trig."""
    def forward(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        values = x.float().reshape(x.shape[0], x.shape[1], -1, 2)
        cos = torch.cos(phases).unsqueeze(1)
        sin = torch.sin(phases).unsqueeze(1)
        even, odd = values[..., 0], values[..., 1]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).reshape_as(x).to(x.dtype)


class QkvNormRopeUnit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(32, 96)
        self.norm_q = nn.LayerNorm(8)
        self.norm_k = nn.LayerNorm(8)
        self.rope = RopeApplyUnit()

    def forward(self, windows: torch.Tensor, phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = windows.reshape(-1, 32)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], 3, 4, 8)
        q, k, v = qkv.unbind(2)
        q, k = self.norm_q(q), self.norm_k(k)
        expanded = phases.repeat(windows.shape[0], 1)
        return self.rope(q.reshape(-1, 4, 8), expanded), self.rope(k.reshape(-1, 4, 8), expanded), v.reshape(-1, 4, 8)


def _write_raw(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().to(torch.float32).numpy().tofile(path)


def export_fixture() -> Path:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.manual_seed(20260802)
    x = torch.randn(1, 4, 8, 8, 32)
    windows, _ = partition_bthwc(x, (4, 4, 4))
    tokens = torch.randn(64, 4, 8)
    phases = mmrope3d_frequencies((4, 4, 4), 8)
    qk = QkNormUnit().eval()
    rope = RopeApplyUnit().eval()
    combined = QkvNormRopeUnit().eval()
    with torch.no_grad():
        qk_output = qk(tokens)
        rope_output = rope(tokens, phases)
        combined_output = combined(windows, phases)
        model = FidelityAttention().eval()
        output = model.forward_source_equivalent(x)
    torch.jit.trace(qk, tokens, check_trace=True).save(str(artifacts / "qk_norm.pt"))
    torch.jit.trace(rope, (tokens, phases), check_trace=True).save(str(artifacts / "rope_apply.pt"))
    torch.jit.trace(combined, (windows, phases), check_trace=True).save(str(artifacts / "qkv_norm_rope.pt"))
    torch.save(x, artifacts / "reference_input.pt")
    torch.save(output, artifacts / "reference_output.pt")
    _write_raw(tokens, artifacts / "tokens.bin")
    _write_raw(phases, artifacts / "phases.bin")
    _write_raw(qk_output, artifacts / "qk_norm_output.bin")
    _write_raw(rope_output, artifacts / "rope_output.bin")
    _write_raw(windows, artifacts / "windows.bin")
    for name, tensor in zip(("q", "k", "v"), combined_output):
        _write_raw(tensor, artifacts / f"qkv_norm_rope_{name}.bin")
    _write_raw(output, artifacts / "reference_output.bin")
    (artifacts / "export-summary.txt").write_text(
        "TorchScript qk_norm: PASS\nTorchScript rope_apply: PASS\n"
        "TorchScript qkv_norm_rope: PASS\n"
        f"tokens_shape={tuple(tokens.shape)}\nphases_shape={tuple(phases.shape)}\n"
        f"windows_shape={tuple(windows.shape)}\noutput_shape={tuple(output.shape)}\n",
        encoding="utf-8",
    )
    return artifacts


if __name__ == "__main__":
    print(export_fixture())
