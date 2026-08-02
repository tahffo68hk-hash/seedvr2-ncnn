"""Export full head0 B/C/D RMSNorm candidates for pnnx graph inspection."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from attention_fidelity import partition_bthwc
from export_attention_heads_rms import ROOT, write
from full_attention_reference import make_padding_mask
from rms_attention import RMSPerHeadBranchRewrite
from rope3d_reference import apply_rope_cos_sin, host_rope_cos_sin


class Head0Rewrite(nn.Module):
    def __init__(self, reference: RMSPerHeadBranchRewrite, rewrite: str):
        super().__init__()
        self.qkv = nn.Linear(32, 96)
        self.qkv.load_state_dict(reference.qkv.state_dict())
        self.norm_q_weight = nn.Parameter(reference.norm_q_weight.detach().clone())
        self.norm_k_weight = nn.Parameter(reference.norm_k_weight.detach().clone())
        self.eps, self.rewrite = reference.norm_eps, rewrite

    def _inverse(self, mean_square):
        stabilized = mean_square + self.eps
        if self.rewrite == "B": return 1.0 / torch.sqrt(stabilized)
        if self.rewrite == "C": return torch.pow(stabilized, -0.5)
        root = torch.sqrt(stabilized)
        return torch.reciprocal(root)

    def _norm(self, value, weight):
        flat = value.reshape(-1, 8)
        square = flat * flat
        mean_square = torch.mean(square, dim=-1, keepdim=True)
        return (flat * self._inverse(mean_square) * weight).reshape_as(value)

    def forward(self, windows, mask, cos, sin):
        qkv = self.qkv(windows).reshape(1, 64, 96)
        q, k, v = qkv.split(32, dim=-1)
        q, k, v = q[..., :8], k[..., :8], v[..., :8]
        q, k = self._norm(q, self.norm_q_weight), self._norm(k, self.norm_k_weight)
        q = apply_rope_cos_sin(q.reshape(-1, 1, 8), cos, sin).reshape_as(q)
        k = apply_rope_cos_sin(k.reshape(-1, 1, 8), cos, sin).reshape_as(k)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(8.0))
        return torch.matmul(torch.softmax(logits + mask, dim=-1), v)


def main() -> Path:
    torch.manual_seed(20260805)
    root = ROOT.parent.parent / "build" / "rms-head-rewrites"
    root.mkdir(parents=True, exist_ok=True)
    reference = RMSPerHeadBranchRewrite().eval()
    x = torch.randn(1, 4, 8, 8, 32)
    windows, _ = partition_bthwc(x, (4, 4, 4))
    mask = make_padding_mask(4, 64, valid_tokens=60)
    cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
    arguments = (windows[0], mask[0], cos, sin)
    for name, value in zip(("input", "mask", "rope_cos", "rope_sin"), arguments): write(value, root / f"{name}.bin")
    for rewrite in "BCD":
        model = Head0Rewrite(reference, rewrite).eval()
        directory = root / f"head_rms_{rewrite.lower()}"; directory.mkdir(exist_ok=True)
        with torch.no_grad(): output = model(*arguments)
        torch.jit.trace(model, arguments, check_trace=True).save(str(directory / "model.pt"))
        write(output, directory / "reference.bin")
    return root


if __name__ == "__main__": print(main())
