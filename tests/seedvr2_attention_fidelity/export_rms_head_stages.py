"""Export head0 RMS attention diagnostic stages without changing the production artifact."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from attention_fidelity import partition_bthwc
from export_attention_heads_rms import ROOT, write
from full_attention_reference import make_padding_mask
from rms_attention import RMSPerHeadBranchRewrite
from rope3d_reference import apply_rope_cos_sin, host_rope_cos_sin


class Head0Stage(nn.Module):
    def __init__(self, reference: RMSPerHeadBranchRewrite, stage: int):
        super().__init__()
        self.qkv = nn.Linear(reference.channels, reference.channels * 3)
        self.qkv.load_state_dict(reference.qkv.state_dict())
        self.norm_q_weight = nn.Parameter(reference.norm_q_weight.detach().clone())
        self.norm_k_weight = nn.Parameter(reference.norm_k_weight.detach().clone())
        self.eps = reference.norm_eps
        self.stage = stage

    def _raw(self, windows):
        qkv = self.qkv(windows).reshape(1, windows.shape[0], 96)
        q, k, v = qkv.split(32, dim=-1)
        return q[..., :8], k[..., :8], v[..., :8]

    def _rms(self, value, weight):
        flat = value.reshape(-1, 8)
        square = flat * flat
        mean_square = torch.mean(square, dim=-1, keepdim=True)
        inverse = torch.reciprocal(torch.sqrt(mean_square + self.eps))
        return (flat * inverse * weight).reshape_as(value)

    def forward(self, windows, mask=None, cos=None, sin=None):
        q, k, v = self._raw(windows)
        if self.stage == 1:
            return torch.cat((q, k, v), dim=-1).reshape(-1, 24)
        q, k = self._rms(q, self.norm_q_weight), self._rms(k, self.norm_k_weight)
        if self.stage == 2:
            return torch.cat((q, k, v), dim=-1).reshape(-1, 24)
        q = apply_rope_cos_sin(q.reshape(-1, 1, 8), cos, sin).reshape_as(q)
        k = apply_rope_cos_sin(k.reshape(-1, 1, 8), cos, sin).reshape_as(k)
        if self.stage == 3:
            return torch.cat((q, k, v), dim=-1).reshape(-1, 24)
        logits = torch.matmul(q, k.transpose(-2, -1))
        if self.stage == 4:
            return logits
        masked = logits * (1.0 / math.sqrt(8.0)) + mask
        if self.stage == 5:
            return masked
        probabilities = torch.softmax(masked, dim=-1)
        if self.stage == 6:
            return probabilities
        return torch.matmul(probabilities, v)


def main() -> Path:
    torch.manual_seed(20260805)
    root = ROOT.parent.parent / "build" / "rms-head-stages"
    root.mkdir(parents=True, exist_ok=True)
    reference = RMSPerHeadBranchRewrite().eval()
    x = torch.randn(1, 4, 8, 8, 32)
    windows, _ = partition_bthwc(x, (4, 4, 4))
    mask = make_padding_mask(4, 64, valid_tokens=60)
    cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
    head_windows, head_mask = windows[0], mask[0]
    for name, value in (("input", head_windows), ("mask", head_mask), ("rope_cos", cos), ("rope_sin", sin)):
        write(value, root / f"{name}.bin")
    stage_args = {1: (head_windows,), 2: (head_windows,), 3: (head_windows, head_mask, cos, sin), 4: (head_windows, head_mask, cos, sin),
                  5: (head_windows, head_mask, cos, sin), 6: (head_windows, head_mask, cos, sin), 7: (head_windows, head_mask, cos, sin)}
    metadata = {"epsilon": reference.norm_eps, "q_weight_first_last": [float(reference.norm_q_weight[0]), float(reference.norm_q_weight[-1])],
                "k_weight_first_last": [float(reference.norm_k_weight[0]), float(reference.norm_k_weight[-1])],
                "input_shape": [64, 32], "reduction_dimension": -1, "mean_square_shape": [64, 1], "bias": False}
    for stage, args in stage_args.items():
        model = Head0Stage(reference, stage).eval()
        with torch.no_grad(): output = model(*args)
        directory = root / f"stage{stage}"
        directory.mkdir(exist_ok=True)
        torch.jit.trace(model, args, check_trace=True).save(str(directory / "model.pt"))
        write(output, directory / "reference.bin")
        metadata[f"stage{stage}"] = {"shape": list(output.shape), "elements": output.numel()}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return root


if __name__ == "__main__":
    print(main())
