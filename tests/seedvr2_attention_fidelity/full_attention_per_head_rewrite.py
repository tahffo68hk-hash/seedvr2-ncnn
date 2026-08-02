"""Static four-head deployment rewrite that avoids post-AV 4-D permutation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from attention_fidelity import partition_bthwc, reverse_bthwc
from rope3d_reference import apply_rope_cos_sin


class PerHeadBranchRewrite(nn.Module):
    """Four explicit [windows,tokens,head_dim] attention branches."""
    def __init__(self, channels: int = 32, heads: int = 4, head_dim: int = 8) -> None:
        super().__init__()
        if heads != 4 or channels != heads * head_dim:
            raise ValueError("Phase 7.3 export is fixed to four heads")
        self.channels, self.heads, self.head_dim = channels, heads, head_dim
        self.qkv = nn.Linear(channels, channels * 3)
        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_k = nn.LayerNorm(head_dim)
        self.output = nn.Linear(channels, channels)

    def _branch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.layer_norm(q, (self.head_dim,), self.norm_q.weight, self.norm_q.bias, self.norm_q.eps)
        k = F.layer_norm(k, (self.head_dim,), self.norm_k.weight, self.norm_k.bias, self.norm_k.eps)
        repeated_cos, repeated_sin = cos.repeat(q.shape[0], 1), sin.repeat(q.shape[0], 1)
        q = apply_rope_cos_sin(q.reshape(-1, 1, self.head_dim), repeated_cos, repeated_sin).reshape_as(q)
        k = apply_rope_cos_sin(k.reshape(-1, 1, self.head_dim), repeated_cos, repeated_sin).reshape_as(k)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        probabilities = torch.softmax(logits + mask, dim=-1)
        return q, k, logits, probabilities, torch.matmul(probabilities, v)

    def concat_heads(self, head0: torch.Tensor, head1: torch.Tensor, head2: torch.Tensor, head3: torch.Tensor) -> torch.Tensor:
        return torch.cat((head0, head1, head2, head3), dim=-1)

    def _core(self, windows: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        flat = windows.reshape(-1, self.channels)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], 3 * self.channels)
        q, k, v = torch.split(qkv, self.channels, dim=-1)
        d = self.head_dim
        branch0 = self._branch(q[..., :d], k[..., :d], v[..., :d], mask, cos, sin)
        branch1 = self._branch(q[..., d:2*d], k[..., d:2*d], v[..., d:2*d], mask, cos, sin)
        branch2 = self._branch(q[..., 2*d:3*d], k[..., 2*d:3*d], v[..., 2*d:3*d], mask, cos, sin)
        branch3 = self._branch(q[..., 3*d:4*d], k[..., 3*d:4*d], v[..., 3*d:4*d], mask, cos, sin)
        # concat(heads) -> output Linear is exactly the sum of four Linear
        # contributions over the corresponding input-column blocks.  ncnn's
        # four-input Concat failed for this layout, so retain the same weight
        # matrix as four static 2-D Gemms plus the original single bias.
        out0 = F.linear(branch0[4].reshape(-1, d), self.output.weight[:, :d])
        out1 = F.linear(branch1[4].reshape(-1, d), self.output.weight[:, d:2*d])
        out2 = F.linear(branch2[4].reshape(-1, d), self.output.weight[:, 2*d:3*d])
        out3 = F.linear(branch3[4].reshape(-1, d), self.output.weight[:, 3*d:4*d], self.output.bias)
        output = (out0 + out1 + out2 + out3).reshape_as(windows)
        return output, (branch0, branch1, branch2, branch3)

    def forward(self, windows: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return self._core(windows, mask, cos, sin)[0]

    def forward_stages(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> dict[str, torch.Tensor]:
        windows, metadata = partition_bthwc(x, (4, 4, 4))
        output, branches = self._core(windows, mask, cos, sin)
        result = {"windows": output, "output": reverse_bthwc(output, metadata)}
        for index, branch in enumerate(branches):
            result[f"head{index}_q"] = branch[0]
            result[f"head{index}_k"] = branch[1]
            result[f"head{index}_logits"] = branch[2]
            result[f"head{index}_probabilities"] = branch[3]
            result[f"head{index}_av"] = branch[4]
        return result
