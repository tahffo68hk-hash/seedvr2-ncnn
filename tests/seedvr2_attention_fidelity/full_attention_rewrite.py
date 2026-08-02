"""Deployment-shaped full fidelity core: flatten/Gemm/restore projections."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from attention_fidelity import partition_bthwc, reverse_bthwc
from rope3d_reference import apply_rope_cos_sin


class FullAttentionRewrite(nn.Module):
    def __init__(self, channels: int = 32, heads: int = 4, head_dim: int = 8) -> None:
        super().__init__()
        self.channels, self.heads, self.head_dim = channels, heads, head_dim
        self.qkv = nn.Linear(channels, 3 * channels)
        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_k = nn.LayerNorm(head_dim)
        self.output = nn.Linear(channels, channels)

    def forward_stages(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> dict[str, torch.Tensor]:
        windows, metadata = partition_bthwc(x, (4, 4, 4))
        flat = windows.reshape(-1, self.channels)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = F.layer_norm(q, (self.head_dim,), self.norm_q.weight, self.norm_q.bias, self.norm_q.eps)
        k = F.layer_norm(k, (self.head_dim,), self.norm_k.weight, self.norm_k.bias, self.norm_k.eps)
        expanded_cos, expanded_sin = cos.repeat(windows.shape[0], 1), sin.repeat(windows.shape[0], 1)
        q = apply_rope_cos_sin(q.reshape(-1, self.heads, self.head_dim), expanded_cos, expanded_sin).reshape_as(q)
        k = apply_rope_cos_sin(k.reshape(-1, self.heads, self.head_dim), expanded_cos, expanded_sin).reshape_as(k)
        logits = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) * (1.0 / math.sqrt(self.head_dim))
        probabilities = torch.softmax(logits + mask.unsqueeze(1), dim=-1)
        av = torch.matmul(probabilities, v.permute(0, 2, 1, 3))
        merged = av.permute(0, 2, 1, 3).reshape(windows.shape[0], windows.shape[1], self.channels)
        output_windows = self.output(merged.reshape(-1, self.channels)).reshape_as(windows)
        return {"q": q, "k": k, "logits": logits, "probabilities": probabilities, "av": av,
                "merged": merged, "windows": output_windows, "output": reverse_bthwc(output_windows, metadata)}

    def forward(self, windows: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        flat = windows.reshape(-1, self.channels)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.norm_q(q), self.norm_k(k)
        expanded_cos, expanded_sin = cos.repeat(windows.shape[0], 1), sin.repeat(windows.shape[0], 1)
        q = apply_rope_cos_sin(q.reshape(-1, self.heads, self.head_dim), expanded_cos, expanded_sin).reshape_as(q)
        k = apply_rope_cos_sin(k.reshape(-1, self.heads, self.head_dim), expanded_cos, expanded_sin).reshape_as(k)
        logits = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) * (1.0 / math.sqrt(self.head_dim))
        probabilities = torch.softmax(logits + mask.unsqueeze(1), dim=-1)
        merged = torch.matmul(probabilities, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(windows.shape[0], windows.shape[1], self.channels)
        return self.output(merged.reshape(-1, self.channels)).reshape_as(windows)
