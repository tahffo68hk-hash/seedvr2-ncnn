"""Readable mathematical fixed-window masked attention reference."""

from __future__ import annotations

import math

import torch
from torch import nn

from attention_fidelity import partition_bthwc, reverse_bthwc
from rope3d_reference import apply_rope_cos_sin


def make_padding_mask(num_windows: int, tokens: int, *, valid_tokens: int) -> torch.Tensor:
    if not 0 < valid_tokens <= tokens:
        raise ValueError("valid_tokens must be in (0, tokens]")
    mask = torch.zeros(num_windows, tokens, tokens, dtype=torch.float32)
    mask[:, :, valid_tokens:] = -1.0e9
    return mask


class FullAttentionReference(nn.Module):
    def __init__(self, channels: int = 32, heads: int = 4, head_dim: int = 8) -> None:
        super().__init__()
        self.channels, self.heads, self.head_dim = channels, heads, head_dim
        self.qkv = nn.Linear(channels, 3 * channels)
        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_k = nn.LayerNorm(head_dim)
        self.output = nn.Linear(channels, channels)

    def forward_stages(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> dict[str, torch.Tensor]:
        windows, metadata = partition_bthwc(x, (4, 4, 4))
        qkv = self.qkv(windows).reshape(windows.shape[0], windows.shape[1], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.norm_q(q), self.norm_k(k)
        expanded_cos, expanded_sin = cos.repeat(windows.shape[0], 1), sin.repeat(windows.shape[0], 1)
        q = apply_rope_cos_sin(q.reshape(-1, self.heads, self.head_dim), expanded_cos, expanded_sin).reshape_as(q)
        k = apply_rope_cos_sin(k.reshape(-1, self.heads, self.head_dim), expanded_cos, expanded_sin).reshape_as(k)
        logits = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) / math.sqrt(self.head_dim)
        probabilities = torch.softmax(logits + mask.unsqueeze(1), dim=-1)
        av = torch.matmul(probabilities, v.permute(0, 2, 1, 3))
        merged = av.permute(0, 2, 1, 3).reshape(windows.shape[0], windows.shape[1], self.channels)
        output_windows = self.output(merged)
        return {"q": q, "k": k, "logits": logits, "probabilities": probabilities, "av": av,
                "merged": merged, "windows": output_windows, "output": reverse_bthwc(output_windows, metadata)}

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(x, mask, cos, sin)["output"]
