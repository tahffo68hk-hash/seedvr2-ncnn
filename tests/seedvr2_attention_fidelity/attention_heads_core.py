"""ncnn stage 1: four independent attention-head outputs."""

from __future__ import annotations

from full_attention_per_head_rewrite import PerHeadBranchRewrite


class SingleAttentionHeadCore(PerHeadBranchRewrite):
    """Fixed one-window deployment fallback for one static attention head."""
    def __init__(self, reference, head_index: int) -> None:
        super().__init__()
        if not 0 <= head_index < 4:
            raise ValueError("head_index must be in [0, 3]")
        self.load_state_dict(reference.state_dict())
        self.head_index = head_index

    def forward(self, windows, mask, cos, sin):
        qkv = self.qkv(windows.reshape(-1, self.channels)).reshape(1, 64, 3 * self.channels)
        q, k, v = qkv.split(self.channels, dim=-1)
        begin = self.head_index * self.head_dim
        end = begin + self.head_dim
        return self._branch(q[..., begin:end], k[..., begin:end], v[..., begin:end], mask, cos, sin)[4]


class AttentionHeadsCore(PerHeadBranchRewrite):
    def forward(self, windows, mask, cos, sin):
        flat = windows.reshape(-1, self.channels)
        qkv = self.qkv(flat).reshape(windows.shape[0], windows.shape[1], 3 * self.channels)
        q, k, v = qkv.split(self.channels, dim=-1)
        d = self.head_dim
        b0 = self._branch(q[..., :d], k[..., :d], v[..., :d], mask, cos, sin)[4]
        b1 = self._branch(q[..., d:2*d], k[..., d:2*d], v[..., d:2*d], mask, cos, sin)[4]
        b2 = self._branch(q[..., 2*d:3*d], k[..., 2*d:3*d], v[..., 2*d:3*d], mask, cos, sin)[4]
        b3 = self._branch(q[..., 3*d:4*d], k[..., 3*d:4*d], v[..., 3*d:4*d], mask, cos, sin)[4]
        return b0, b1, b2, b3
