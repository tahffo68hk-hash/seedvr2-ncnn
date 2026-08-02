"""Mathematical RMSNorm per-head attention used by the Phase 8 fixture."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from attention_fidelity import partition_bthwc, reverse_bthwc
from rope3d_reference import apply_rope_cos_sin


class RMSNorm(nn.Module):
    def __init__(self, dim=8, eps=1.0e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = None

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight


class RMSPerHeadBranchRewrite(nn.Module):
    def __init__(self, channels=32, heads=4, head_dim=8, eps=1.0e-5):
        super().__init__()
        if channels != heads * head_dim:
            raise ValueError("channels must equal heads * head_dim")
        self.channels, self.heads, self.head_dim = channels, heads, head_dim
        self.qkv = nn.Linear(channels, channels * 3)
        self.norm_q_weight = nn.Parameter(torch.ones(head_dim))
        self.norm_k_weight = nn.Parameter(torch.ones(head_dim))
        self.norm_eps = float(eps)
        self.output = nn.Linear(channels, channels)

    def _branch(self, q, k, v, mask, cos, sin):
        q_flat = q.reshape(-1, self.head_dim)
        k_flat = k.reshape(-1, self.head_dim)
        q_mean_square = torch.mean(q_flat * q_flat, dim=-1, keepdim=True)
        k_mean_square = torch.mean(k_flat * k_flat, dim=-1, keepdim=True)
        q = (q_flat * torch.reciprocal(torch.sqrt(q_mean_square + self.norm_eps)) * self.norm_q_weight).reshape_as(q)
        k = (k_flat * torch.reciprocal(torch.sqrt(k_mean_square + self.norm_eps)) * self.norm_k_weight).reshape_as(k)
        repeated_cos, repeated_sin = cos.repeat(q.shape[0], 1), sin.repeat(q.shape[0], 1)
        q = apply_rope_cos_sin(q.reshape(-1, 1, self.head_dim), repeated_cos, repeated_sin).reshape_as(q)
        k = apply_rope_cos_sin(k.reshape(-1, 1, self.head_dim), repeated_cos, repeated_sin).reshape_as(k)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        probabilities = torch.softmax(logits + mask, dim=-1)
        return q, k, logits, probabilities, torch.matmul(probabilities, v)

    def _core(self, windows, mask, cos, sin):
        qkv = self.qkv(windows.reshape(-1, self.channels)).reshape(windows.shape[0], windows.shape[1], 3 * self.channels)
        q, k, v = qkv.split(self.channels, dim=-1)
        d = self.head_dim
        branches = tuple(self._branch(q[..., i*d:(i+1)*d], k[..., i*d:(i+1)*d], v[..., i*d:(i+1)*d], mask, cos, sin) for i in range(4))
        outputs = tuple(F.linear(branch[4].reshape(-1, d), self.output.weight[:, i*d:(i+1)*d], self.output.bias if i == 3 else None) for i, branch in enumerate(branches))
        return sum(outputs).reshape_as(windows), branches

    def forward(self, windows, mask, cos, sin):
        return self._core(windows, mask, cos, sin)[0]

    def forward_stages(self, x, mask, cos, sin):
        windows, metadata = partition_bthwc(x, (4, 4, 4))
        output, branches = self._core(windows, mask, cos, sin)
        result = {"windows": output, "output": reverse_bthwc(output, metadata)}
        for index, branch in enumerate(branches):
            result.update({f"head{index}_q": branch[0], f"head{index}_k": branch[1], f"head{index}_logits": branch[2], f"head{index}_probabilities": branch[3], f"head{index}_av": branch[4]})
        return result


class SingleRMSAttentionHead(RMSPerHeadBranchRewrite):
    def __init__(self, reference, head_index):
        super().__init__(reference.channels, reference.heads, reference.head_dim, reference.norm_eps)
        self.load_state_dict(reference.state_dict())
        self.head_index = head_index

    def forward(self, windows, mask, cos, sin):
        token_windows = windows.reshape(1, -1, self.channels)
        qkv = self.qkv(token_windows.reshape(-1, self.channels)).reshape(1, token_windows.shape[1], 3 * self.channels)
        q, k, v = qkv.split(self.channels, dim=-1)
        d = self.head_dim
        return self._branch(q[..., self.head_index*d:(self.head_index+1)*d], k[..., self.head_index*d:(self.head_index+1)*d], v[..., self.head_index*d:(self.head_index+1)*d], mask, cos, sin)[4]
