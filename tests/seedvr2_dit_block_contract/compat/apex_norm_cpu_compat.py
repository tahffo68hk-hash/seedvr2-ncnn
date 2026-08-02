"""Test-only CPU math-equivalent replacement for apex.normalization fused norms.

Phase 11.1: the real SeedVR2 NaMMSRTransformerBlock uses
norm="fusedrms" and qk_norm="fusedrms"
(models/dit_v2/normalization.py) which construct
`apex.normalization.FusedRMSNorm`.  NVIDIA Apex (pinned apex==0.1 in
SeedVR2 environment.yml) provides fused CUDA kernels that cannot load in a
Windows CPU-only box.  This module implements the minimal math-equivalent
interface used by the real block:

  RMSNorm(x) = x * rsqrt(mean(x^2, dim=-1) + eps) * weight   (weight optional)

FusedLayerNorm is provided as a thin nn.LayerNorm-based fallback for
completeness (the real block config does not use fusedln; only
txt_in_norm in the wider model does).  This is a TEST-HARNESS-ONLY
replacement: it does NOT claim to reproduce Apex fused-kernel numerics.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["FusedRMSNorm", "FusedLayerNorm", "CPU_COMPAT_VERSION"]

CPU_COMPAT_VERSION = "1.0.0"
_NAME = "apex.normalization.cpu_compat"


class FusedRMSNorm(nn.Module):
    """RMSNorm matching the constructor/forward contract used by SeedVR2.

    Constructor kwargs mirror apex.normalization.FusedRMSNorm:
    normalized_shape, eps=1e-5, elementwise_affine=True, dtype=None,
    device=None.  ``elementwise_affine=False`` yields no weight parameter
    (used by the block's attn_norm/mlp_norm).
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, dtype=None, device=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out.to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class FusedLayerNorm(nn.Module):
    """Minimal LayerNorm fallback with the apex constructor signature."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, dtype=None, device=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return self.ln(x)