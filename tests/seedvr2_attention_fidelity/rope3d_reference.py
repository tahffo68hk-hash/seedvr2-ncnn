"""Host frequency generation and FP32 3-D RoPE application for the Phase 7 fixture.

This uses an interleaved complex-pair layout.  It deliberately keeps the host
frequency calculation outside the export unit, which is the intended ncnn
migration boundary when rotary_embedding_torch is unavailable in this workspace.
"""

from __future__ import annotations

import math

import torch


def mmrope3d_frequencies(shape: tuple[int, int, int], head_dim: int, text_length: int = 0) -> torch.Tensor:
    """Return [T*H*W, head_dim/2] phases; T/H/W occupy first three pairs."""
    if head_dim < 6 or head_dim % 2:
        raise ValueError("fixture RoPE requires an even head_dim >= 6")
    time, height, width = shape
    positions_t = torch.arange(time, dtype=torch.float32).view(time, 1, 1).expand(time, height, width).reshape(-1)
    positions_h = torch.arange(height, dtype=torch.float32).view(1, height, 1).expand(time, height, width).reshape(-1)
    positions_w = torch.arange(width, dtype=torch.float32).view(1, 1, width).expand(time, height, width).reshape(-1)
    pairs = head_dim // 2
    phases = torch.zeros(time * height * width, pairs, dtype=torch.float32)
    scale = torch.tensor([1.0, 1.0 / 100.0, 1.0 / 10000.0], dtype=torch.float32)
    phases[:, 0] = positions_t * scale[0]
    phases[:, 1] = positions_h * scale[1]
    phases[:, 2] = positions_w * scale[2]
    return phases


def apply_mmrope3d(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    """Apply Q or K RoPE in FP32 then restore input dtype, as SeedVR2 does."""
    if x.ndim != 3 or x.shape[0] != phases.shape[0] or x.shape[-1] != phases.shape[1] * 2:
        raise ValueError("x must be [tokens, heads, head_dim] with matching phases")
    original_dtype = x.dtype
    values = x.float().reshape(x.shape[0], x.shape[1], -1, 2)
    cos = phases.cos().view(phases.shape[0], 1, -1)
    sin = phases.sin().view(phases.shape[0], 1, -1)
    even, odd = values[..., 0], values[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.reshape_as(x).to(original_dtype)


def host_rope_cos_sin(window_t: int, window_h: int, window_w: int, *, head_dim: int,
                      rotary_dim: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Host contract: return FP32 [T*H*W, head_dim/2] interleaved-pair cos/sin.

    T/H/W occupy consecutive two-channel rotary groups; unrotated pairs are
    identity.  Tokens are flattened with W fastest, then H, then T.
    """
    if head_dim % 2 or rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("head_dim and rotary_dim must be even and rotary_dim <= head_dim")
    positions = mmrope3d_frequencies((window_t, window_h, window_w), head_dim).clone()
    pairs = head_dim // 2
    active_pairs = rotary_dim // 2
    scale = torch.exp(-math.log(theta) * torch.arange(active_pairs, dtype=torch.float32) / active_pairs)
    positions[:, :min(3, active_pairs)] *= scale[:min(3, active_pairs)]
    if active_pairs < pairs:
        positions[:, active_pairs:] = 0.0
    return positions.cos(), positions.sin()


def apply_rope_cos_sin(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply host-generated interleaved cos/sin to `[tokens, heads, head_dim]`."""
    if cos.shape != sin.shape or x.ndim != 3 or cos.shape != (x.shape[0], x.shape[-1] // 2):
        raise ValueError("cos/sin must be [tokens, head_dim/2]")
    values = x.float().reshape(x.shape[0], x.shape[1], -1, 2)
    cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    even, odd = values[..., 0], values[..., 1]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).reshape_as(x).to(x.dtype)


def host_dynamic_rope3d_cos_sin(window_t, window_h, window_w, *, head_dim, rope_dim, theta=10000.0,
                                axis_dims=(2, 2, 2), axis_scales=(1.0, 0.01, 1e-4)):
    """Mirror the C++ ``generate_dynamic_rope3d`` FP32 contract exactly.

    Returns FP32 [T*H*W, head_dim/2] interleaved-pair cos/sin for the dynamic
    MM-RoPE3D contract.  Pairs are laid out axis-major: all T pairs, then all
    H pairs, then all W pairs, then identity pairs for the remaining dims.
    Tokens are flattened with W fastest, then H, then T.
    """
    if head_dim % 2 or rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2:
        raise ValueError("head_dim and rope_dim must be even and rope_dim <= head_dim")
    if len(axis_dims) != 3 or len(axis_scales) != 3:
        raise ValueError("axis_dims and axis_scales must be length 3")
    if sum(axis_dims) != rope_dim or any(dim % 2 or dim < 0 for dim in axis_dims):
        raise ValueError("axis_dims must be non-negative even dims summing to rope_dim")
    if any(scale <= 0.0 for scale in axis_scales):
        raise ValueError("axis_scales must be positive")
    pairs = head_dim // 2
    active_pairs = rope_dim // 2
    total_tokens = window_t * window_h * window_w
    positions_t = torch.arange(window_t, dtype=torch.float32).view(window_t, 1, 1).expand(window_t, window_h, window_w).reshape(-1)
    positions_h = torch.arange(window_h, dtype=torch.float32).view(1, window_h, 1).expand(window_t, window_h, window_w).reshape(-1)
    positions_w = torch.arange(window_w, dtype=torch.float32).view(1, 1, window_w).expand(window_t, window_h, window_w).reshape(-1)
    positions = [positions_t, positions_h, positions_w]
    cos = torch.ones(total_tokens, pairs, dtype=torch.float32)
    sin = torch.zeros(total_tokens, pairs, dtype=torch.float32)
    pair_offset = 0
    for axis in range(3):
        axis_pairs = axis_dims[axis] // 2
        for local_pair in range(axis_pairs):
            pair = pair_offset + local_pair
            inv_freq = torch.exp(-torch.log(torch.tensor(theta, dtype=torch.float32))
                                 * torch.tensor(pair, dtype=torch.float32)
                                 / torch.tensor(active_pairs, dtype=torch.float32)).item()
            angles = positions[axis] * float(axis_scales[axis]) * float(inv_freq)
            cos[:, pair] = torch.cos(angles)
            sin[:, pair] = torch.sin(angles)
        pair_offset += axis_pairs
    return cos, sin


def host_lang_rope3d_cos_sin(window_t, window_h, window_w, *, head_dim, rope_dim, theta=10000.0,
                             axis_dims=(42, 42, 42), axis_scales=(1.0, 1.0, 1.0)):
    """Mirror the C++ ``generate_dynamic_rope3d`` FP32 "lang_rope" contract.

    Reproduces rotary-embedding-torch 0.5.3 ``RotaryEmbedding(dim=axis_dims[a],
    freqs_for="lang", theta=theta)`` per axis: local pair k within axis ``a``
    uses ``inv_freq = theta ** (-2*k / axis_dims[a])`` (each frequency
    duplicated for the interleaved pair).  The three axis blocks are
    concatenated T, H, W (axis-major pairs) and tokens flatten with W fastest:
    ``(lt * window_h + lh) * window_w + lw``.  Pairs beyond ``rope_dim`` stay
    identity (cos=1, sin=0), matching SeedVR2\'s partial rotary with
    ``identity_tail_dim = head_dim - rope_dim``.
    """
    if head_dim % 2 or rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2:
        raise ValueError("head_dim and rope_dim must be even and rope_dim <= head_dim")
    if len(axis_dims) != 3 or len(axis_scales) != 3:
        raise ValueError("axis_dims and axis_scales must be length 3")
    if sum(axis_dims) != rope_dim or any(dim % 2 or dim < 0 for dim in axis_dims):
        raise ValueError("axis_dims must be non-negative even dims summing to rope_dim")
    if any(scale <= 0.0 for scale in axis_scales):
        raise ValueError("axis_scales must be positive")
    pairs = head_dim // 2
    total_tokens = window_t * window_h * window_w
    positions_t = torch.arange(window_t, dtype=torch.float32).view(window_t, 1, 1).expand(window_t, window_h, window_w).reshape(-1)
    positions_h = torch.arange(window_h, dtype=torch.float32).view(1, window_h, 1).expand(window_t, window_h, window_w).reshape(-1)
    positions_w = torch.arange(window_w, dtype=torch.float32).view(1, 1, window_w).expand(window_t, window_h, window_w).reshape(-1)
    positions = [positions_t, positions_h, positions_w]
    cos = torch.ones(total_tokens, pairs, dtype=torch.float32)
    sin = torch.zeros(total_tokens, pairs, dtype=torch.float32)
    pair_offset = 0
    for axis in range(3):
        axis_pairs = axis_dims[axis] // 2
        for local_pair in range(axis_pairs):
            pair = pair_offset + local_pair
            inv_freq = torch.exp(-torch.log(torch.tensor(theta, dtype=torch.float32))
                                 * torch.tensor(local_pair, dtype=torch.float32)
                                 / torch.tensor(axis_pairs, dtype=torch.float32)).item()
            angles = positions[axis] * float(axis_scales[axis]) * float(inv_freq)
            cos[:, pair] = torch.cos(angles)
            sin[:, pair] = torch.sin(angles)
        pair_offset += axis_pairs
    return cos, sin
