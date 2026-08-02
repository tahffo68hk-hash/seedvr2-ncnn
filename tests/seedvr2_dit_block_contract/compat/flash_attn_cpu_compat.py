"""Test-only CPU math-equivalent replacement for flash_attn.flash_attn_varlen_func.

Phase 11.1: the real SeedVR2 NaMMSRTransformerBlock import chain requires
`from flash_attn import flash_attn_varlen_func` (models/dit_v2/attention.py).
flash-attn is a fused CUDA kernel library pinned at 2.5.9.post1 in
SeedVR2 environment.yml and cannot load in a Windows CPU-only box.  This
module implements the varlen block-diagonal attention CONTRACT used by the
real call chain:

  * inputs q/k/v: (total_seq, heads, head_dim) tensors, same dtype/device
  * cu_seqlens_q / cu_seqlens_k: int32 (n_segments + 1) monotonic; both must
    end at the packed length and start at 0
  * each segment is computed independently in FP32 (q @ k^T * softmax_scale,
    softmax, @ v); segments can never attend across each other
  * output: (total_seq_q, heads, head_dim) cast back to the input dtype
    (the real call chain casts inputs to bf16 before calling and the real
    kernel stores its result in the input dtype)

Only the parameter combinations used by the real SeedVR2 call chain are
supported (dropout 0, non-causal by default, no windowing/softcap/alibi);
unsupported combinations raise a clear NotImplementedError instead of
silently changing semantics.  This is a TEST-HARNESS-ONLY replacement: it
does NOT claim to reproduce flash-attn CUDA kernel numerics/fidelity.
"""
from __future__ import annotations

import torch

__all__ = ["flash_attn_varlen_func", "CPU_COMPAT_VERSION"]

CPU_COMPAT_VERSION = "1.0.0"
_NAME = "flash_attn.cpu_compat"


def _validate(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    dropout_p, causal, window_size, softcap, alibi_slopes,
    return_attn_probs, block_table,
):
    if not (q.ndim == 3 and k.ndim == 3 and v.ndim == 3):
        raise NotImplementedError(f"{_NAME}: q/k/v must be rank-3 (total, heads, head_dim), got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise NotImplementedError(f"{_NAME}: q/k/v shape mismatch {tuple(q.shape)} vs {tuple(k.shape)} vs {tuple(v.shape)}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise NotImplementedError(f"{_NAME}: q/k/v dtype mismatch {q.dtype}/{k.dtype}/{v.dtype}")
    if q.device != k.device or q.device != v.device:
        raise NotImplementedError(f"{_NAME}: q/k/v device mismatch")
    if dropout_p != 0.0:
        raise NotImplementedError(f"{_NAME}: dropout_p={dropout_p} unsupported; real SeedVR2 call chain uses dropout_p=0.0")
    if softcap != 0.0:
        raise NotImplementedError(f"{_NAME}: softcap={softcap} unsupported; real SeedVR2 call chain uses softcap=0.0")
    if alibi_slopes is not None:
        raise NotImplementedError(f"{_NAME}: alibi_slopes unsupported; real SeedVR2 call chain passes alibi_slopes=None")
    if window_size not in ((-1, -1), (-1, -1)):
        raise NotImplementedError(f"{_NAME}: window_size={window_size} unsupported; real SeedVR2 call chain uses default (-1,-1)")
    if return_attn_probs:
        raise NotImplementedError(f"{_NAME}: return_attn_probs=True unsupported")
    if block_table is not None:
        raise NotImplementedError(f"{_NAME}: block_table unsupported")
    if causal not in (False, True):
        raise NotImplementedError(f"{_NAME}: causal={causal!r} unsupported")

    for name, cu, length in (("cu_seqlens_q", cu_seqlens_q, q.shape[0]), ("cu_seqlens_k", cu_seqlens_k, k.shape[0])):
        if cu.dim() != 1:
            raise NotImplementedError(f"{_NAME}: {name} must be rank-1, got {tuple(cu.shape)}")
        if cu.shape[0] < 2:
            raise NotImplementedError(f"{_NAME}: {name} needs at least one segment (len>=2), got len {cu.shape[0]}")
        cu_l = cu.tolist()
        if cu_l[0] != 0:
            raise NotImplementedError(f"{_NAME}: {name}[0] must be 0, got {cu_l[0]}")
        if any(b < a for a, b in zip(cu_l, cu_l[1:])):
            raise NotImplementedError(f"{_NAME}: {name} not monotonic non-decreasing: {cu_l}")
        if cu_l[-1] != length:
            raise NotImplementedError(f"{_NAME}: {name}[-1]={cu_l[-1]} does not match packed length {length}")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise NotImplementedError(f"{_NAME}: cu_seqlens_q/cu_seqlens_k length mismatch {cu_seqlens_q.shape} vs {cu_seqlens_k.shape}")


def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise NotImplementedError(f"{_NAME}: cu_seqlens_q/cu_seqlens_k are required")
    _validate(q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p, causal, window_size, softcap, alibi_slopes, return_attn_probs, block_table)

    head_dim = q.shape[-1]
    scale = float(softmax_scale) if softmax_scale is not None else head_dim ** -0.5
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    n_segments = len(cu_q) - 1
    out = torch.empty_like(q)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    for i in range(n_segments):
        qs = q_f[cu_q[i]:cu_q[i + 1]]
        ks = k_f[cu_k[i]:cu_k[i + 1]]
        vs = v_f[cu_k[i]:cu_k[i + 1]]
        if qs.shape[0] == 0 or ks.shape[0] == 0:
            continue
        # (q_len, h, d) x (k_len, h, d) -> (h, q_len, k_len)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if causal:
            q_len = qs.shape[0]
            k_len = ks.shape[0]
            mask = (torch.arange(k_len, device=scores.device)[None, :]
                    > torch.arange(q_len, device=scores.device)[:, None])
            scores = scores.masked_fill(mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        seg_out = torch.einsum("hqk,khd->qhd", probs, vs)
        out[cu_q[i]:cu_q[i + 1]] = seg_out.to(q.dtype)

    return out