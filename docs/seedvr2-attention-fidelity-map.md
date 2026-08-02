# SeedVR2 Attention Fidelity Map

## Source read

- `D:\AIProjects\third_party\SeedVR2\models\dit_v2\nablocks\attention\mmattn.py`: `NaMMAttention` creates `proj_qkv`, distinct `norm_q` / `norm_k`, `proj_out`, and `get_na_rope`; `NaSwinAttention.forward` partitions the sequence with `window_op`, normalizes Q/K, applies RoPE, then calls `FlashAttentionVarlen`.
- `D:\AIProjects\third_party\SeedVR2\models\dit_v2\rope.py`: `NaMMRotaryEmbedding3d.forward` changes `[L,h,d] -> [h,L,d]`, calls `apply_rotary_emb(freqs, value.float())`, then restores the original dtype. `get_freqs` calls `get_axial_freqs(1024,128,128)`, slices `(f,h,w)`, then flattens it.
- `D:\AIProjects\third_party\SeedVR2\models\dit_v2\normalization.py`: `get_norm_layer` can choose LayerNorm, RMSNorm, fused LayerNorm or fused RMSNorm; all are constructed with `dim=head_dim` at the `mmattn.py` call site.
- `D:\AIProjects\third_party\SeedVR2\models\dit_v2\window.py`: `make_720Pwindows_bysize` returns slices in W/H/T window-enumeration order. Each selected `[t,h,w]` tensor retains normal PyTorch flattening, so W is fastest within one window.

## Tensor path

`NaSwinAttention` projects sequence features `[..., C]` to `[..., 3*heads*head_dim]`, reorders to `[L,3,H,D]`, then splits Q/K/V to `[L,H,D]`. Q and K alone are normalized on their final `D` axis using separate parameter sets. RoPE is applied before attention. The attention call converts Q/K/V to BF16 for FlashAttention and casts its output back to the Q dtype; this CPU fixture validates the pre-Flash FP32 portion.

The Phase 7 fixture uses BTHWC `[1,4,8,8,32]`, window `[4,4,4]`, four heads and `head_dim=8`: partitioned windows are `[4,64,32]`; Q/K/V become `[4,64,4,8]`; the migration boundaries flatten this to `[256,4,8]` for RoPE.

## Answers constrained by source

1. RoPE frequencies are axial T/H/W frequencies returned as one combined final frequency tensor by `get_axial_freqs`, not three attention passes.
2. The exact number of rotated head channels is delegated to `rotary_embedding_torch.RotaryEmbedding`; this workspace lacks that package, so the source alone does not prove whole-head coverage for a concrete SeedVR2 config.
3. The source delegates pair layout (interleaved versus half split) to `rotary_embedding_torch.apply_rotary_emb`; it cannot be truthfully inferred without that dependency/version. The fixture's host implementation explicitly uses interleaved pairs and labels this choice.
4. Q/K normalization is before RoPE, with separate `norm_q` and `norm_k`; its exact LayerNorm/RMS/fused variant is configuration-driven.
5. Yes: norm is constructed with `dim=head_dim`, hence operates along the final head dimension.
6. Yes: Q/K own independent module weights in `NaMMAttention.__init__`.
7. Yes: RoPE application casts Q/K to FP32 and restores their incoming dtype; FlashAttention separately receives BF16 Q/K/V.
