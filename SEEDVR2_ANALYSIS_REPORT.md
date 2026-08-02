# SeedVR2 Analysis Report

## Repository

Official source: `https://github.com/ByteDance-Seed/SeedVR`, local checkout `D:\AIProjects\third_party\SeedVR2`, `main` at commit `e4de8c24441a67e1b7df56abea10645059bb1185`. License: Apache License 2.0. No weights were downloaded or used.

## Model Architecture

The 3B config constructs `models.dit_v2.nadit.NaDiT` with 32 transformer layers and alternating adaptive/shifted window methods. It uses multimodal video/text attention, 3D rotary embeddings, RMS-style normalization and SwiGLU MLP configuration. The VAE is the causal/inflated 3D `VideoAutoencoderKL` family under `models/video_vae_v3`.

## Tensor Flow

Input is VAE-encoded to latent tokens. `VideoDiffusionInfer.inference` flattens variable-size samples, concatenates noisy latent and condition channels, and calls NaDiT with explicit `(T,H,W)` shape metadata. The result is unflattened, rescaled, rearranged to channel-first latent layout, and decoded by the VAE.

## Adaptive Window Attention

Implemented in `models/dit_v2/window.py`, `models/dit_v2/na.py`, and `models/dit_v2/nablocks/attention/mmattn.py`. Window dimensions are resolution-dependent, alternate between normal and shifted slicing, and use varlen FlashAttention with cumulative sequence lengths rather than a dense mask. See `docs/adaptive-window-attention.md`.

## Export Difficulty

| Module | Status | Reason |
|---|---|---|
| Linear | A: direct candidate | Standard projections and output layers |
| Norm | A/B | Standard norms may map; fused RMS/LN needs decomposition or replacement |
| MatMul | A: direct candidate | Core tensor operation |
| Softmax | A/B | Standard softmax is supported; FlashAttention fusion is not equivalent automatically |
| Window Partition | B | Dynamic slices, shape metadata and `index_select` |
| Window Reverse | B | Cached inverse index/scatter path |
| Dynamic Padding | A | No explicit padding found in adaptive window implementation; other VAE paths need separate inspection |
| Attention Mask | B | No dense mask; varlen sequence boundaries must be represented |
| FlashAttention varlen | C | Likely requires decomposition, graph rewrite, or a custom ncnn layer |
| 3D RoPE | B | Requires explicit graph lowering and layout validation |
| Causal/inflated VAE | B/C | 3D causal slicing, memory state, and custom inflated operations |

## ncnn Implementation Plan

1. Export a small NaDiT/window-attention fixture with fixed shapes and compare PyTorch tensors.
2. Replace dynamic window generation with precomputed fixed-shape partition/reverse indices for each supported resolution.
3. Decompose or replace FlashAttention varlen with an ncnn-supported attention sequence, validating numerical parity.
4. Treat the VAE as a separate conversion target and validate encoder/decoder independently.

## Unknowns

- Exact full-model exportability depends on FlashAttention, fused normalization, distributed gather/scatter and checkpoint availability.
- The official repository pins Torch 2.3.0, while the prepared local environment is Torch 2.7.0+cu128; this was not used to run SeedVR2.
- No model checkpoint or video was used, so no numerical alignment claim is made.

## Status

READY:

- Official repository and commit recorded.
- Entry point, pipeline, NaDiT, VAE and adaptive window modules located.
- Main conversion risks documented.

BLOCKED:

- Full model conversion is blocked until a permitted checkpoint is supplied and the dynamic/FlashAttention path is lowered.

NEXT:

- Build a weight-free fixed-shape attention fixture from the located modules and compare its PyTorch graph/tensors before attempting model export.
