# Phase 7 SeedVR2 Attention Fidelity Report

## Executed fixture

`tests/seedvr2_attention_fidelity` contains deterministic no-weight QK norm and
RoPE migration tests. Python parity uses BTHWC `[1,4,8,8,32]`, `[4,4,4]` windows,
four heads and head dimension eight. The source-style and independent FP32
attention paths matched below `1e-6`; padded partition/reverse and T/H/W
position changes also passed.

## Conversion and ncnn results

| Stage | TorchScript | pnnx | Release ncnn Extractor | Max error |
|---|---|---|---|---:|
| QK LayerNorm | PASS | PASS (`LayerNorm`) | PASS | 4.76837e-07 |
| Host-phase RoPE apply | PASS | PASS | PASS | 2.38419e-07 |
| QKV + Q/K norm + RoPE | PASS | PASS | PASS (Q/K/V outputs) | 8.94070e-07 |
| Full attention + output projection | fixture PyTorch PASS | not exported this round | not run | — |

The combined graph uses `Gemm`, two `LayerNorm` nodes, host phase tiling, and
native slice/reshape/elementwise/concat RoPE operations. It is below the FP32
`1e-4` criterion.

## Integration boundary

No Phase 7 artifact was added to `ArtifactRegistry` or the production runtime.
The existing registry-driven static cores remain the actual production path. A
new fidelity core requires an exact upstream RoPE package/config parity run and
end-to-end ncnn attention output validation before a manifest may be added.

## Blockers

The available project `.conda-cuda` contains PyTorch 2.7.0+cu128 but lacks both
`rotary_embedding_torch` and `diffusers`; no dependency was downloaded. Thus
direct instantiation of SeedVR2's `NaMMRotaryEmbedding3d` and its configured
RMS/fused norm variants is blocked. The source-equivalent fixture is auditable,
but it is not a replacement for that final upstream-package parity test.

## Phase 7.2 full masked core

Phase 7.2 adds a complete fixed-window mathematical reference and deployment
rewrite with windows `[4,64,32]`, additive mask `[4,64,64]`, and host FP32
cos/sin `[64,4]`. Reference A and rewrite B match all recorded stages below
`1e-6`; masked keys do not affect valid tokens.

TorchScript and pnnx conversion succeeded. Release ncnn runs through QK
MatMul, mask Add, Softmax and AV, then fails at the generated post-AV 4-D
head-merge Permute/Reshape (blob 89; `extract` returns `-1`). Consequently the
artifact is not added to ArtifactRegistry or Runtime. See
`full-fidelity-pnnx-graph.md` for the precise stage boundary.
