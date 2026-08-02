# Adaptive Window Attention Fixture Analysis

## Input shape

The fixed test uses `B=1, T=2, H=8, W=8, C=32`, layout `[B,T,H,W,C]`, dtype `float32`, and `window_size=4`.

## Window count and tokens

The temporal window is `min(T, window_size)=2`; the spatial window is `4x4`. The number of windows is:

`B * (T / 2) * (H / 4) * (W / 4) = 1 * 1 * 2 * 2 = 4`.

Each window has `2 * 4 * 4 = 32` tokens. The partitioned tensor shape is `[4,32,32]`.

## Intermediate tensor shapes

| Tensor | Shape |
|---|---|
| Input | `[1,2,8,8,32]` |
| Reshaped block view | `[1,1,2,2,4,2,4,32]` |
| Partitioned windows | `[4,32,32]` |
| QK^T scores | `[4,32,32]` |
| Softmax weights | `[4,32,32]` |
| Attention output | `[4,32,32]` |
| Reversed output | `[1,2,8,8,32]` |

## Verification

The test checks exact equality after partition/reverse, output shape equality, and finite values. Reference tensors are saved as `reference_input.pt` and `reference_output.pt` in the fixture directory.

## Differences from real SeedVR2

- This fixture uses fixed, non-overlapping windows. SeedVR2 computes resolution-dependent windows and also has shifted-window mode.
- This fixture uses `Q=K=V=input` and plain `QK^T -> softmax -> V`. SeedVR2 performs learned multimodal QKV projections, Q/K normalization, 3D RoPE, video/text concatenation, and FlashAttention varlen.
- This fixture has no text tokens, variable-length sequence metadata, distributed gather/scatter, cache, fused normalization, or VAE.
- The fixture is intended to validate tensor layout and partition/reverse mechanics, not model accuracy or export parity.
