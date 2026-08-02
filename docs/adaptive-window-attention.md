# Adaptive Window Attention

## Input

The attention module receives flattened variable-length sequences:

- `vid`: `(L_video, C)`
- `txt`: `(L_text, C)`
- `vid_shape`: `(B, 3)`, storing per-sample `(T,H,W)` token dimensions
- `txt_shape`: `(B, 1)`, storing per-sample text lengths

Before windowing, `na.window_idx` reconstructs each video sample as `(T,H,W,C)`. This is an NHWC-like token layout, not an image NCHW tensor. QKV is projected to `(L, 3*heads*head_dim)`, rearranged to `(L, 3, heads, head_dim)`, and split into Q/K/V. The window attention inputs are cast to `bfloat16`; output is converted back to the query dtype.

## Window processing

`configs_3b/main.yaml` supplies `window=(4,3,3)` and alternates `720pwin_by_size_bysize` with `720pswin_by_size_bysize`.

- The values are dynamic target counts `(temporal,height,width)`.
- Spatial scale is `sqrt((45*80)/(H*W))`, so the calculated window size changes with resolution.
- Temporal sizing is capped using `min(T,30)` before dividing by the target temporal count.
- `ceil` determines window dimensions and the final slices are clipped to the actual tensor extent.
- There is no explicit zero-padding in this implementation.
- Temporal and spatial dimensions are both partitioned; shifted mode applies a half-window offset where the dimension is larger than one window, producing overlap/edge clipping.

## Attention flow

```text
flattened video/text tokens
  -> multimodal QKV Linear projection
  -> reshape to (L, 3, heads, head_dim)
  -> dynamic window partition for video
  -> normalize Q/K and apply 3D multimodal RoPE
  -> repeat/concatenate text with each video window
  -> FlashAttention varlen using cumulative sequence lengths
  -> split video/text outputs
  -> window reverse and sequence scatter
  -> output projection
```

`NaSwinAttention` does not construct a conventional dense attention mask. It isolates each window through `cu_seqlens_q/k`; window boundaries therefore act as the attention grouping mechanism. Text tokens are repeated for each video window and included in the same variable-length attention sequence.

## Conversion implications

The fixed neural layers are comparatively straightforward, but the graph contains dynamic shape metadata, `index_select`-based partition/reverse, repeated multimodal concatenation, cumulative sequence lengths, 3D RoPE, and FlashAttention varlen. These are the primary ncnn export risks.
