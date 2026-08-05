# Phase 18D padding and fusion contract

## Planner

The fixed MVP configuration is tile 128, overlap 32, stride 96. For dimension
`d`, generate stride positions below `max(0,d-128)`, append that anchored final
position, sort, and deduplicate. The Cartesian product is row-major: `grid_y`
outer, `grid_x` inner, and `tile_index` increasing from zero.

`source_right` and `source_bottom` are exclusive. Valid width and height equal
the exclusive bounds difference. Padding fields sum with valid dimensions to
128. The planner never invents negative source coordinates or coordinates
beyond the image.

Coverage-mask tests are required for 1x1, 16x16, 64x64, 127x127, 128x128,
129x128, 256x256, 320x240, 512x512, 1024x768, and 3200x2136. Acceptance is
100% coverage, no holes, no duplicate coordinates, and no out-of-bounds tile.

## Padding

Tile extraction reads RGB8 HWC pixels from the full normalized image. An axis
uses reflect only when its valid length is greater than one and every padding
extent on that axis is strictly smaller than the valid length. Otherwise the
whole axis uses replicate. Modes are selected and recorded independently as
`padding_mode_x` and `padding_mode_y`.

For a legal reflect axis of valid length `n`, a requested coordinate `p` is
mapped with period `2*(n-1)`: reduce `p` to non-negative `m` in that period,
then use `m` when `m<n`, otherwise `2*(n-1)-m`. Replicate clamps `p` to
`[0,n-1]`. This definition is independent of image-library padding defaults.

Model input is always 128x128. After inference, only `[0,valid_height) x
[0,valid_width)` is retained. Padding pixels never contribute outside the
source image. Tests cover 1x1, 1x64, 64x1, 64x64, 127x128, right-edge,
bottom-edge, and bottom-right tiles.

## Blend window

The normalized two-dimensional weight is separable:

```text
w(x,y) = wx(x) * wy(y)
```

An axis uses complementary squared-sine/cosine weights across each overlap. For
an overlap of `n` integer samples and overlap coordinate `k` in `[0,n)`, define

```text
theta(k,n) = (pi/2) * (k + 0.5) / n
leading_tile_trailing_weight = cos(theta)^2
trailing_tile_leading_weight = sin(theta)^2
```

The pair therefore sums to one at every overlap sample and neither endpoint is
zero. Non-overlapped samples have weight 1. At an outer photo boundary the
corresponding edge weight is 1. The implementation derives actual left/right/
top/bottom overlap from neighbouring tile bounds; it does not assume every
overlap is exactly 32 because the anchored final tile can overlap more.

For each valid RGB sample:

```text
weighted_sum += sample * weight
weight_sum += weight
output = weighted_sum / max(weight_sum, 1e-12)
```

Non-finite weights or sums fail validation. `weight_sum` must be strictly
positive for every source pixel.

Stage 1 identity fusion supplies the original source pixels as every tile's
output. Required sizes are 64x64, 128x128, 129x128, 256x256, 320x240, 512x512,
and 1024x768. Acceptance is zero uncovered and zero-weight pixels. RGB8 exact
reconstruction is required after the final round-to-nearest and clamp; FP32
pre-quantization maximum error must be at most 1e-5.

Stage 1 does not use fusion output as model evidence. Real multi-tile
accumulation and disk-backed buffering begin in Stage 2.
