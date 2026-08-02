# Adaptive Window Attention Host Dispatcher Design

## Design

```text
BTHWC input -> host shape dispatcher -> Metadata
            -> host partition + additive key-mask generator
            -> static ncnn attention core (windows, mask)
            -> host reverse/crop -> BTHWC output
```

`Metadata` carries the selected window, grid, padding, number of windows, and
tokens per window.

| Case | Input | Window | Grid | Padding | Windows | Tokens |
| --- | --- | --- | --- | --- | ---: | ---: |
| A | `[1,4,16,16,32]` | `[4,4,4]` | `[1,4,4]` | `[0,0,0]` | 16 | 64 |
| B | `[1,3,16,16,32]` | `[3,4,4]` | `[1,4,4]` | `[0,0,0]` | 16 | 48 |
| C | `[1,4,17,17,32]` | `[4,4,4]` | `[1,5,5]` | `[0,3,3]` | 25 | 64 |

## Why Dynamic Logic Lives on the Host

Window selection, padding, validity-mask construction, and reverse/crop are
control flow over runtime shape metadata. C++ host code keeps their layout
contract explicit instead of exporting Python shape branches as a static graph.

## Why the Core Is Static

The ncnn core receives only `[windows,tokens,C]` and an FP32 additive mask. Its
QKV/output projections use flatten-Gemm-restore; each head remains
`MatMul -> Add(mask) -> Softmax -> MatMul`. Case C uses a real ncnn extract and
matches PyTorch BTHWC with max error `1.7881e-7`, mean `1.6054e-8`.

## Core Registry

The pnnx Gemm graph embeds flattened rows, so window size alone is not enough:
the key includes grid dimensions (or `num_windows`).

| Registry key | Status |
| --- | --- |
| `core_444_grid_1x4x4` (A) | Registered, artifact not generated |
| `core_344_grid_1x4x4` (B) | Registered, artifact not generated |
| `core_444_grid_1x5x5` (C) | Available and ncnn-verified |

Unavailable entries fail closed. Case A was verified to report its missing core
instead of falling back to the incompatible Case-C graph.

## Future Custom-Layer Boundary

Fixed cores, precomputed masks, and host dispatch need no custom layer. A custom
layer is justified only for arbitrary runtime shapes entirely inside ncnn:
dynamic allocation, grid reshaping, padding/mask generation, partition/reverse,
and crop. No Vulkan code is implemented.

## Relation to SeedVR2

This models SeedVR2's adaptive-window intent: choose windows from video shape,
apply masked local attention, then restore video layout. Variable sequence
packing, RoPE, text conditioning, and FlashAttention are intentionally omitted.
