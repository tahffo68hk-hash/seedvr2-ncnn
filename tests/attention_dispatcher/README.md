# Adaptive Window Attention Host Dispatcher

The dispatcher owns runtime BTHWC shape decisions, padding metadata, mask
generation, window partition, and reverse/crop. ncnn receives only static
`[windows,tokens,C]` and `[windows,tokens,tokens]` tensors.

Current registry entries:

- `core_444_grid_1x4x4`: planned, unavailable.
- `core_344_grid_1x4x4`: planned, unavailable.
- `core_444_grid_1x5x5`: available; verified Case C FP32 dynamic-mask core.

The grid is part of the key because current pnnx Gemm conversion embeds the
flattened row count. There is no silent fallback between static models.
