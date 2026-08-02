# AdaptiveWindowAttentionLayer Design (Preliminary)

## Input

- Video tokens: contiguous logical `[B,T,H,W,C]`, float32/float16/bfloat16 as supported by the deployment target.
- Text tokens (real SeedVR2 path): `[B,L_text,C]` plus lengths.
- ncnn representation: for a fixed batch of one, video can be stored as `Mat(w=C,h=W,d=H,c=T)`; dynamic batch should be handled outside the layer or as separate input records.

## Output

Video tokens in the same logical `[B,T,H,W,C]` layout. Text output is optional only if the layer owns SeedVR2 multimodal attention; otherwise it remains a separate branch.

## Parameters

- `window_t`, `window_h`, `window_w`, shifted-window flag and dynamic window policy.
- `num_heads`, `head_dim`, Q/K/V/output weights and biases.
- Q/K normalization parameters, RoPE configuration, and dtype/precision policy.
- Optional text length metadata and attention sequence offsets.

## CPU forward flow

1. Derive adaptive clipped window slices from `(T,H,W)`.
2. Build or cache partition/reverse indices; do not materialize copies where a strided gather is possible.
3. Project Q/K/V, normalize Q/K, apply RoPE.
4. For every window, concatenate its video tokens with the matching text tokens.
5. Compute scaled dot-product attention and output projection.
6. Scatter video results back through the cached reverse map.

## Vulkan forward requirements

- Shader for partition/gather or index-addressed QKV loading.
- QKV projection and Q/K normalization shaders.
- RoPE shader using precomputed sin/cos buffers.
- Tiled attention shaders for QK, softmax reduction, and AV accumulation.
- Scatter/reverse shader for window output. Text repetition should use offsets/views rather than an actual repeated buffer.

## Fusion candidates

Fuse QKV projection with gather where memory layout permits; fuse QK scale + softmax; fuse attention output with reverse scatter only when windows are non-overlapping. Shifted/overlapping windows require a defined accumulation policy.

## Why ordinary ncnn layers are insufficient

Ordinary layers can express a fixed attention core, but not the complete real operation: window count and boundaries depend on runtime resolution; reverse mapping uses runtime indices; text is repeated per variable number of windows; and SeedVR2 uses varlen FlashAttention plus RoPE. Lowering all of that to static reshape/permute layers creates unsupported ranks and unsafe memory behavior, as demonstrated by the pnnx conversion warning in this fixture.
