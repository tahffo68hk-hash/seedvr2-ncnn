# Phase 8 RMSNorm analysis

The real SeedVR2 3B and 7B configs use `qk_norm: fusedrms`, `norm_eps: 1e-5`,
and `elementwise_affine: true`. The checked-in SeedVR2 normalization source
constructs `diffusers.models.normalization.RMSNorm` for the mathematical
`rms` mode and imports Apex only for `fusedrms`. No local Apex fused kernel or
diffusers package was available, so this work does not claim fused-kernel
parity.

The fixture uses FP32 `[4,64,8]`, independent Q/K weights, no bias, and the
formula `x * rsqrt(mean(x*x, -1) + eps) * weight`. Rewrite A (`rsqrt`) converts
but its standalone ncnn error is `4.31776e-4`; B (`sqrt` plus reciprocal), C
(`pow(-0.5)`), and D (explicit operations) each pass the standalone ncnn
`1e-4` threshold with max error `4.76837e-7`.

The complete RMS attention head graph converts only after flattening each head
to token rows. It emits primitive `Reduction`, `Sqrt`, and `Reciprocal` nodes;
no custom layer is required by the graph. Phase 8.1 found that the former
complete-head failure was not an RMSNorm failure: pnnx had been invoked with
its default `fp16=1`, quantizing QKV/projection weights. The QKV Stage-1 error
was `9.59277e-4` before RMSNorm. Re-exporting every head and the projection
with `fp16=0` reduces QKV to `2.38419e-7`, Q/K RMSNorm to at most
`4.76837e-7`, and final BTHWC to `1.78814e-7`.

`convert_attention_heads_rms.py` makes `fp16=0` reproducible. The production
artifact records `rmsnorm_rewrite: explicit_sqrt_reciprocal` and
`pnnx_fp16: false`; direct Apex fused-kernel parity remains unverified.
