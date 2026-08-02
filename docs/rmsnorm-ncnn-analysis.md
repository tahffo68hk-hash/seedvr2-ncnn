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
no custom layer is required by the graph. Its current staged runtime result is
not yet within the acceptance threshold: head maxima are
`2.95371e-4`, `3.47316e-4`, `3.22342e-4`, and `3.12507e-4`. This is recorded as
a real numerical blocker for the next iteration, not as fusedrms parity.
