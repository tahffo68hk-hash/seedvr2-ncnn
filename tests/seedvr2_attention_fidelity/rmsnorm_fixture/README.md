# Mathematical RMSNorm fixture

This fixture models the SeedVR2 `fusedrms` mathematical contract without
claiming Apex fused-kernel parity. The verified source configuration is
`eps=1e-5`, `elementwise_affine=true`, no bias, and normalization over the
last dimension. Input shape is `[4,64,8]` FP32.

Rewrite A uses `rsqrt`, B uses `sqrt` plus reciprocal, C uses `pow(-0.5)`, and
D explicitly exposes square, ReduceMean, Add, Sqrt, Reciprocal, multiply, and
weight multiply for ncnn graph inspection.
