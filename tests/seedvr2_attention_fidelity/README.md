# SeedVR2 Attention Fidelity fixture

This fixture is a small FP32 migration probe for the code path used by
`NaSwinAttention`: fixed BTHWC windows, flattened 2-D QKV projection, separate
Q/K LayerNorm, host-generated 3-D RoPE phases, scaled dot-product attention and
reverse/crop.  It has no weights from SeedVR2.

Run `test_attention_fidelity.py` with the project's `.conda-cuda` Python and run
`export.py` before invoking pnnx on `artifacts/qk_norm.pt`, `rope_apply.pt`, or
`qkv_norm_rope.pt`.
