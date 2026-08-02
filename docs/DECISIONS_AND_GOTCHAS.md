# Decisions and gotchas

- Use flatten -> Gemm -> restore for PyTorch Linear mapping; direct 3-D InnerProduct was invalid.
- Use four heads + HostHeadMerge + projection; single-graph merge is blocked.
- Fixed RMS `valid_tokens=60` is a fidelity override; dynamic masks are key-only `-10000.f`.
- Dynamic RoPE uses local-window coordinates: theta=10000, rope_dim=6, axis_dims=[2,2,2], scales=[1,.01,.0001], fourth pair identity.
- `pnnx fp16=0` is required for FP32 parity; clean rebuild after public ABI changes.
- Metadata modes, not file existence, distinguish fixed/dynamic paths; no variant fallback.
- Phase 10 synthetic HD128 RoPE contract: rope_dim=128, theta=10000, axis_dims=[44,42,42] (even, sum 128), scales=[1,.01,.0001]. This is a documented synthetic contract, not a real-SeedVR2 claim: `NaMMRotaryEmbedding3d` derives from `rotary-embedding-torch` (dim//3=42 per axis) and the true T/H/W pair split is not resolvable locally. Phase 10 is PARTIAL for that reason.
- HD128 ncnn rope/mask Mats use the `Mat(w=64,h=64)` layout; the C=32 Phase 9.5 contract is untouched.
