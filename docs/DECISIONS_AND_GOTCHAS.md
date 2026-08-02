# Decisions and gotchas

- Use flatten → Gemm → restore for PyTorch Linear mapping; direct 3-D InnerProduct was invalid.
- Use four heads + HostHeadMerge + projection; single-graph merge is blocked.
- Fixed RMS `valid_tokens=60` is a fidelity override; dynamic masks are key-only `-10000.f`.
- Dynamic RoPE uses local-window coordinates: theta=10000, rope_dim=6, axis_dims=[2,2,2], scales=[1,.01,.0001], fourth pair identity.
- `pnnx fp16=0` is required for FP32 parity; clean rebuild after public ABI changes.
- Metadata modes, not file existence, distinguish fixed/dynamic paths; no variant fallback.
