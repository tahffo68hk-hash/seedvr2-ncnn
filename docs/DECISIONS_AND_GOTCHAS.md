# Decisions and gotchas

- Use flatten -> Gemm -> restore for PyTorch Linear mapping; direct 3-D InnerProduct was invalid.
- Use four heads + HostHeadMerge + projection; single-graph merge is blocked.
- Fixed RMS `valid_tokens=60` is a fidelity override; dynamic masks are key-only `-10000.f`.
- Dynamic RoPE uses local-window coordinates: theta=10000, rope_dim=6, axis_dims=[2,2,2], scales=[1,.01,.0001], fourth pair identity.
- `pnnx fp16=0` is required for FP32 parity; clean rebuild after public ABI changes.
- Metadata modes, not file existence, distinguish fixed/dynamic paths; no variant fallback.
- Phase 10 synthetic HD128 RoPE contract (historical regression, unchanged): rope_dim=128, theta=10000, axis_dims=[44,42,42] (even, sum 128), scales=[1,.01,.0001].
- Phase 10.1 real SeedVR2 MM-RoPE3D contract (Phase 10 READY): rotary-embedding-torch==0.5.3, SeedVR2 `e4de8c2`; configured rope_dim=128 -> `RotaryEmbedding(dim=42, freqs_for="lang", theta=10000)`; effective rotary dim 126 (axis_dims=[42,42,42]), identity tail 2, per-axis freq `theta^(-k/21)`, T/H/W axis-major, W-fastest tokens. Runtime metadata encodes `rope_dim=126` + `freq_mode="lang_rope"`; `freq_mode` is an optional general extension (default `"global_pairs"` keeps the legacy rope_dim=6 and synthetic contracts). Configured 128 is documentation-only (oracle manifest / JSON / docs).
- HD128 ncnn rope/mask Mats use the `Mat(w=64,h=64)` layout; the C=32 Phase 9.5 contract is untouched.
- Phase 11 real single DiT block contract (PARTIAL, source/AST): `NaMMSRTransformerBlock` (SeedVR2 `e4de8c2`) - flattened `[L,C]` tokens (video T-major W-fastest; text 1-D), per-window `[vid_win; txt]` varlen block-diagonal attention with txt repeated per window and mean-coalesced outputs, `vid_freqs[l:l+f, :h, :w]` (l = text length, f = post-patch frame count), AdaSingle emb 6*dim split shift/scale/gate per attn/mlp, SwiGLU hidden round-up to multiple of 256 (2560->6912), pre-norm residual with post-attention/MLP gate. Real block probe is BLOCKED by missing non-model deps (einops/diffusers/flash_attn/apex/rotary-embedding-torch); do not install them without approval, do not substitute a hand-written probe.
