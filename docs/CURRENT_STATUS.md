# Current status

Phases 7.5-B, 8, 8.1, 9.1, 9.2, 9.3, 9.4, and 9.5 are READY for their documented fixture scope. Recent real validation: Dynamic A `[1,4,8,8,32]` error `1.49012e-7`; B `[1,4,9,10,32]` `1.49012e-7`; C `[1,5,9,10,32]` `2.08616e-7`; all validity/mask parity exact, no NaN/Inf. Cache grows 5->10->15; A->B->C->A is deterministic. Stability: multi-shape x3, runtime x3, dispatcher x5 PASS.

Phase 10 synthetic head_dim=128 / rope_dim=128 = PARTIAL: shape-only synthetic fixture validated end-to-end on CPU FP32 (single/multi/padded profiles), but the real SeedVR2 MM-RoPE3D axis split is not uniquely resolvable locally (`rotary-embedding-torch` absent), so a documented synthetic axis contract `[44,42,42]` / `[1,0.01,1e-4]` / theta=10000 is used and the state is PARTIAL, not READY. RoPE cos/sin max `5.96e-8`; heads/merge/projection/BTHWC max `8.6e-7..1.55e-6`; validity/mask exact; S->M->P->S bit-identical; negatives 19/19; cache 15; stability synthetic x5 PASS.

No blocker. TODO: confirm the real SeedVR2 RoPE axis split, complete DiT block, real weights, further shapes, Vulkan, VAE. Recommended next: extract the real single DiT block interface/shape contract (including the true MM-RoPE3D axis split).
