# Phase 12: synthetic single SeedVR2 DiT block staged ncnn fidelity

Validates a scaled synthetic `NaMMSRTransformerBlock` (real SeedVR2 topology,
SeedVR2 commit `e4de8c2`) as a staged ncnn + host hybrid runtime.

- `generate_synthetic_dit_block_reference.py` - runs the REAL block at scaled
  C=512/H=4/HD=128 with deterministic seed `20260802` and writes FP32 stage
  references + manifests for profiles A/B/C/D (normal and last-layer variants).
- `export_synthetic_dit_block.py` - exports the stage models with `pnnx fp16=0`
  (dynamic-Gemm Linears, no-affine RMSNorm dim 512, per-head q/k norm dim 128,
  rope_apply/head fixed `q_max=128`, swiglu_act).
- `hybrid_dit_block_runtime.{h,cpp}` - `HybridDiTBlockRuntime`: ncnn stable
  stages + host windowing/packing/txt mean merge/AdaSingle/residuals.
- `synthetic_dit_block_validation.cpp` - per-stage A->B->C->D->A x5 parity
  report (`reports/synthetic_dit_block_ncnn_validation.json`).
- `synthetic_dit_block_last_layer_validation.cpp` - vid_only semantics
  (`final_txt == 2*attention_residual_txt`).
- `swiglu_contract_tests.cpp` / `ada_single_contract_tests.cpp` /
  `dit_block_artifact_registry_tests.cpp` - unit/negative suites.
- `validate_reference_manifest.py` - validates manifests + sha256.

Build (clean Release): `cmake --build build/phase12-synthetic-dit-block-release --config Release`.

Key contract facts (verified, do not re-derive): qkv is per-token `(o h d)` so
q/k/v are strided 512-wide slices; q/k RMSNorm is per-head dim 128; SwiGLU
hidden = `256*ceil(int(2*512*4/3)/256) = 1536`; MM-RoPE3D effective rotary dim
126 of configured 128 with identity tail 2; batch>1 packed q/k txt follows the
real `repeat_concat_idx` indexing (repeated-txt rows `[cum_b, cum_b+l_b)` per
window of sample b; v uses the original txt). No real weights, no real
flash-attn/apex CUDA kernels, no full 2560-width block here (Phase 13).
