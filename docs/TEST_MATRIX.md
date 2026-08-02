# Test matrix

Release ncnn tests: `artifact_registry_tests`, `dispatcher_tests`, `model_cache_tests`, `runtime_validation`, `dynamic_padding_mask_tests`, `dynamic_rope3d_tests`, `ncnn_mat_contract_tests`, `hybrid_window_adapter_tests`, `hybrid_runtime_validation`, `dynamic_multi_shape_validation`, `synthetic_hd128_validation`, `real_mmrope_contract_validation` (Phase 10.1, build `build/phase10_1-real-mmrope-release`). Acceptance: stage errors `<1e-4`, RoPE `<1e-6`, validity/mask exact, NaN/Inf=0. Known repeats: dispatcher x20, runtime x10, multi-shape x10, synthetic HD128 x5, real-contract x5.

Phase 10.1 real SeedVR2 MM-RoPE3D contract (release build `build/phase10_1-real-mmrope-release`):

```text
real_mmrope_contract_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

Acceptance: variant `seedvr2_hd128_mmrope3d_hybrid` S/M/P PASS; upstream oracle cos/sin/Q/K parity `<1e-6` (`dynamic_rope3d_tests`); effective rope_dim=126, identity tail 2; heads/merge/projection/windows/BTHWC `<1e-4`; validity/mask exact; S->M->P->S bit-identical; negatives 22/22; cache 15; 10 base regressions PASS; JSON parses with no NaN/Inf and numeric fields; `real_seedvr2_weights=false`; `overall_status=PASS`. Historical synthetic report unchanged.

Phase 11 real single DiT block contract (Python probe, no ncnn build needed):

```text
D:\AIProjects\seedvr2-ncnn\.conda-cuda\python.exe tests\seedvr2_dit_block_contract\extract_seedvr2_block_contract.py
```

Acceptance: real block class/call chain located; forward signature and field-level inputs/outputs AST-verified; token layout and `vid_freqs[l:l+f]` offset confirmed; modulation split (emb 15360 = 6*2560, shift/scale/gate per attn/mlp), attention (window (4,3,3), varlen block-diagonal, bf16), MLP (SwiGLU 2560->6912->2560) and residual order confirmed; executable window probe A=1/B=12-27/C=36-64 windows with full coverage and determinism; 16/16 negative tests PASS; JSON no NaN/Inf, numeric fields, `real_seedvr2_weights=false`, `overall_status=PARTIAL` (real forward probe BLOCKED by missing deps).

Phase 10 synthetic HD128 (release build `build/phase10-hd128-synthetic-release`):

```text
synthetic_hd128_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

Acceptance: three HD128 cases PASS, heads/merge/projection/windows/BTHWC `<1e-4`, rope cos/sin `<1e-6`, validity/mask exact, S->M->P->S bit-identical, negatives 19/19, cache 15, 10 base regressions PASS, JSON parses with no NaN/Inf and numeric fields, `real_seedvr2_weights=false`, `overall_status=PASS`.
