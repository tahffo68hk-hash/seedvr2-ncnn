# Test matrix

Release ncnn tests: `artifact_registry_tests`, `dispatcher_tests`, `model_cache_tests`, `runtime_validation`, `dynamic_padding_mask_tests`, `dynamic_rope3d_tests`, `ncnn_mat_contract_tests`, `hybrid_window_adapter_tests`, `hybrid_runtime_validation`, `dynamic_multi_shape_validation`, `synthetic_hd128_validation`, `real_mmrope_contract_validation` (Phase 10.1, build `build/phase10_1-real-mmrope-release`). Acceptance: stage errors `<1e-4`, RoPE `<1e-6`, validity/mask exact, NaN/Inf=0. Known repeats: dispatcher x20, runtime x10, multi-shape x10, synthetic HD128 x5, real-contract x5.

Phase 10.1 real SeedVR2 MM-RoPE3D contract (release build `build/phase10_1-real-mmrope-release`):

```text
real_mmrope_contract_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

Acceptance: variant `seedvr2_hd128_mmrope3d_hybrid` S/M/P PASS; upstream oracle cos/sin/Q/K parity `<1e-6` (`dynamic_rope3d_tests`); effective rope_dim=126, identity tail 2; heads/merge/projection/windows/BTHWC `<1e-4`; validity/mask exact; S->M->P->S bit-identical; negatives 22/22; cache 15; 10 base regressions PASS; JSON parses with no NaN/Inf and numeric fields; `real_seedvr2_weights=false`; `overall_status=PASS`. Historical synthetic report unchanged.

Phase 11 real single DiT block contract (Python, no ncnn build needed) - extraction:

```text
D:\AIProjects\seedvr2-ncnn\.conda-cuda\python.exe tests\seedvr2_dit_block_contract\extract_seedvr2_block_contract.py
```

Phase 11.1 real random-weight forward execution (isolated conda env `seedvr2-block-probe`):

```text
conda run -n seedvr2-block-probe python tests\seedvr2_dit_block_contract\probe_real_block.py
conda run -n seedvr2-block-probe python run_phase11_1_probe.py
```

Acceptance (Phase 11.1 READY): real `NaMMSRTransformerBlock` (SeedVR2 `e4de8c2`) imported + instantiated at full 2560 width (158,633,472 params); real forward A `(1,4,4)`/B `(5,32,32)`/C `(7,45,64)`/D batch2 `(2,8,8)+(3,10,12)` PASS with finite outputs, shapes round-trip, deterministic bit-identical; compat parity RMSNorm/varlen max_abs < 1e-6; runtime hooks confirm `[vid_win; txt]` per-window packing + txt repeat/mean coalesce, `vid_freqs[l:l+f]` offset, identity tail 2 elementwise, effective rotary dim 126/128, AdaSingle six-segment modulation + gate placement, SwiGLU 2560->6912->2560, residual order, last-layer vid_only, window reassembly without cross-segment leakage; negatives 17/17; Phase 10 regressions PASS (dynamic_rope3d_tests, real_mmrope_contract_validation, synthetic_hd128_validation, dynamic_multi_shape_validation, runtime_validation); report states `FlashAttention CUDA kernel fidelity = not validated`, `Apex fused kernel fidelity = not validated`, `real_seedvr2_weights=false`; `overall_status=READY`.

Phase 10 synthetic HD128 (release build `build/phase10-hd128-synthetic-release`):

```text
synthetic_hd128_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

Acceptance: three HD128 cases PASS, heads/merge/projection/windows/BTHWC `<1e-4`, rope cos/sin `<1e-6`, validity/mask exact, S->M->P->S bit-identical, negatives 19/19, cache 15, 10 base regressions PASS, JSON parses with no NaN/Inf and numeric fields, `real_seedvr2_weights=false`, `overall_status=PASS`.
