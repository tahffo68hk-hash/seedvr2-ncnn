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

Phase 12 synthetic single DiT block staged fidelity (release build `build/phase12-synthetic-dit-block-release`):

```text
synthetic_dit_block_validation tests/seedvr2_dit_block_ncnn/artifacts reports/synthetic_dit_block_ncnn_validation.json
synthetic_dit_block_last_layer_validation tests/seedvr2_dit_block_ncnn/artifacts reports/synthetic_dit_block_last_layer_validation.json
swiglu_contract_tests
ada_single_contract_tests
dit_block_artifact_registry_tests tests/seedvr2_dit_block_ncnn/artifacts
```

Acceptance (Phase 12 READY): real `NaMMSRTransformerBlock` topology at scaled C=512/H=4/HD=128 (seed 20260802, MLP hidden 1536); staged ncnn + host hybrid runtime; per-stage parity - RMSNorm/Linear/rope `1e-5`, packed q/k/v `1e-5`, attention/merge/projection/gate/residual/SwiGLU `1e-4`, `ada_*_in` `1e-5` (documented upstream-noise amplification), final vid/txt `1e-4`; profiles A/B/C/D all PASS; A->B->C->D->A bit-identical; last-layer `final_txt == 2*attention_residual_txt` PASS; registry/cache 19; negatives 24+7+registry; Phase 9.5/10/11 regressions PASS; JSON parses with no NaN/Inf, `real_seedvr2_weights=false`, `full_32_block_dit=false`, `android_vulkan=false`, `overall_status=PASS`.

Phase 10 synthetic HD128 (release build `build/phase10-hd128-synthetic-release`):

```text
synthetic_hd128_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

Acceptance: three HD128 cases PASS, heads/merge/projection/windows/BTHWC `<1e-4`, rope cos/sin `<1e-6`, validity/mask exact, S->M->P->S bit-identical, negatives 19/19, cache 15, 10 base regressions PASS, JSON parses with no NaN/Inf and numeric fields, `real_seedvr2_weights=false`, `overall_status=PASS`.

Phase 13 real-width feasibility + weight mapping (release build `build/phase13-real-width-release`, probe env `seedvr2-block-probe` for Python):

```text
build/phase13-real-width-release/realwidth_dit_block_validation.exe .phase13_local_artifacts reports/realwidth_seedvr2_block_feasibility.json
build/phase13-real-width-release/realwidth_artifact_inventory_tests.exe .phase13_local_artifacts reports/realwidth_artifact_inventory.json
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/aggregate_realwidth_feasibility.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/validate_weight_mapping_contract.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/test_weight_mapping_roundtrip.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/negative_tests.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/validate_realwidth_artifact_inventory.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/verify_realwidth_last_layer.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/run_phase13_regressions.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/check_no_large_tracked_artifacts.py
```

Acceptance (Phase 13 READY): RW-A->RW-A final vid/txt < 2e-4 (measured ~1e-5), all stages within thresholds, bit-identical across 3 processes, zero NaN/Inf; 19 models / cache 19; mapping contract 29/29 mapped_numel==source_numel with 0 unmapped/duplicates (12 checks x5 PASS); inventory 38 entries x3 PASS; negatives 25/25; last-layer reuse PASS (20 keys, txt MLP/ada absent, all 19 models reused); regressions 21/21 PASS; git protection PASS; flags `real_seedvr2_weights=false`, `full_32_block_dit=false`, `android_vulkan=false`, `overall_status=PASS`. Historical report baselines must be restored after regression runs (`git restore`).

Phase 14 real-weight milestone (BLOCKED_CHECKPOINT_REQUIRED - prep only, no checkpoint):

```text
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\checkpoint_loader_preflight.py --report reports\phase14_checkpoint_preflight.json
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\test_checkpoint_preflight.py
```

Acceptance (prep): preflight reports `overall_status=BLOCKED_CHECKPOINT_REQUIRED`, `download_attempted=false`, `no_network=true`, exit code 2; preflight tests 11/11 PASS (missing path, directory, wrong filename, too-small, sha mismatch, valid match, expected-filename contract, no-network source audit, CLI flags, manifest consistency). Full Phase 14 acceptance (when the checkpoint lands): real checkpoint SHA recorded; schema vs Phase 13 mapping contract (29 keys, mapped_numel==source_numel, real qk-norm all-ones gate); real-weight 19-stage FP32 export (`pnnx fp16=0`) into ignored `.phase14_local_artifacts/`; real-context block parity (norm/ada-in <1e-5, qkv/qk-norm/rope <2e-5, packed attention/projection <1e-4, SwiGLU/residuals/finals <2e-4, NaN/Inf=0, final vid/txt <2e-4); single-block adapter (capture/pytorch/ncnn, `ncnn_adapter_call_count>0`, `original_block_call_count==0`, `HYBRID BLOCK EXECUTED`); visible PNG/MP4 outputs under `.phase14_visual_outputs/`; latent/pixel/PSNR/SSIM comparison; Phase 14 negatives; Phase 13/12/core/Phase 11 regressions; flags `real_seedvr2_weights=true`, `single_ncnn_block=true`, `full_32_block_ncnn=false`, `full_pipeline_pytorch=true`, `visible_output_generated=true`, `android_vulkan=false`, `checkpoint_downloaded_by_task=false`, `large_artifacts_committed=false`.
