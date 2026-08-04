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

Phase 14 real-weight single-block milestone (release build `build/phase14-real-weight-release`, GPU env `.conda-cuda`, probe env `seedvr2-block-probe` for Python):

```text
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\checkpoint_loader_preflight.py --path .phase14_local_checkpoint\seedvr2_ema_3b.pth --expected-sha256 6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b --report reports\phase14_checkpoint_preflight.json
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\update_preflight_status.py
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\real_checkpoint_schema.py
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\real_weight_block_mapping.py
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\export_real_weight_block_stages.py
build\phase14-real-weight-release\realwidth_dit_block_validation.exe .phase14_local_artifacts reports\real_weight_block_parity_rwa.json real_seedvr2_block16_c2560_h20_hd128 A
build\phase14-real-weight-release\real_block_adapter.exe --root .phase14_local_artifacts --variant real_seedvr2_block16_c2560_h20_hd128 --input-dir <dir> --output-dir <dir> --report <json>
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\run_official_pipeline.py --mode reference
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\run_official_pipeline.py --mode hybrid
.conda-cuda\python.exe tests\seedvr2_real_weight_hybrid\run_official_pipeline.py --milestone-only
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/phase14_negative_tests.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/run_phase14_regressions.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/phase14_git_protection.py
```

Acceptance (Phase 14 READY): preflight `overall_status=READY`, `download_completed=true`, 5/5 sha256 verified, `official_source_only=true`, `mirror_used=false`, no absolute paths/tokens; schema 635 fp32 keys / 32 blocks derived / block 16 = floor(32/2) middle (not last layer); forced q/k RMSNorm real-affine check (`all_ones=false` -> `REAL_AFFINE_REQUIRED`, Phase 13 0-byte reuse FORBIDDEN); real-weight 15-key mapping contract (`mapped_numel==source_numel`, unmapped/duplicates/shape-mismatches empty, QKV split/concat bit-exact, SwiGLU correct, AdaSingle runtime-only); RW-A real-weight parity `phase14_verdict=PASS` (final vid 1.812e-4 / txt 1.144e-4 < 2e-4, NaN/Inf=0, bit-identical); official-context block-16 parity (vid abs 5.15e-5 / txt abs 2.44e-4, rel 1.4e-6 / 8.9e-9 < 1e-3 - absolute 2e-4 documented for the O(1) RW-A fixture); adapter evidence `ncnn_adapter_call_count=1` / `original_block_call_count==0` / `silent_fallback=false` / `HYBRID BLOCK EXECUTED`; visible outputs (01_input/02_pytorch_reference/03_hybrid_one_block_ncnn/04_abs_diff/05_side_by_side, panels `Input | PyTorch | Hybrid ncnn | Amplified difference`); latent PSNR 36.3 dB, pixel PSNR 52.5 dB / SSIM 0.9975; negatives 29/29; regressions 22/22 (Phase 12/13 6 + core 12 + Phase 11 Python 4, historical baselines restored with `git restore`); git protection 8/8; milestone `verdict=PASS`; flags `real_seedvr2_weights=true`, `single_ncnn_block=true`, `full_32_block_ncnn=false`, `full_pipeline_pytorch=true`, `visible_output_generated=true`, `android_vulkan=false`, `checkpoint_downloaded_by_task=true`, `large_artifacts_committed=false`. Phase 15 (full 32-block DiT / Android / Vulkan / VAE-in-ncnn) NOT started.
Phase 15 real-weight multi-block streaming hybrid (release build `build/phase15-multiblock-release`, probe env `seedvr2-block-probe` for Python; GPU env `.conda-cuda` for the FP32 reference/visual pipeline):

```text
build/phase15-multiblock-release/multiblock_streaming_adapter.exe --root .phase15_local_artifacts --blocks 16:real_seedvr2_block16_c2560_h20_hd128 --input-dir .phase15_local_artifacts/official_context_capture/block_16 --output-dir <dir> --load-mode streaming --report <json>
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/run_phase15_chain_parity.py --skip-capture
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/run_phase15_negative_tests.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/run_phase15_regressions.py
.conda-cuda\python.exe tests/seedvr2_real_weight_hybrid/run_phase15_memory_profile.py
.conda-cuda\python.exe tests/seedvr2_real_weight_hybrid/run_phase15_visual.py
```

Acceptance (Phase 15 READY): streaming lifecycle strict per block (`LOAD -> EXECUTE -> UNLOAD`, `cache_cleared=true`, `cache_size_after=0`, `resident_bytes_after=0`, file rename-probe passes, `max_resident_weighted_blocks=1`, `order_ok=true`, `streaming_mode_validated=true`); chain parity chain_1 rel 8.872e-9 (limit 1e-3) / chain_2 2.835e-6 (2e-3) / chain_4 3.777e-6 (5e-3), NaN/Inf=0, streaming==resident bit-identical; adapter rejects out-of-range/mismatched block bindings (index must match metadata `block.block_index`, range 0..30); visual 1/2/4 blocks: pixel PSNR 53.65/52.72/52.90 dB (>=38), SSIM 0.9978/0.9978/0.9977 (>=0.98), per-block parity 1.8e-8..3.8e-6 (1e-3), adapter evidence per block `ncnn=1`/`original=0`/`silent_fallback=false`; memory profile streaming peak flat (474/496/476 MB, 4/1 ratio 1.004) vs resident linear (446/839/1530 MB, 3.43); negatives 48/48; regressions 27/27 (Phase 12/13/14 8 + core 12 + Phase 11 Python 4 + Phase 15 streaming adapter 2, historical baselines restored with `git restore`); git protection PASS (no `.phase15_*`/heavy files tracked); flags `real_seedvr2_weights=true`, `multiblock_hybrid=true`, `max_resident_weighted_blocks=1`, `streaming_mode_validated=true`, `full_32_block_dit=false`, `android_vulkan=false`, `large_artifacts_committed=false`. Phase 16 (full 32-block DiT / Android / Vulkan / VAE-in-ncnn) NOT started.
Phase 16 full SeedVR2 DiT streaming (release build `build/phase16-full-dit-release`, probe env `seedvr2-block-probe` for Python; GPU env `.conda-cuda` for the FP32 reference/visual pipeline):

```text
build/phase16-full-dit-release/full_dit_streaming_adapter.exe --root .phase16_local_artifacts --range 9:16 --input-dir <dir> --output-dir <dir> --report <json>   # or 8:23 / 0:31
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/phase16_block_schedule.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/phase16_all_block_mapping.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/phase16_artifact_inventory.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/phase16_block_reuse.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/run_phase16_chain_parity.py --skip-capture
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/phase16_last_layer_parity.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/run_phase16_negative_tests.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/run_phase15_regressions.py
conda run -n seedvr2-block-probe python tests/seedvr2_real_weight_hybrid/run_phase16_git_protection.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/check_no_large_tracked_artifacts.py
.conda-cuda\python.exe tests/seedvr2_real_weight_hybrid/run_phase16_visual.py
```

Acceptance (Phase 16 READY): block schedule + all-block mapping 32/32 (`expected==consumed`, unmapped/duplicates/shape-mismatches empty, QKV split/concat bit-exact, SwiGLU correct, AdaSingle runtime-only; q/k RMSNorm NOT all-ones -> `REAL_AFFINE_REQUIRED`); artifact inventory 32/32 (logical 13.32 GB, blocks 15-18 hardlink-reused from Phase 15, weight-free stages deduplicated); strict streaming lifecycle per block (`LOAD -> EXECUTE -> UNLOAD`, `cache_cleared=true`, `cache_size_after=0`, `resident_bytes_after=0`, file rename-probe passes, `max_resident_weighted_blocks=1`, `order_ok=true`, `streaming_mode_validated=true`); chain parity chain_8 rel 5.6465e-4 (limit 1e-2) / chain_16 2.0992e-3 (2e-2) / chain_full 5.0234e-3 (5e-2), NaN/Inf=0, 3x bit-identical; last-layer standalone combined rel 4.05e-6 (limit 1e-3); visual 8/16/full pixel PSNR 51.336/52.551/51.050 dB (>=34), SSIM 0.9974/0.9976/0.9972 (>=0.95), adapter evidence `ncnn=8/16/32`/`original=0`/`silent_fallback=false`/`state_machine_final=IDLE`; error growth E1..Efull finite (`reports/full_dit_error_growth.json`); negatives 80/80; regressions 27/27 (Phase 12/13/14/15 8 + core 12 + Phase 11 Python 4 + Phase 15 streaming adapter 2, historical baselines restored with `git restore`); git protection PASS (no `.phase16_*`/heavy files tracked); flags `real_seedvr2_weights=true`, `full_32_block_dit=true`, `max_resident_weighted_blocks=1`, `android_vulkan=false`, `large_artifacts_committed=false`. Phase 17 (VAE-in-ncnn / Android / Vulkan / fused-kernel parity) NOT started.

Phase 17 real-weight SeedVR2 VAE ncnn + full video restoration (probe env `seedvr2-block-probe` for CPU parity; GPU env `.conda-cuda` for the official-pipeline visual runs):

```text
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/inspect_real_vae_schema.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/generate_vae_weight_mapping.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/validate_vae_weight_mapping.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/verify_vae_export_wrapper.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/export_real_vae_stages.py --profile A   # or B
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_vae_stage_parity.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_vae_encoder_parity.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_vae_decoder_parity.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_vae_roundtrip.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_vae_tiling_contract.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_vae_streaming_lifecycle.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_phase17_profiles.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_phase17_negative_tests.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_phase17_regressions.py
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/check_no_large_tracked_artifacts.py
conda run -n seedvr2-block-probe python tests/seedvr2_vae_ncnn/run_phase17_git_protection.py
.conda-cuda\python.exe tests/seedvr2_vae_ncnn/run_full_ncnn_core_pipeline.py --mode pytorch_reference
.conda-cuda\python.exe tests/seedvr2_vae_ncnn/run_full_ncnn_core_pipeline.py --mode dit_ncnn_vae_pytorch
.conda-cuda\python.exe tests/seedvr2_vae_ncnn/run_full_ncnn_core_pipeline.py --mode dit_ncnn_decoder_ncnn
.conda-cuda\python.exe tests/seedvr2_vae_ncnn/run_full_ncnn_core_pipeline.py --mode full_ncnn_core
.conda-cuda\python.exe tests/seedvr2_vae_ncnn/run_full_ncnn_core_pipeline.py --metrics
```

Acceptance (Phase 17 READY): official `ema_vae.pth` (sha256 `c7df8a67...`) exported as 14 FP32 ncnn stage chains (`pnnx fp16=0`, profiles A/B, git-ignored `.phase17_local_artifacts/`); weight mapping 250/250 (`mapped_numel == source_numel`, no unmapped/duplicates/shape mismatches, 6 deconv layout + 4 reshape conversions); stage parity 14/14 rel <= ~2.1e-6 (limit 1e-3); encoder parity A 2.166e-6 / B 1.897e-6 (limit 2e-3); decoder parity A/B rel 2.08e-5 / 1.73e-5, PNG PSNR 117.9 / 119.3 dB (>=45), SSIM 1.0 (>=0.99); full-VAE roundtrip A/B pixel PSNR ~116.7 / 117.8 dB, SSIM 1.0, adapter ncnn encoder 7 / decoder 14 vs original 1/2 and 2/4, silent_fallback=false; tiling contract (4k+1 padded convention, T=9/T=17 slicing equivalence < 8.4e-7, raw T=8 slice crash reproduced); streaming lifecycle (encoder 405 MB / decoder 607 MB, A->B->A error 0, cache 0, file handles released); latency / memory / IO profiles PASS; negatives 70/70; regressions PASS (Phase 14/15 matrices + Phase 16 CPU-light + cited GPU baselines, historical baselines restored); git protection PASS. Official-pipeline full video restoration (4 frames 128x128, seed 20260803, steps 1, cfg 1.0): `full_ncnn_core` pixel PSNR 51.195 dB (>=40) / SSIM 0.9990 (>=0.98), latent RMSE 0.0206, NaN/Inf=0, non-constant; `dit_ncnn_vae_pytorch` 51.998 dB / 0.9989; `dit_ncnn_decoder_ncnn` 52.105 dB / 0.9990; adapter evidence `ncnn_vae_encoder=1` / `ncnn_vae_decoder=1` / `original=0` / `silent_fallback=false` / `chain_hybrid_ran=true` / `max_resident_weighted_dit_blocks=1` / `max_resident_vae_model_sets=1`; harness RNG anchor keeps z bit-identical across modes; flags `real_seedvr2_weights=true`, `real_vae_weights=true`, `full_pure_ncnn_application=false`, `max_resident_weighted_dit_blocks=1`, `android_vulkan=false`, `large_artifacts_committed=false`. Phase 18 (Android/Vulkan / fused-kernel parity / full-pure-ncnn) NOT started.
| Phase 18A Android Gradle/JNI/ncnn ARM64 build | PASS | Debug + unsigned Release, arm64-v8a/AArch64 only |
| Phase 18A host model-manifest validation | PASS | 9/9 executed on Windows host; not Android execution |
| Phase 18A APK model/ABI/sensitive-path inspection | PASS | No weights, one ABI, no local paths |
| Phase 18A synthetic ncnn CPU on Android | NOT RUN | Binary built; device required |
| Phase 18A Vulkan device/inference | NOT RUN | Vulkan compiled/prepared only; no Pad connected |
| Phase 18A real VAE/DiT/full restoration on Android | NOT RUN | Phase 18B+ device work |
