# Full SeedVR2 DiT streaming report (Phase 16)

Status: **READY** (branch `phase16-full-dit-streaming`, base `a4b9a5e`).
The complete 32-block real-weight SeedVR2-3B DiT body runs through ncnn in
strict per-block streaming mode (`LOAD -> EXECUTE -> UNLOAD`,
`max_resident_weighted_blocks=1`) and is embedded into the official PyTorch
pipeline for end-to-end visual validation. Real checkpoint sha256
`6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b` (Phase 14
download, 5/5 verified, official source only).

## Scope

- All 32 real blocks `0..31` of the official DiT, including the unique last
  layer 31 (`is_last_layer=true`, 15-key shared `.all.` layout).
- Blocks `0..9`: split 29-key layout (separate `.vid.` / `.txt.` weights,
  158,633,493 numel per block); blocks `10..31`: shared 15-key `.all.` layout
  (79,316,757 numel per block).  The block count and layouts are derived from
  the checkpoint, never hardcoded.
- Ranges: `range_8 = [9..16]`, `range_16 = [8..23]`, `range_full = [0..31]`
  (contiguous, strictly ascending, last layer only as the final block).
- Real q/k RMSNorm weights are NOT all-ones on any block ->
  `REAL_AFFINE_REQUIRED`; real affine bins are exported (no Phase 13 0-byte
  reuse).
- All 32/32 artifacts verified (`reports/phase16_artifact_inventory.json`,
  logical 13.32 GB, physical dedup; blocks 15-18 hardlink-reused from
  Phase 15; weight-free stages deduplicated to one shared set).
- Full state_dict->ncnn 29/15-key mapping contract PASS for all 32 blocks
  (`reports/phase16_all_block_mapping.json`): expected == consumed, no
  unmapped/duplicate keys, no shape mismatches, QKV split/concat bit-exact,
  SwiGLU gate/up/out correct, AdaSingle runtime-only.

## Adapter

`tests/seedvr2_dit_block_ncnn/full_dit_streaming_adapter.cpp`
(`build/phase16-full-dit-release/full_dit_streaming_adapter.exe`):
range 0..31, strictly increasing, last layer 31 at most once and only as the
final block, declared index must match artifact metadata; per-block strict
state machine (IDLE/LOADING/READY/EXECUTING/UNLOADING/FAILED) with
`expected_next_block_index` progression; on every block: runtime destroyed,
model cache cleared to 0, resident bytes 0, artifact file rename-probe passes;
prints `HYBRID BLOCK EXECUTED` with `denoising_step/block_index/
is_last_layer/invocation_index/artifact_set_sha256/input_hash/...`.  Per-block
outputs are written so the ncnn chain result can be replayed through the
official pipeline (no silent fallback; `original_block_call_count=0`).

## Chain parity (streaming == FP32 reference)

Fresh GPU processes; FP32 PyTorch chain reference (bf16 cast patched to
identity, TF32 off) using a sequential move-to-GPU strategy (one weighted
block resident on GPU at a time; the shared MM-RoPE3D grid is cached per
block) so the full 32-block chain fits in 16 GB VRAM.

| chain | blocks | final relative max error | limit | NaN/Inf | repeats |
|-------|--------|--------------------------|-------|---------|---------|
| chain_8 | 9..16 | 5.6465e-4 | 1e-2 | 0 | 3 (bit-identical) |
| chain_16 | 8..23 | 2.0992e-3 | 2e-2 | 0 | 3 (bit-identical) |
| chain_full | 0..31 | 5.0234e-3 | 5e-2 | 0 | 3 (bit-identical) |

Per-block relative errors: chain_8 `5.64e-4..5.66e-4`; chain_16
`6.9e-4..2.10e-3`; chain_full `2.8e-4..7.3e-3` (all under the 5e-2 full-chain
limit; the 1e-3 single-block budget is only the Phase 14/15 single-block
criterion).  Last-layer standalone parity: combined rel `4.05e-6` (limit
1e-3), deterministic, `max_resident_weighted_blocks=1`
(`reports/phase16_last_layer_parity.json`).

## Official-pipeline visual hybrid (8/16/full)

4 frames 128x128, seed 20260803, steps 1, cfg 1.0.  The whole chain is
replaced by ONE strict adapter call (per hybrid run) with the input captured
at the chain's first block; the ncnn per-block outputs are replayed through
the pipeline so the full ncnn chain flows end-to-end.

| mode | blocks | pixel PSNR (dB) | SSIM | pixel RMSE | latent RMSE | adapter ncnn/orig/fallback |
|------|--------|-----------------|------|------------|-------------|----------------------------|
| 8 | 9..16 | 51.336 | 0.9974 | 0.002711 | 0.02209 | 8 / 0 / false |
| 16 | 8..23 | 52.551 | 0.9976 | 0.002357 | 0.01561 | 16 / 0 / false |
| full | 0..31 | 51.050 | 0.9972 | 0.002802 | 0.02373 | 32 / 0 / false |

Thresholds for the full chain: PSNR >= 34 dB and SSIM >= 0.95 -> PASS.  All
modes: `order_ok=true`, `streaming_mode_validated=true`,
`max_resident_weighted_blocks=1`, `state_machine_final=IDLE`,
`expected_next_block_index` = 17 / 24 / 32; frames non-constant; NaN/Inf = 0.
Visible outputs under git-ignored `.phase16_visual_outputs/`:
`02_hybrid_8_blocks` / `03_hybrid_16_blocks` / `04_hybrid_full_dit` /
`05_diff_8` / `06_diff_16` / `07_diff_full` / `08_side_by_side_*` /
`09_error_growth` (panels `Input | PyTorch | Hybrid ncnn | Amplified
difference`).

## Error growth

`reports/full_dit_error_growth.json` (latent RMSE / pixel RMSE): E1 `0.01286 /
0.00208`, E2 `0.01159 / 0.00231`, E4 `0.01424 / 0.00226`, E8 `0.02209 /
0.00271`, E16 `0.01561 / 0.00236`, Efull `0.02373 / 0.00280`.  E1/E2/E4 reuse
the Phase 15 `multiblock_visual_comparison.json`; E8/E16/EFull are measured in
this phase.  The curve is not monotonic because the error is dominated by the
blocks surrounding the Phase 14/15 anchor (block 16) and by the last-layer
head path, not by the raw chain length.  See
[full-dit-error-growth-report.md](full-dit-error-growth-report.md).

## Lifecycle / negatives / regressions / protection

- Negative tests `reports/phase16_negative_tests.json`: **80/80 PASS**
  (range/selection incl. last-layer rules, artifact/weight binding, strict
  streaming lifecycle, adapter negative invocations, silent-fallback/call
  counting, numerical guards, visual sanity, Git protection).
- Historical regressions `reports/phase15_regressions.json`: **27/27 PASS**
  (Phase 12/13/14 binaries, 12 core attention-dispatcher tests, 4 Phase 11
  Python suites, Phase 15 streaming adapter; tracked baselines restored).
- Git protection `reports/phase16_git_protection.json`: PASS (no
  checkpoint/weights, no `.phase16_*` dirs, no large binaries, no secrets, no
  conflict markers; all Phase 16 JSON reports parse and are finite).
- `tests/seedvr2_weight_mapping/check_no_large_tracked_artifacts.py`: PASS.

## Known gotchas

- The official `vid_out_ada` head modulation reuses the shared cache key
  `emb_repeat_0_vid` that a real block's attn-in vid modulation writes (cache
  keys only carry `idx` + `branch_tag`, not layer/mode).  With the whole
  chain replaced by ncnn no real block runs, so the visual harness rebuilds
  that weight-independent entry exactly (einops `l=2,g=3` + repeat by
  `vid_len`) before the head executes (`run_phase16_visual.ChainHybrid`).
- flash_attn / apex fused-CUDA kernel fidelity is NOT validated; the
  reference and hybrid runs both use harness-injected math-equivalent CPU
  compat.
- 32-block all-resident FP32 ~20.1 GB / FP16 ~10.0 GB (estimates) - per-block
  streaming is the validated strategy (`max_resident_weighted_blocks=1`).

## Not in scope

VAE-in-ncnn, full video restoration, Android/Vulkan/Pad 7 Pro, FP16/quantized
ncnn, fused-CUDA direct parity (Phase 17+).