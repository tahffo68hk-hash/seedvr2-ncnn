# Multi-block SeedVR2 streaming hybrid runtime report (Phase 15)

Status: **READY** (branch `phase15-multiblock-streaming-hybrid`, base
`17ce482`). Real-weight SeedVR2 blocks are chained through ncnn in strict
streaming mode: per block `LOAD -> EXECUTE -> UNLOAD` (runtime destroyed, model
cache cleared to 0, resident bytes 0, file handles released and rename-probed)
before the next block loads, so `max_resident_weighted_blocks = 1` at all
times.

## Scope

- Checkpoint: official `ByteDance-Seed/SeedVR2-3B`
  `seedvr2_ema_3b.pth` sha256
  `6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b`
  (5/5 files verified in Phase 14; no mirror, no login/token).
- Anchor: block 16 = floor(32/2) middle; last layer 31 excluded.
- Ranges: `range_1 = [16]`, `range_2 = [16,17]`,
  `range_4 = [15,16,17,18]` (contiguous, ascending, never crossing the last
  layer).
- Artifacts: blocks 15/16/17/18 real-weight 19-stage FP32 exports
  (`pnnx fp16=0`) under git-ignored `.phase15_local_artifacts/`; block 16
  reuses the Phase 14 artifact via read-only hardlinks
  (`reused_from_phase14=true`, `source_block_index=16`, same inode, zero-copy).
- Runtime: existing `HybridDiTBlockRuntime` + `NcnnModelCache` with new
  streaming support (`erase_by_prefix` / `release_block`,
  `resident_bytes_estimate`, `load_count`), driven by the new
  `multiblock_streaming_adapter` executable (clean build
  `build/phase15-multiblock-release`).

## Streaming lifecycle contract

The adapter emits a strict event sequence per block:

```
LOAD i (cache 0 -> 19-stage models, resident bytes ~317 MB)
EXECUTE i (chain order; `HYBRID BLOCK EXECUTED` with content hashes)
UNLOAD i (runtime destroyed, cache_cleared=true, cache_size_after=0,
          resident_bytes_after=0, file rename-probe passes)
```

Validation (Python `streaming_block_manager.validate_streaming_lifecycle` +
C++ in-process order validator):

- `order_ok = true`, `streaming_mode_validated = true`;
- `max_resident_weighted_blocks = 1` for every streaming run;
- every UNLOAD has `cache_cleared = true`, `cache_size_after = 0`,
  `resident_bytes_after = 0`;
- `ncnn_adapter_call_count >= 1`, `original_block_call_count = 0`,
  `silent_fallback = false` (the adapter is the ncnn path; there is no
  PyTorch fallback);
- wrong block bindings are rejected: the adapter now cross-checks the
  CLI-declared index against `metadata.json` `block.block_index` and the valid
  range 0..30 (fabricated indices such as `99:variant16`, out-of-range, and
  mismatched bindings exit non-zero before execution).

## Chain parity (streaming == resident, FP32)

`run_phase15_chain_parity.py --skip-capture` (fresh Phase 15 build, 3 repeats
per chain, official captured block-16 inputs):

| chain | final relative max error | limit | NaN/Inf |
|-------|--------------------------|-------|---------|
| chain_1 (block 16)          | 8.872e-9  | 1e-3 | 0/0 |
| chain_2 (blocks 16,17)      | 2.835e-6  | 2e-3 | 0/0 |
| chain_4 (blocks 15..18)     | 3.777e-6  | 5e-3 | 0/0 |

All `rc=0`, `order_ok=true`, `streaming_mode_validated=true`,
`max_resident_weighted_blocks=1`. Streaming and resident modes produce
bit-identical final outputs. Note: for chain_2/chain_4 the upstream blocks in
the hybrid run consume the hybrid output of the previous block by design
(`chained_upstream_hybrid_input`), so per-block error accumulates only over
the replaced prefix; the per-block `chain_parity_per_block` rows record the
errors of the same block index under the respective chain.

## Visual milestone (official pipeline, 1/2/4 blocks swapped to ncnn)

`run_phase15_visual.py`: official SeedVR2 SR pipeline (seed 20260803, steps 1,
cfg 1.0, 4 frames 128x128) with the target block range replaced by the ncnn
streaming adapter. Reference runs in a separate FP32 eval process with a shared
`Cache()` and non-target blocks offloaded to CPU (the MM-RoPE3D frequency grid
is module-cached, so per-block `Cache(disable=True)` OOMs; shared cache +
CPU-offload is the required memory strategy).

| mode | pixel PSNR (dB) | SSIM  | latent PSNR (dB) | NaN/Inf | non-constant |
|------|-----------------|-------|------------------|---------|--------------|
| 1 block  | 53.65 | 0.9978 | 37.82 | 0/0 | true |
| 2 blocks | 52.72 | 0.9978 | 38.72 | 0/0 | true |
| 4 blocks | 52.90 | 0.9977 | 36.93 | 0/0 | true |

Thresholds: pixel PSNR >= 38 dB, SSIM >= 0.98. `verdict = PASS`.
Per-block official-context parity: relative max error 1.8e-8 .. 3.8e-6
(limit 1e-3); block-16 single-block parity 8.87e-9 (vid) / 1.4e-6 (txt,
absolute 2.44e-4 on txt magnitude ~2.7e4). Adapter evidence for every block in
every mode: `ncnn_adapter_call_count=1`, `original_block_call_count=0`,
`silent_fallback=false`, `order_ok=true`, `max_resident_weighted_blocks=1`,
stdout contains `HYBRID BLOCK EXECUTED`.

Visible outputs (git-ignored `.phase15_visual_outputs/`):
`01_input`, `02_pytorch_reference`, `03_hybrid_1_block_ncnn`,
`04_hybrid_2_block_ncnn`, `05_hybrid_4_block_ncnn`, `06_abs_diff_4`,
`07_latent_abs_diff`, `08_side_by_side` (panels
`Input | PyTorch | Hybrid ncnn 1 | Hybrid ncnn 2 | Hybrid ncnn 4 |
Amplified difference`).

## Memory profile

`run_phase15_memory_profile.py` -> `reports/multiblock_memory_profile.json`:

| mode | process peak RSS (MB) |
|------|-----------------------|
| streaming 1/2/4 blocks | 474 / 496 / 476 (flat, 4/1 ratio 1.004) |
| resident 1/2/4 blocks   | 446 / 839 / 1530 (linear, 4/1 ratio 3.43) |

`streaming_flat=true`, `resident_grows=true`: streaming keeps one block
resident regardless of chain length (per-block streaming required for the
32-block DiT).

## Negative tests

`run_phase15_negative_tests.py` -> `reports/phase15_negative_tests.json`:
48/48 PASS covering range/selection (duplicate, non-contiguous, out-of-bounds,
negative, last-layer, empty, crossing-last-layer rejected; valid anchor/range_1
accepted), artifact/weight binding (metadata block_index + `pnnx fp16=0`,
manifest exists, block-16 Phase-14 reuse, q/k RMSNorm real affine NOT all-ones
and block-specific), strict streaming lifecycle (out-of-order LOAD/EXECUTE/
UNLOAD rejected, cache-not-cleared and resident-bytes-not-zero detected),
adapter negative invocations (unknown variant, missing input, out-of-range
block index), silent-fallback/call-count detection, numerical guards
(NaN/Inf/limit), resident==streaming final vid, visual sanity (constant
frames, frame count, PSNR/SSIM thresholds), and Git protection (no heavy
`.pth/.pt/.mp4/.npy/.png`, no `.phase15_*` tracked).

## Historical regressions

`run_phase15_regressions.py` -> `reports/phase15_regressions.json`: 27/27 PASS
(Phase 12/13/14 binaries from `build/phase14-real-weight-release` incl.
`realwidth_dit_block_validation` and `real_block_adapter` block-16; 12 core
tests from `build/phase10_1-real-mmrope-release`; 4 Phase 11 Python suites in
`seedvr2-block-probe`; Phase 15 streaming adapter on range_1 and range_2).
Historical baseline JSONs regenerated by the run are restored with
`git restore` (byte-identical, same convention as Phase 13/14).

## Git protection

`reports/phase15_git_protection.json` PASS: no `.phase15_*`/`.phase14_*`
directories or heavy weight/video/image files tracked; no large binary
tracked; `.phase15_*` dirs ignored; no conflict markers; no secrets in
committed reports/docs/tests. Heavy real-weight artifacts stay in
git-ignored `.phase15_local_artifacts/`, `.phase15_export_tmp/`,
`.phase15_visual_outputs/`, `.phase15_runtime_traces/`, `.phase15_diag/`.

## Reports

- `reports/multiblock_chain_parity.json`
- `reports/multiblock_error_growth.json`
- `reports/multiblock_latency_profile.json`
- `reports/multiblock_memory_profile.json`
- `reports/multiblock_visual_comparison.json`
- `reports/phase15_block_ranges.json`, `reports/phase15_multiblock_mapping.json`
- `reports/phase15_negative_tests.json`, `reports/phase15_regressions.json`,
  `reports/phase15_git_protection.json`

## Not validated / not started

flash_attn/apex fused-CUDA kernel fidelity NOT validated (harness-injected
math-equivalent compat used identically by reference and hybrid runs). Full
32-block DiT, VAE-in-ncnn, Android/Vulkan/Pad 7 Pro NOT started. No real
checkpoint or weights committed (`large_artifacts_committed=false`).
