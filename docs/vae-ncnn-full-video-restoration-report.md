# Phase 17: real-weight SeedVR2 VAE ncnn + full video restoration

Status: **READY** (branch `codex/phase17-vae-full-video-restoration`, base `87b032c`).

## Scope

- Official `ema_vae.pth` (sha256 `c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac`,
  250 fp32 tensors, 250,648,227 numel, Phase 14 download 5/5 verified, official source only)
  runs as **14 FP32 ncnn stage chains** (7 encoder + 7 decoder, `pnnx fp16=0`,
  profiles A = 4x128x128 / B = 8x192x256, git-ignored `.phase17_local_artifacts/`).
- Full official-pipeline video restoration (4 frames 128x128, seed 20260803, steps 1,
  cfg 1.0, scheduler/timestep/conditioning/pos-neg emb on the official host) with the
  VAE encoder/decoder and the full 32-block DiT routed through ncnn.

## VAE stage validation (CPU parity, official PyTorch vs ncnn FP32)

| Check | A | B | Limit |
| --- | --- | --- | --- |
| Stage parity | 14/14 stages rel max ~2.1e-6 | 14/14 | rel 1e-3 |
| Encoder end-to-end rel | 2.166e-6 | 1.897e-6 | 2e-3 |
| Decoder end-to-end rel | 2.08e-5 | 1.73e-5 | 2e-3 |
| Decoder PNG PSNR (dB) | 117.9 | 119.3 | >=45 |
| Decoder SSIM | 1.0 | 1.0 | >=0.99 |
| Full-VAE roundtrip pixel PSNR (dB) | ~116.7 | ~117.8 | >=45 |
| Roundtrip SSIM | 1.0 | 1.0 | >=0.99 |
| Roundtrip adapter calls | ncnn enc 7 / dec 14, original 1/2 | ncnn 7/14, original 2/4 | silent_fallback=false |

NaN/Inf = 0 everywhere. Weight mapping contract 250/250 (`mapped_numel == source_numel`,
no unmapped/duplicate keys, no shape mismatches, 6 deconv layout + 4 reshape conversions).
Tiling contract uses the official 4k+1 padded convention (T=9/T=17 whole-vs-sliced
equivalence max_abs < 8.4e-7; the official raw T=8 slice crash is reproduced as wrapper
behavior). Streaming lifecycle: encoder 405 MB / decoder 607 MB resident, A->B->A error 0,
cache 0, file handles released, `max_resident_vae_model_sets=1`. Latency / memory / IO
profiles PASS. Negatives 70/70; historical regressions PASS
(`reports/phase17_regressions.json`, Phase 14/15 matrices + Phase 16 CPU-light suites +
cited GPU baselines, historical baselines restored); git protection PASS.

## Full video restoration (official pipeline)

| Mode | Pixel PSNR (dB) | SSIM | Latent RMSE |
| --- | --- | --- | --- |
| `pytorch_reference` | baseline | baseline | baseline |
| `dit_ncnn_vae_pytorch` (ncnn DiT + PyTorch VAE) | 51.998 | 0.9989 | 0.0237 |
| `dit_ncnn_decoder_ncnn` (ncnn DiT + ncnn decode) | 52.105 | 0.9990 | 0.0217 |
| `full_ncnn_core` (ncnn VAE + ncnn DiT + ncnn VAE) | **51.195** | **0.9990** | 0.0206 |

Thresholds: pixel PSNR >= 40 dB, SSIM >= 0.98, latent RMSE sanity <= 0.5, NaN/Inf = 0,
frames non-constant. `full_ncnn_core` adapter evidence: `ncnn_vae_encoder_call_count=1`,
`ncnn_vae_decoder_call_count=1`, `original=0`, `silent_fallback=false`,
`chain_hybrid_ran=true`, `max_resident_weighted_dit_blocks=1`, `max_resident_vae_model_sets=1`.
Visible outputs in git-ignored `.phase17_visual_outputs/` (`00_input`,
`01_pytorch_reference`, `02_phase16_dit_ncnn_vae_pytorch`, `03_dit_ncnn_decoder_ncnn`,
`04_full_ncnn_core`, `05_diff_decoder`, `06_diff_full`, `07_side_by_side`).

Error budget (`reports/phase17_full_pipeline_error_budget.json`): encoder 2.166e-6 +
DiT 2.971e-3 + decoder 2.078e-5 -> combined pixel RMSE 2.756e-3, largest contributor = DiT.

## Harness gotcha fixed in this phase

The official VAE encoder consumes CUDA RNG while the ncnn VAE encoder runs in a
subprocess (no RNG), so the four pipeline modes previously drew **different** latent/noise
`z`. The visual harness now anchors RNG after encoding so `z` is bit-identical across all
modes. Before the fix `full_ncnn_core` measured pixel PSNR 30.82 dB purely from the `z`
mismatch; after the fix 51.19 dB (reported manifest `cond_sha256` identical across modes).

## Not validated / out of scope

- `full_pure_ncnn_application=false`: scheduler, timestep transform, conditioning and
  pos/neg embeddings stay on the official host/PyTorch side.
- VAE-in-ncnn is CPU FP32 only; no GPU/Vulkan/Android, no Pad 7 Pro.
- Direct fused-kernel parity (flash_attn/apex) NOT validated (harness CPU compat).
- Phase 18 (Android/Vulkan, fused-kernel parity, full-pure-ncnn) NOT started.

## Evidence

- `reports/phase17_*.json` (schema, io contract, stage/encoder/decoder/roundtrip parity,
  tiling, streaming lifecycle, wrapper equivalence, weight mapping + validation,
  artifact inventory, latency/memory/io profiles, negative tests, regressions,
  full pipeline visual + error budget, git protection)
- `tests/seedvr2_vae_ncnn/` (runtime, exporters, validators, pipeline, protection checks)
- `docs/TEST_MATRIX.md` Phase 17 section (commands + acceptance)
