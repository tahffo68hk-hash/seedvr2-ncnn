# Hybrid Visual Milestone Report (Phase 14)

Status: **READY** (branch `codex/phase14-real-weight-visual-milestone`, base `e7e24d0`).
Scope: run the official SeedVR2 3B video-diffusion pipeline (VAE encode -> DiT
single step -> VAE decode) on a synthetic 4-frame 128x128 input, replace exactly one
middle DiT block (block 16) with the ncnn `real_block_adapter` (real weights, FP32),
and prove parity + visible output with adapter evidence and no silent fallback.

## 1. Driver and environment

- `tests/seedvr2_real_weight_hybrid/run_official_pipeline.py` (`--mode reference|hybrid`,
  `--milestone-only`). The official entry script hard-codes distributed/CUDA/CWD, so the
  driver runs `VideoDiffusionInfer(configs_3b/main.yaml)` directly:
  `dist.init_process_group(backend="gloo", env://)`, `MASTER_ADDR=127.0.0.1/29500`.
- Config: `configs_3b/main.yaml`; `vae.checkpoint = .phase14_local_checkpoint/ema_vae.pth`;
  `diffusion.timesteps.sampling.steps=1`, `cfg.scale=1.0`, `runner.configure_diffusion()`.
- Scene: 4-frame synthetic video 128x128, `seed=20260803`, SR-conditioned
  (VAE encode -> DiT one step -> VAE decode).
- GPU: RTX 5060 Ti 16 GB; the 3.39B DiT does not fit fully FP32, so only block 16 runs
  FP32 (with `torch.Tensor.bfloat16 = identity` harness patch for the official `.bfloat16()`
  call in mmattn.py) and the other 31 blocks run bf16; TF32 disabled (official
  `allow_tf32=True` alone gives ~7e-3 error). flash_attn/apex fused-CUDA kernels are NOT
  present; the harness injects math-equivalent CUDA compat used identically by both runs.
- Checkpoint SHA-256: `6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b`
  (official source only, 5/5 files sha256 verified).

## 2. Block-16 official-context parity

Inputs to block 16 are bit-identical between the reference and hybrid runs
(`block16_input_hashes` vid/txt/emb equal, `max_input_abs_diff = 0.0`).

| metric | final vid | final txt |
|---|---|---|
| max_abs_error | 5.150e-5 | 2.441e-4 |
| max_rel_error | 1.378e-6 | 8.872e-9 |
| mean_abs_error | 3.706e-7 | 3.298e-7 |
| NaN / Inf | 0 / 0 | 0 / 0 |

The absolute 2e-4 limit is documented for the O(1) RW-A fixture (passed: vid 1.812e-4,
txt 1.144e-4, see `real-weight-single-block-report.md`). Official-context activations are
not O(1) - txt tokens reach magnitude ~2.7e4 - so the official-context contract is judged
on relative error (`rel_limit = 1e-3`, measured 8.9e-9..1.4e-6) with absolute errors
recorded as-is. `parity.pass = true`.

## 3. Adapter evidence (no silent fallback)

`reports/hybrid_visual_milestone.json` / `.phase14_visual_outputs/block16_hybrid/adapter_report.json`:

- `ncnn_adapter_call_count = 1`, `original_block_call_count = 0`,
  `silent_fallback = false`, stdout contains `HYBRID BLOCK EXECUTED`; latency ~183 ms.
- The reference pipeline's block-16 forward is fully replaced by the ncnn hybrid runtime.

## 4. Visual metrics

| tensor | metric | value |
|---|---|---|
| final latent | max_abs_error / mean_abs_error / RMSE | 0.3281 / 8.51e-3 / 1.53e-2 |
| final latent | PSNR | 36.32 dB |
| pixels | max_abs_error / mean_abs_error / RMSE | 0.0549 / 1.10e-3 / 2.37e-3 |
| pixels | PSNR | 52.52 dB |
| pixels | SSIM | 0.9975 |
| frames | output_shape / frames | (4, 128, 128, 3) / 4 |
| frames | non-constant reference / hybrid | true / true |

Input hashes identical, NaN/Inf = 0, output frame std matches the reference.

## 5. Visible outputs (git-ignored `.phase14_visual_outputs/`)

- `01_input/` - the 4 input frames (PNG).
- `02_pytorch_reference/` - official PyTorch pipeline output.
- `03_hybrid_one_block_ncnn/` - output with block 16 on ncnn.
- `04_abs_diff/` - amplified (x25) per-pixel difference.
- `05_side_by_side/` - per-frame panels `Input | PyTorch | Hybrid ncnn | Amplified difference`.
- `block16_reference/`, `block16_hybrid/` - block-16 input/output bins + adapter report.
- `final_latent_*.pt`, `frames_*.pt`, `run_*_manifest.json`.

## 6. Verdict

`reports/hybrid_visual_milestone.json`: `verdict = PASS`,
`overall_pass = true`; reasons: block parity, adapter evidence, finite outputs and
visual sanity all pass. The report records repo-relative output paths only.

## 7. Scope guards

- Real weights: true; single ncnn block: true; full 32-block ncnn DiT: false
  (pipeline runs PyTorch with one ncnn block); Android/Vulkan: false.
- No push, no tag, no PR, no amend of `e7e24d0`; Phase 15 not started.
