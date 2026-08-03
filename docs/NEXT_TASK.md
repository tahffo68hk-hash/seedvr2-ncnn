# Next task prompt

Phase 14 real-weight single-block hybrid visual milestone is **BLOCKED_CHECKPOINT_REQUIRED** (branch `codex/phase14-real-weight-visual-milestone`, base `9b1e909`). Prep committed: `.gitignore` Phase 14 rules; `tests/seedvr2_real_weight_hybrid/` (checkpoint acquisition manifest, expected filenames, sha256 fields, non-downloading loader preflight, tests 11/11 PASS); `reports/phase14_checkpoint_preflight.json` (BLOCKED). No real checkpoint, no download, no random-weight substitute.

Single input required to continue:

```text
Place the official ByteDance-Seed/SeedVR2-3B checkpoint at
D:/AIProjects/seedvr2-ncnn/.phase14_local_checkpoint/seedvr2_ema_3b.pth
and record its sha256 (or explicitly authorize the official download).
```

When the checkpoint is present, continue the full Phase 14 task: validate the checkpoint schema against the Phase 13 weight-mapping contract (key names, shapes, numel, dtype, all-ones qk-norm expectation, AdaSingle runtime-only semantics); select middle block `floor(N/2)`; export real-weight 19 FP32 ncnn stage models with `pnnx fp16=0` into the git-ignored `.phase14_local_artifacts/`; run real-context block parity (thresholds: norm/ada-in <1e-5, qkv/qk-norm/rope <2e-5, packed attention/projection <1e-4, SwiGLU/residuals/finals <2e-4, NaN/Inf=0); build the single-block hybrid adapter (`--mode capture|pytorch|ncnn`, `ncnn_adapter_call_count>0`, `original_block_call_count==0`, no silent fallback); generate a visible output milestone (input / PyTorch reference / hybrid ncnn / abs-diff / side-by-side under `.phase14_visual_outputs/`); compare latent, pixel, PSNR/SSIM; run Phase 14 negatives and Phase 13/12/core/Phase 11 regressions; update reports and docs. Do not start 32-block DiT, Android or Vulkan until explicitly authorized. READY requires executed evidence with recorded thresholds.
