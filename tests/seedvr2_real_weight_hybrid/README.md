# Phase 14: Real-Weight Single-Block Hybrid Visual Milestone

Status: **BLOCKED_CHECKPOINT_REQUIRED** (2026-08-03).

## Why blocked

The task requires the real SeedVR2 3B checkpoint as the single input
(`<LOCAL_SEEDVR2_CHECKPOINT>`). This repo has **no local checkpoint** and this
task **does not authorize** downloading, logging in to Hugging Face /
ModelScope, accepting a model license, using unofficial mirrors, or using
unknown-source weights. Random weights are **not** an acceptable substitute for
Phase 14 readiness.

Official expected source and files (from `D:\AIProjects\third_party\SeedVR2`,
commit `e4de8c24441a67e1b7df56abea10645059bb1185`):

```text
repo_id: ByteDance-Seed/SeedVR2-3B (huggingface.co)
official ref: projects/inference_seedvr2_3b.py
  configure_dit_model(device="cuda", checkpoint="./ckpts/seedvr2_ema_3b.pth")
expected filenames: seedvr2_ema_3b.pth (DiT), ema_vae.pth (VAE, configs_3b/main.yaml)
code license: Apache-2.0; model license: must be accepted on the HF model card (NOT authorized here)
```

## What is prepared (committed)

- `checkpoint_acquisition_manifest.json` - expected filenames, sha256 fields
  (empty until provided), source/provenance and license notes.
- `checkpoint_loader_preflight.py` - stdlib-only, non-downloading loader
  preflight (path exists / regular file / expected filename / size floor /
  sha256 match; exit codes 0 READY, 2 BLOCKED, 3 INVALID, 4 SHA_MISMATCH).
- `test_checkpoint_preflight.py` - controlled-failure preflight tests
  (missing path, directory, wrong name, too small, sha mismatch, valid match,
  no-network source audit, manifest consistency).
- `reports/phase14_checkpoint_preflight.json` - executed preflight result
  (`BLOCKED_CHECKPOINT_REQUIRED`, no checkpoint, zero sha, no download).

## Single input needed to continue

```text
Place the official ByteDance-Seed/SeedVR2-3B checkpoint at
D:\AIProjects\seedvr2-ncnn\.phase14_local_checkpoint\seedvr2_ema_3b.pth
and record its sha256 (or explicitly authorize the official download).
```

## Plan once the checkpoint lands (Phase 14 full execution)

- Validate checkpoint schema against the Phase 13 weight-mapping contract
  (key names, shapes, numel, dtype, all-ones qk-norm expectation,
  AdaSingle runtime-only). Blocking check: real qk-norm weights must be
  all-ones or the 0-byte folded bins must not be reused.
- Select block `floor(N/2)` (middle block, `--block-index` override), export
  19 FP32 ncnn stage models with `pnnx fp16=0` into the git-ignored
  `.phase14_local_artifacts/`.
- Adapter (`tests/seedvr2_real_weight_hybrid/real_block_adapter.py`):
  `--mode capture|pytorch|ncnn`; ncnn mode must print `HYBRID BLOCK EXECUTED`
  and enforce `ncnn_adapter_call_count > 0`, `original_block_call_count == 0`
  (no silent fallback to PyTorch).
- Visual milestone: synthetic or licensed short input video -> official PyTorch
  pipeline with one block replaced -> VAE decode -> PNG/MP4 outputs in
  `.phase14_visual_outputs/` (`01_input`, `02_pytorch_reference`,
  `03_hybrid_one_block_ncnn`, `04_abs_diff`, `05_side_by_side`).
- Reports: `reports/real_checkpoint_schema.json`,
  `reports/real_weight_block_mapping.json`, `reports/real_weight_block_parity.json`,
  `reports/hybrid_visual_milestone.json`, `reports/phase14_negative_tests.json`,
  `reports/phase14_git_protection.json`, `docs/real-weight-single-block-report.md`,
  `docs/hybrid-visual-milestone-report.md`.

Constraints honored in this phase: no download, no faked results, no random
weight upgrade path, no 32-block DiT, no Android/Vulkan, no push/tag/PR.
