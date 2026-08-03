# Phase 14: Real-Weight Single-Block Hybrid Visual Milestone

Status: **READY** (2026-08-03, official checkpoint verified and milestone executed).

## Scope

Validate the real SeedVR2 3B attention block on ncnn: acquire the official
`ByteDance-Seed/SeedVR2-3B` checkpoint (task-authorized download, official HF
repo only), safely load and schema-probe it, force real affine q/k RMSNorm,
map the real weights through the state_dict->ncnn contract, export 19 FP32
stages (`pnnx fp16=0`), run RW-A real-weight parity and an official-pipeline
hybrid visual milestone with one middle block replaced by the ncnn adapter.

## Checkpoint

```text
repo_id: ByteDance-Seed/SeedVR2-3B (huggingface.co)
official_source_only=true mirror_used=false login_required=false token_used=false
files (5/5 sha256 verified): seedvr2_ema_3b.pth (6bcc5ac5...), ema_vae.pth,
pos_emb.pt, neg_emb.pt, README.md
main checkpoint SHA-256: 6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b
local landing: git-ignored .phase14_local_checkpoint/ (never committed)
```

Details: `reports/phase14_checkpoint_download.json` (sizes + sha + source +
verification, no absolute paths, no tokens); preflight `reports/phase14_checkpoint_preflight.json`
(READY, exit 0).

## Key tools (tests/seedvr2_real_weight_hybrid/)

- `download_official_checkpoint.py` - official-only downloader (SHA-verified,
  resume support, no mirrors).
- `checkpoint_loader_preflight.py` - stdlib-only non-downloading preflight
  (exit 0 READY / 2 BLOCKED / 3 INVALID / 4 SHA_MISMATCH).
- `update_preflight_status.py` - records the 5-file manifest and flips the
  tracked status to CHECKPOINT_AVAILABLE / download report.
- `real_checkpoint_schema.py` - safe load (`weights_only=True, mmap=True`) +
  schema + forced q/k RMSNorm all-ones check.
- `real_weight_block_mapping.py` - 15-key real-weight mapping contract.
- `export_real_weight_block_stages.py` - 19-stage FP32 streaming export
  (`pnnx fp16=0`) into git-ignored `.phase14_local_artifacts/`.
- `run_official_pipeline.py` - official pipeline driver
  (`--mode reference|hybrid`, `--milestone-only`); block 16 replaced by
  `real_block_adapter.exe`; writes `.phase14_visual_outputs/` and
  `reports/hybrid_visual_milestone.json`.
- `phase14_negative_tests.py` - 29 controlled-failure negatives.
- `run_phase14_regressions.py` - 22 historical regression targets.
- `phase14_git_protection.py` - 8 git protection checks.

## Results

- Schema: 635 fp32 keys, 32 blocks (derived), block 16 = floor(32/2) middle,
  not last layer; `reports/real_checkpoint_schema.json`.
- q/k RMSNorm: NOT all-ones -> `REAL_AFFINE_REQUIRED`, Phase 13 0-byte reuse
  FORBIDDEN, real affine bins exported.
- Mapping contract: 15/15, mapped_numel == source_numel, QKV/SwiGLU bit-exact,
  AdaSingle runtime-only; `reports/real_weight_block_mapping.json`.
- RW-A parity: final vid 1.812e-4 / txt 1.144e-4 (< 2e-4), NaN/Inf=0,
  bit-identical; `reports/real_weight_block_parity_rwa.json` (`phase14_verdict=PASS`).
- Milestone: block-16 official-context parity PASS (rel < 1e-3; absolute 2e-4
  documented for the O(1) RW-A fixture), adapter ncnn=1/original=0/no silent
  fallback, pixel PSNR 52.5 dB / SSIM 0.9975; `reports/hybrid_visual_milestone.json`.
- Negatives 29/29, regressions 22/22, git protection 8/8.

## Constraints honored

No real checkpoint/weights in Git; no faked results; no 32-block DiT; no
Android/Vulkan; no push/tag/PR; Phase 15 not started. See
`docs/real-weight-single-block-report.md` and `docs/hybrid-visual-milestone-report.md`.
