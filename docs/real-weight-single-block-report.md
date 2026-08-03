# Real-Weight SeedVR2 Single-Block Report (Phase 14)

Status: **READY** (branch `codex/phase14-real-weight-visual-milestone`, base `e7e24d0`).
Scope: acquire and safely load the official `ByteDance-Seed/SeedVR2-3B` checkpoint,
validate the schema against the Phase 13 mapping contract, force real affine q/k
RMSNorm (Phase 13 0-byte reuse FORBIDDEN), select the middle DiT block, stream-export
the real-weight FP32 stages (`pnnx fp16=0`), run real-context single-block parity on
the existing `HybridDiTBlockRuntime`, and prove the adapter evidence with no silent
fallback. 32-block DiT, Android and Vulkan are NOT started.

## 1. Provenance

- Repo branch: `codex/phase14-real-weight-visual-milestone`; HEAD base `e7e24d0`.
- Checkpoint: official `ByteDance-Seed/SeedVR2-3B` (`huggingface.co`), `official_source_only=true`,
  `mirror_used=false`, `login_required=false`, `token_used=false`.
- Main checkpoint SHA-256: `6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b`
  (13,566,090,228 bytes, sha256_verified).
- 5/5 files verified (`seedvr2_ema_3b.pth`, `ema_vae.pth`, `pos_emb.pt`, `neg_emb.pt`,
  `README.md`); recorded in `reports/phase14_checkpoint_download.json` (no absolute
  paths, no tokens).
- Local landing: git-ignored `.phase14_local_checkpoint/` (never committed).
- SeedVR2 source: `e4de8c2` (`D:\AIProjects\third_party\SeedVR2`, unmodified).
- Load path: `torch.load(map_location="cpu", weights_only=True, mmap=True)`;
  `succeeded=true`, `fallback_used=false` (see `reports/real_checkpoint_schema.json`).

## 2. Checkpoint schema (derived, not hardcoded)

- Top-level type: `dict`; flat key count 635; tensor count 635; dtype distribution
  `{torch.float32: 635}`; total numel 3,391,476,448.
- File is the EMA weights copy (flat unprefixed module paths, no `ema.` prefix).
- DiT block count **derived** from keys: 32 (`block_indices` 0..31), prefix `blocks.<i>.`.
- Target block: `block_index = floor(32/2) = 16`, `is_last_layer=false`, prefix
  `blocks.16.`, 15 keys, numel 79,316,757.
- Components: DiT transformer blocks only; VAE weights live in the separate
  `ema_vae.pth` artifact (not inside `seedvr2_ema_3b.pth`).

## 3. Forced q/k RMSNorm real-affine check

Checked `blocks.16.attn.norm_q.all.weight` and `blocks.16.attn.norm_k.all.weight`
(and block-0 vid/txt variants for the shared/unshared comparison):

| key | min | max | mean | std | all_ones | sha256 |
|---|---|---|---|---|---|---|
| blocks.16.norm_q.all | 0.7623 | 1.6978 | 1.4038 | 0.1902 | false | 99c5742e... |
| blocks.16.norm_k.all | 0.7659 | 1.7889 | 1.4273 | 0.1880 | false | 02cae8eb... |

Conclusion: `REAL_AFFINE_REQUIRED`; `phase13_zero_byte_bins_reuse = FORBIDDEN`.
The real q/k RMSNorm affine weights were exported as real FP32 bins (512 bytes each),
never the Phase 13 all-ones folded 0-byte bins.

## 4. Real-weight 29/15-key mapping contract

`reports/real_weight_block_mapping.json` (schema_version 2, variant
`real_seedvr2_block16_c2560_h20_hd128`):

- `source_key_count = 15`, `mapped_key_count = 15`, `mapped_numel == source_numel`
  (79,316,757).
- `expected_keys_equals_consumed_keys = true`, `unmapped_keys = []`,
  `duplicate_keys = []`, `shape_mismatches = []`.
- QKV split/concat bit-exact: `[3C, C]` rows `q=[0,C) k=[C,2C) v=[2C,3C)`.
- SwiGLU mapping: `proj_in_gate -> swiglu_gate`, `proj_in -> swiglu_up`,
  `proj_out -> swiglu_out`, all bias-free, `[out, in]` ncnn layout (no transpose).
- AdaSingle: real learned shift/scale/gate parameters written to runtime-only
  `params/ada_{side}_{layer}_{part}.bin` (12 bins), math unchanged
  `hid*(scaleA+scaleB)+(shiftA+shiftB)` / `hid*(gateA+gateB)`.
- `validation.overall = true`.

## 5. Real-weight FP32 staged export

- 19-stage export (`pnnx fp16=0`) into git-ignored
  `.phase14_local_artifacts/real_seedvr2_block16_c2560_h20_hd128/`; 12 unique FP32
  stage models (`attn_norm, qkv, q_norm, k_norm, rope_apply, attn_proj, mlp_norm,
  swiglu_gate, swiglu_up, swiglu_act, swiglu_out, head`) + 12 AdaSingle params bins.
- Phase 13 random-weight artifacts are untouched; TorchScript/pnnx intermediates
  cleaned; real-weight bins never enter Git.

## 6. RW-A real-weight parity (O(1) fixture, absolute limit 2e-4)

`reports/real_weight_block_parity_rwa.json` (`phase14_verdict = PASS`):

- final vid max_abs_error `1.812e-4`, final txt max_abs_error `1.144e-4` (< 2e-4).
- `all_stages_max_abs_below_2e4 = true`, NaN=0, Inf=0, `runs_bit_identical = true`
  (A,A), determinism across forwards.
- First-load init ~117 ms, cached forward ~35 ms; working set 311-379 MB; model
  cache 12 (shared block: vid/txt reuse one physical artifact).

## 7. Negatives, regressions, git protection

- Phase 14 negatives: 29/29 PASS (`reports/phase14_negative_tests.json`) - preflight
  missing/bad-sha/wrong-name/too-small, mapping missing/extra/duplicate/shape/numel,
  QKV order, SwiGLU swap, qk-norm 0-byte reuse, non-middle/last-layer block, adapter
  missing variant/root/inputs/bad shapes, report NaN/Inf, silent-fallback, seed-guard,
  milestone verdict/numeric fields, git tracking, secrets.
- Historical regressions: 22/22 PASS (`reports/phase14_regressions.json`) - Phase 12/13
  staged targets from the Phase 14 build (6), core dispatcher (12), Phase 11 Python
  suites (4); tracked historical baselines restored byte-identical.
- Git protection: 8/8 PASS (`reports/phase14_git_protection.json`) - no `.phase14_*` /
  `.phase13_*` / build / third_party / conda / weight files / >10 MiB files tracked,
  local dirs ignored, no conflict markers, no secrets, Phase 14 JSON finite.

## 8. Flags

```
real_seedvr2_weights=true single_ncnn_block=true full_32_block_dit=false
full_pipeline_pytorch=true visible_output_generated=true android_vulkan=false
checkpoint_downloaded_by_task=true large_artifacts_committed=false
overall_status=PASS (per documented thresholds, see hybrid-visual-milestone-report.md)
```
