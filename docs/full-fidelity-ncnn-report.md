# Full-fidelity ncnn status

`Single graph full fidelity = BLOCKED`: the pre-existing vectorized 4-D merge fails at blob 89, the per-head graph Concat fails at `cat_0`, and the contribution-Gemm single graph has no usable final output. Those artifacts and reports are retained as failure evidence.

`Hybrid staged full fidelity = READY`: the ArtifactRegistry-driven Release
runtime executes four 2-D static head nets, explicit host merge, one ncnn 2-D
output Gemm, and reverse/crop. Its final BTHWC maximum absolute error was
`1.49012e-07`, with mean `2.10550e-08`, zero NaN, and zero Inf values.

The first attempted tuple-head core is also preserved under the hybrid artifact's legacy `attention_heads/model.ncnn.*`; extracting `out0` returned shape `[64,64,8]` (32,768 elements), while its required shape is `[4,64,8]` (2,048 elements). The ready path replaces that invalid rank-3 batched layout with four valid static 2-D head artifacts.

Phase 8 adds a separate mathematical RMSNorm path. Its standalone ncnn
fixture passes for the sqrt/reciprocal, pow, and explicit rewrites, but the
complete RMS hybrid head currently exceeds the `1e-4` gate; it is not promoted
to READY and does not change the LayerNorm result above.

## Phase 9.4 metadata/runtime routing

`Phase 9.4 metadata/runtime routing = READY`: fixed LayerNorm, fixed RMSNorm,
and Dynamic A select their typed metadata routes without cross-profile fallback.
Dynamic A uses natural `valid_tokens=64`; fixed RMS retains the artificial
fidelity override `valid_tokens=60`. Phase 9 multi-shape validation remains
`PARTIAL` because Dynamic B/C are not implemented.

## Phase 9.5 multi-shape status

`Phase 9.5 Dynamic multi-shape RMS Hybrid Runtime = READY`. Dynamic A/B/C use one
metadata-driven route with grids `[1,2,2]`, `[1,3,3]`, `[2,3,3]`; their BTHWC
maximum errors are `1.49012e-7`, `1.49012e-7`, and `2.08616e-7`. This remains a
CPU FP32 fixture subsystem, not complete SeedVR2 inference.

## Phase 10 head_dim=128 status

`Phase 11 real single DiT block contract = READY`. The real
`NaMMSRTransformerBlock` (SeedVR2 `e4de8c2`) was extracted with source/AST
evidence (call chain, token layout, `vid_freqs[l:l+f]` RoPE offset, AdaSingle
modulation, NaSwinAttention, SwiGLU MLP, residual order; see
`seedvr2-single-dit-block-contract-report.md`) and its real random-weight
forward was EXECUTED in the isolated conda env `seedvr2-block-probe` at full
2560 width (profiles A/B/C/D, runtime-hook contract evidence, negatives 17/17,
Phase 10 regressions PASS; flash_attn/apex fused-CUDA fidelity not validated -
test-harness CPU compat only). ncnn single DiT block execution remains TODO
(Phase 12).

`Phase 10 head_dim=128 = READY`. The real SeedVR2 MM-RoPE3D contract was
confirmed from `rotary-embedding-torch==0.5.3` (SeedVR2 commit `e4de8c2`,
pinned in requirements.txt:13 / environment.yml:181): SeedVR2 configures
rope_dim=128 and builds `RotaryEmbedding(dim=128//3=42, freqs_for="lang",
theta=10000)`, giving effective rotary dim 126 (axis_dims=[42,42,42]), identity
tail 2, T/H/W axis-major pairs, W-fastest tokens, per-axis freq
`theta^(-k/21)`. An upstream tensor oracle (oracle A/B/C) reproduces the exact
package, and the real-contract variant `seedvr2_hd128_mmrope3d_hybrid` passes
S/M/P end-to-end on CPU FP32 (cos/sin `5.96e-8`, heads/merge/projection/BTHWC
`5.4e-7..1.79e-6`, validity/mask exact, S->M->P->S bit-identical, negatives
22/22, cache 15) plus all Phase 7.5-9.5 and Phase 10 synthetic regressions.
The synthetic fixture remains byte-identical as a historical regression. Real
weights, complete DiT block, Vulkan, and VAE remain TODO. See
[real-seedvr2-mmrope-contract-report.md](real-seedvr2-mmrope-contract-report.md)
and [synthetic-hd128-runtime-report.md](synthetic-hd128-runtime-report.md).

`Phase 12 synthetic single DiT block staged fidelity = READY` (commit on
`codex/phase12-synthetic-dit-block-ncnn`). The real `NaMMSRTransformerBlock`
topology is executed as a scaled C=512/H=4/HD=128 staged ncnn + host hybrid
runtime (`HybridDiTBlockRuntime`, seed 20260802, MLP hidden 1536 from the real
SwiGLUMLP formula). Profiles A/B/C/D all PASS: final vid/txt max errors ~4.5e-6
(< 1e-4), A->B->C->D->A bit-identical; packed q/k/v, MM-RoPE3D, `[vid_win; txt]`
packing, txt repeat/mean merge, AdaSingle, SwiGLU, residuals and the last-layer
vid_only double-residual semantics all validated stage-by-stage (registry/cache
19; negatives 24+7+registry; Phase 9.5/10/11 regressions PASS). Batch>1 txt
indexing of the real `repeat_concat_idx` is reproduced exactly (q/k reuse
repeated-txt rows `[cum_b, cum_b+l_b)`; v uses the original txt). `ada_*_in`
threshold 1e-5 documented (reference-internal recompute bit-exact). Real
weights, full 2560-width ncnn block, Vulkan and VAE remain Phase 13 scope. See
[synthetic-seedvr2-dit-block-ncnn-report.md](synthetic-seedvr2-dit-block-ncnn-report.md)
and `reports/synthetic_dit_block_ncnn_validation.json`.

## Phase 13 real-width status

`Phase 13 real-width single-block feasibility + weight mapping = READY` (commit on `codex/phase13-real-width-weight-map`). The real `NaMMSRTransformerBlock` config (C=2560/H=20/HD=128, MLP 6912, seed 20260803, random weights) is exported to 19 FP32 ncnn stage models (`pnnx fp16=0`) and runs RW-A -> RW-A end-to-end on the existing `HybridDiTBlockRuntime`: all stages within Phase 13 thresholds (final vid 1.00e-5 / txt 6.4e-6, limit 2e-4), bit-identical across processes, zero NaN/Inf; first-load ~236-258 ms, cached ~31-35 ms, peak working set ~685 MB, cache 19. A full state_dict->ncnn weight mapping contract (29/29, mapped_numel==source_numel, AdaSingle runtime-only, qk-norm all-ones folded by pnnx, last-layer reuses all 19 models) is committed and validated (12 checks x5, negatives 25/25). Regressions 21/21 PASS. Estimates only: 32 blocks ~20.1 GB FP32 / ~10.0 GB FP16, per-block streaming required; Pad 7 Pro / Android / Vulkan NOT validated. Real weights, 32-block DiT, VAE/scheduler remain TODO (Phase 14+).

## Phase 14 real-weight single-block hybrid visual milestone status

`Phase 14 = BLOCKED_CHECKPOINT_REQUIRED` (branch `codex/phase14-real-weight-visual-milestone`, base `9b1e909`). The real SeedVR2 3B checkpoint is not present locally and this task does not authorize download, login, license acceptance, or unknown-source weights; random weights cannot upgrade the phase. Official expected input: `ByteDance-Seed/SeedVR2-3B` -> `seedvr2_ema_3b.pth` (DiT, `projects/inference_seedvr2_3b.py`) and `ema_vae.pth` (VAE). Committed prep: `.gitignore` Phase 14 rules; `tests/seedvr2_real_weight_hybrid/` (acquisition manifest, expected filenames, sha256 fields, non-downloading loader preflight, tests 11/11 PASS); `reports/phase14_checkpoint_preflight.json` (BLOCKED, `download_attempted=false`). Single input to continue: checkpoint at `.phase14_local_checkpoint/seedvr2_ema_3b.pth` + sha256 (or explicit download authorization). Then: schema validation vs the Phase 13 mapping contract (real qk-norm all-ones gate), real-weight 19-stage FP32 export (`pnnx fp16=0`), real-context block parity, single-block hybrid adapter (no silent fallback), and a visible output milestone under `.phase14_visual_outputs/`. 32-block DiT, Android/Vulkan not started.
