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
`phase12-synthetic-dit-block-ncnn`). The real `NaMMSRTransformerBlock`
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

`Phase 13 real-width single-block feasibility + weight mapping = READY` (commit on `phase13-real-width-weight-map`). The real `NaMMSRTransformerBlock` config (C=2560/H=20/HD=128, MLP 6912, seed 20260803, random weights) is exported to 19 FP32 ncnn stage models (`pnnx fp16=0`) and runs RW-A -> RW-A end-to-end on the existing `HybridDiTBlockRuntime`: all stages within Phase 13 thresholds (final vid 1.00e-5 / txt 6.4e-6, limit 2e-4), bit-identical across processes, zero NaN/Inf; first-load ~236-258 ms, cached ~31-35 ms, peak working set ~685 MB, cache 19. A full state_dict->ncnn weight mapping contract (29/29, mapped_numel==source_numel, AdaSingle runtime-only, qk-norm all-ones folded by pnnx, last-layer reuses all 19 models) is committed and validated (12 checks x5, negatives 25/25). Regressions 21/21 PASS. Estimates only: 32 blocks ~20.1 GB FP32 / ~10.0 GB FP16, per-block streaming required; Pad 7 Pro / Android / Vulkan NOT validated. Real weights, 32-block DiT, VAE/scheduler remain TODO (Phase 14+).

## Phase 14 real-weight single-block hybrid visual milestone status

`Phase 14 = READY` (branch `phase14-real-weight-visual-milestone`, base `e7e24d0`). The official `ByteDance-Seed/SeedVR2-3B` checkpoint was task-authorized and downloaded (5/5 sha256 verified, `official_source_only=true`, `mirror_used=false`), safely loaded (`weights_only=True, mmap=True, map_location=cpu`, no fallback), schema-validated (32 blocks derived, 635 fp32 keys, block 16 = floor(32/2) middle, not last layer). q/k RMSNorm forced real-affine check: NOT all-ones -> `REAL_AFFINE_REQUIRED`; Phase 13 0-byte reuse FORBIDDEN; real affine bins exported. Real-weight 15-key mapping contract PASS (`mapped_numel == source_numel`, QKV/SwiGLU bit-exact, AdaSingle runtime-only); 19-stage FP32 export `pnnx fp16=0` (git-ignored `.phase14_local_artifacts/`). RW-A real-weight parity PASS (final vid 1.812e-4 / txt 1.144e-4 < 2e-4, NaN/Inf=0, deterministic); official-context block-16 parity (vid abs 5.15e-5 / txt abs 2.44e-4, rel 1.4e-6 / 8.9e-9 < 1e-3); hybrid visual milestone PASS (adapter ncnn=1 / original=0 / no silent fallback; latent PSNR 36.3 dB, pixel PSNR 52.5 dB / SSIM 0.9975; visible outputs in git-ignored `.phase14_visual_outputs/`); negatives 29/29; regressions 22/22; git protection 8/8. See `docs/real-weight-single-block-report.md` + `docs/hybrid-visual-milestone-report.md`.

## Phase 15 real-weight multi-block streaming hybrid status

`Phase 15 = READY` (branch `phase15-multiblock-streaming-hybrid`, base `17ce482`). Blocks 15/16/17/18 run through ncnn in strict per-block streaming mode (`LOAD -> EXECUTE -> UNLOAD`, runtime destroyed, cache cleared to 0, resident bytes 0, file rename-probe passes, `max_resident_weighted_blocks=1`). Chain parity chain_1 rel 8.872e-9 (1e-3) / chain_2 2.835e-6 (2e-3) / chain_4 3.777e-6 (5e-3), NaN/Inf=0, streaming == resident bit-identical; visual 1/2/4 blocks pixel PSNR 53.65/52.72/52.90 dB (>=38), SSIM 0.9978/0.9978/0.9977 (>=0.98); memory streaming flat (474/496/476 MB) vs resident linear (446/839/1530 MB); negatives 48/48; regressions 27/27; git protection PASS.

## Phase 16 full SeedVR2 DiT streaming status

`Phase 16 = READY` (branch `phase16-full-dit-streaming`, base `a4b9a5e`). All 32 real blocks 0..31 (including the unique last layer 31) run through ncnn in strict per-block streaming (`LOAD -> EXECUTE -> UNLOAD`, `max_resident_weighted_blocks=1`); blocks 0..9 split 29-key layout, blocks 10..31 shared 15-key `.all.` layout (derived from the checkpoint); 32/32 artifact inventory (logical 13.32 GB, blocks 15-18 hardlink-reused from Phase 15); 32/32 mapping contract PASS; chain parity chain_8 5.6465e-4 (1e-2) / chain_16 2.0992e-3 (2e-2) / chain_full 5.0234e-3 (5e-2), NaN/Inf=0, 3x bit-identical; last-layer combined rel 4.05e-6 (1e-3); official-pipeline visual 8/16/full pixel PSNR 51.336/52.551/51.050 dB (>=34), SSIM 0.9974/0.9976/0.9972 (>=0.95), adapter ncnn=8/16/32 / original=0 / silent_fallback=false; error growth E1..Efull finite; negatives 80/80; regressions 27/27; git protection PASS. Gotcha: `vid_out_ada` head modulation reuses cache key `emb_repeat_0_vid` - the visual harness rebuilds it exactly when the whole chain is replaced by ncnn. See `docs/full-dit-streaming-report.md` + `docs/full-dit-error-growth-report.md`. VAE-in-ncnn, full video restoration, Android/Vulkan/Pad 7 Pro, direct fused-kernel parity = Phase 17, NOT started.
