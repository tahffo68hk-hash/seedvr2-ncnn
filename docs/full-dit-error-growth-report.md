# Full DiT error growth report (Phase 16)

Status: **PASS**. FP32 streaming ncnn replaces 8/16/all-32 contiguous real-weight
SeedVR2 blocks inside the official pipeline; the chain-level relative error
stays below every per-chain limit with zero NaN/Inf, and the full-DiT pixel
output remains visually faithful (PSNR 51.05 dB / SSIM 0.9972).

## Data

`reports/full_dit_error_growth.json` (latent and pixel-space, 4 frames
128x128, seed 20260803, steps 1, cfg 1.0):

| x replaced | blocks | latent RMSE | latent PSNR (dB) | pixel RMSE | pixel PSNR (dB) | pixel SSIM |
|------------|--------|-------------|------------------|------------|-----------------|------------|
| 1  | 16 | 0.01286 | 37.817 | 0.002078 | 53.648 | 0.99783 |
| 2  | 16,17 | 0.01159 | 38.718 | 0.002312 | 52.720 | 0.99775 |
| 4  | 15..18 | 0.01424 | 36.929 | 0.002264 | 52.901 | 0.99770 |
| 8  | 9..16 | 0.02209 | 33.116 | 0.002711 | 51.336 | 0.99740 |
| 16 | 8..23 | 0.01561 | 36.132 | 0.002357 | 52.551 | 0.99763 |
| 32 | 0..31 | 0.02373 | 32.495 | 0.002802 | 51.050 | 0.99724 |

E1/E2/E4 reuse the Phase 15 `multiblock_visual_comparison.json` (same
anchor blocks around 16); E8/E16/EFull are measured in this phase.  The
full-chain pixel thresholds PSNR >= 34 dB and SSIM >= 0.95 are satisfied
(`overall_pass=true`).

## Chain parity (streaming vs FP32 reference)

`reports/phase16_chain_{8,16,full}_parity.json` + summary:

| chain | blocks | final rel error | limit | max_abs | RMSE | NaN/Inf | repeats |
|-------|--------|-----------------|-------|---------|------|---------|---------|
| chain_8 | 9..16 | 5.6465e-4 | 1e-2 | 15.541 | 0.062064 | 0 | 3 (bit-identical) |
| chain_16 | 8..23 | 2.0992e-3 | 2e-2 | 57.391 | 0.129464 | 0 | 3 (bit-identical) |
| chain_full | 0..31 | 5.0234e-3 | 5e-2 | 279.104 | 0.644173 | 0 | 3 (bit-identical) |

Per-block relative error ranges: chain_8 `5.64e-4..5.66e-4` (blocks
9..16), chain_16 `6.92e-4..2.10e-3` (blocks 8..23), chain_full
`2.83e-4..7.28e-3` (blocks 0..31).  All per-block rows NaN=0, Inf=0,
pass=true.  The 1e-3 single-block budget is the Phase 14/15 criterion;
the documented chain limits scale with replaced-block count (1e-2 / 2e-2 /
5e-2) and are all satisfied by a wide margin.

## Error growth analysis

- The pixel-space curve is essentially flat (PSNR 51.05..53.65 dB, SSIM
  0.9972..0.9978) regardless of how many blocks are replaced, because
  pixel space is far downstream of the DiT body.
- Latent-space RMSE is not monotonic in chain length (E2 0.0116 < E1 0.0129,
  E16 0.0156 < E8 0.0221).  The error is dominated by the blocks
  surrounding the Phase 14/15 anchor (block 16) and by the last-layer head
  path rather than by the raw number of replaced blocks.
- The full-chain relative error 5.02e-3 is an order of magnitude inside the
  5e-2 limit; absolute error magnitude grows with chain length because each
  hybrid block consumes the previous hybrid output by design
  (`chained_upstream_hybrid_input`).

## Visual impact

- All modes: frames non-constant (reference and hybrid variance intact),
  NaN/Inf = 0, `order_ok=true`, `streaming_mode_validated=true`,
  `max_resident_weighted_blocks=1`, `state_machine_final=IDLE`,
  `original_block_call_count=0`, `silent_fallback=false`.
- Side-by-side panels `Input | PyTorch | Hybrid ncnn | Amplified
  difference` (amplify factor 25) under `.phase16_visual_outputs/`
  (08_side_by_side_full; 09_error_growth chart
  `error_growth.png`).

## Conclusion

Error growth from 1-block to full-32-block ncnn streaming is bounded and
far inside every documented chain limit.  Full-DiT streaming is
visually faithful (>= 51 dB PSNR, >= 0.997 SSIM) and numerically
deterministic; the residual is dominated by the block-16 anchor and the
last-layer head, not by chain length.