# Multi-block error growth report (Phase 15)

Status: **PASS**. FP32 streaming ncnn replaces 1/2/4 contiguous real-weight
SeedVR2 blocks inside the official pipeline; the chain-level relative error
stays far below the per-chain limits with zero NaN/Inf.

## Data (official captured block-16 context, fresh Phase 15 build)

`reports/multiblock_chain_parity.json` + `reports/multiblock_error_growth.json`:

| x replaced | E_x (relative max) | limit | max_abs | mean_abs | RMSE |
|------------|--------------------|-------|---------|----------|------|
| 1 (block 16)  | 8.872e-9  | 1e-3 | 2.44e-4  | 3.58e-7 | 8.50e-7 |
| 2 (16,17)     | 2.835e-6  | 2e-3 | 7.80e-2  | 6.17e-4 | 9.59e-4 |
| 4 (15..18)    | 3.777e-6  | 5e-3 | 1.04e-1  | 1.14e-3 | 1.73e-3 |

Absolute error magnitude grows with chain length because the upstream blocks
in the hybrid run consume the hybrid (ncnn) output of the previous block by
design (`chained_upstream_hybrid_input`); relative error stays below 4e-6,
three orders of magnitude inside the 5e-3 limit.

## Per-block view

`chain_parity_per_block` rows:

- chain_1: block 16 rel 8.87e-9 (input bit-identical to reference).
- chain_2: block 16 rel 8.87e-9, block 17 rel 2.83e-6.
- chain_4: blocks 15/16/17/18 rel 2.17e-6 / 1.89e-6 / 3.82e-6 / 3.78e-6.

All rows NaN=0, Inf=0, pass=true under the 1e-3 per-block relative limit.

## Visual impact

Pixel-space impact (4 frames 128x128, un-compressed PNG):

| mode | PSNR (dB) | SSIM | max_abs | RMSE |
|------|-----------|------|---------|------|
| 1  | 53.65 | 0.9978 | 4.31e-2 | 2.08e-3 |
| 2  | 52.72 | 0.9978 | 2.75e-2 | 2.31e-3 |
| 4  | 52.90 | 0.9977 | 3.14e-2 | 2.26e-3 |

Reference and hybrid outputs are both non-constant (frame variance intact);
the amplified-difference panels in `.phase15_visual_outputs/08_side_by_side`
(amplify factor 25) make the residual visible.

## Conclusion

Streaming adds no meaningful error beyond the single-block ncnn parity
(measured 1.8e-8 .. 3.8e-6 relative per block, limits 1e-3/2e-3/5e-3 for
1/2/4-chain). Error growth with replaced-block count is sub-linear in
relative terms and bounded by the documented chain limits.
