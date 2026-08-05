# Phase 18C.4 `dec_up1` optimization

## Result

`dec_up1` is complete on Xiaomi Pad 7 Pro and passes Host/Pad FP32 parity. It can be run directly from the persisted `dec_up0` checkpoint without encoder, DiT, decoder bridge, `dec_conv_in`, `dec_mid`, or `dec_up0` recomputation.

| Segment | Input CTHW | Output CTHW | Retained optimized Conv3D time (warm run) |
|---|---:|---:|---:|
| residual block A | 512×3×32×32 | 512×3×32×32 | 520 + 455 ms |
| residual block B | 512×3×32×32 | 512×3×32×32 | 470 + 546 ms |
| residual block C | 512×3×32×32 | 512×3×32×32 | 436 + 491 ms |
| deconvolution/crop segment | 513×3×32×32 | 512×5×64×64 | included in 26,849 ms stage remainder |
| final Conv3D | 512×5×64×64 | 512×5×64×64 | 3,065 ms |

The warm Pad stage time is 32,832 ms. The original black-box Android implementation was operator-stopped after multiple minutes, so no fabricated exact baseline or speedup ratio is reported. The Host original ncnn implementation completes in 22,374 ms on a different machine and is used only as a numeric reference, not as an Android performance baseline.

The retained layer supports the verified FP32/no-fused-activation/explicit-temporal-padding contract and rejects unsupported contracts. Internally it allows ncnn ARM 2D packing while the outer Conv3D remains pack1. It creates and loads one Net per stage execution; it does not reload the 206.7 MB artifact for each operator.

## Correctness and rejected experiment

Host/Pad parity: max abs `0.000244140625`, mean abs `0.000008259683`, RMSE `0.000011210390`, cosine `0.9999999999994581`, finite, NaN/Inf zero. Two Pad runs are bit-identical.

A temporal-plane Deconvolution3D prototype passed a small synthetic test but changed the real 513-channel output SHA from `e1b015...476ae` to `20485d...bf040`. It was removed immediately. The final implementation uses ncnn's original Deconvolution3D.

Peak RSS was observed near 1.1 GiB. Temperature was not captured. Xiaomi foreground throttling was reproduced: process CPU dropped to zero after another app became foreground and resumed immediately when SeedEdge returned to foreground. This was distinguished from deadlock and is not counted as model compute time.
