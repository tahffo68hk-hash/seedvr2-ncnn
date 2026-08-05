# Phase 18C.3/18C.4 `dec_up1` profile

`dec_up1` consumes CTHW `[512,3,32,32]` and emits `[512,5,64,64]`. Its 206,686,240-byte artifact contains three residual blocks (six GroupNorm, six SiLU, six 512→512 Conv3D 3×3×3 operations), one constant-channel concat, one 513→512 Deconvolution3D 2×2×2 stride 2, temporal crop/concat, and one final 512→512 Conv3D 3×3×3. There is no attention and no duplicated upsample. The deconvolution produces the only spatial 32→64 expansion; reference temporal crop/concat gives 3→5.

The pinned ncnn revision has no ARM-specific Conv3D layer. Its generic implementation performs scalar multiply-add loops and parallelizes only output channels. The retained optimization decomposes the temporal kernel into optimized ncnn ARM Conv2D planes and accumulates FP32 results. The real Pad output matches the original Host ncnn output with max absolute error `2.44140625e-4` (limit `1e-3`), mean absolute error `8.2597e-6`, RMSE `1.1210e-5`, and cosine similarity `0.9999999999994581`.

Two Pad runs reloaded the same `dec_up0` checkpoint and produced the same SHA-256 `e1b01576...476ae`; the measured times were 153.406 s (cold/background-throttled) and 32.832 s (foreground). The seven optimized Conv3D calls totaled 5.983 s in the second run; the 26.849 s remainder contains the generic Deconvolution3D, normalization/crop/concat, model load, and lifecycle overhead. An attempted Deconvolution3D replacement was rejected and removed because its real 513-channel output failed parity even though a small synthetic test passed.

Five threads remain selected from the Phase 18C.3 1/2/4/5/6/8 sweep. No allocator, lightmode, affinity, global packing, FP16, or model-semantic change is claimed.
