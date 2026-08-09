# Phase 18D 1024×1024 full-photo baseline

The pre-Hybrid real Host CPU FP32 stage-major runtime completed a natural 1024×1024 image on 2026-08-09. All 121 tiles completed all 32 DiT blocks, scheduler, decoder, and fusion. The output is finite, same-size RGB, has no uncovered/zero-weight pixels, and did not silently fall back.

Measured wall time was 1,699.1 s (28.32 min), peak RSS was 1,480.41 MiB, model stages loaded/unloaded 48/48, and peak weighted-stage residency was one. Task-directory monitoring at two-second intervals measured 10,275,165,815 bytes peak and 270,228,302 bytes final. The peak is the baseline for evaluating production checkpoint GC and Hybrid cohorts.

The exact output RGB SHA-256 is `7f852ab6b1e9e2272a6263cf67373492408cd1b8a0f2d9a7652d2318f1bb8173`. This baseline predates the directional seam analyzer, so only the original aggregate seam mean/max are recorded; the report explicitly does not infer perceptual quality from them.

Xiaomi Pad 7 Pro execution remains **READY_FOR_DEVICE / NOT_RUN** and complete Phase 18 remains **NOT READY**.
