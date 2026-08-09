# Phase 18D Hybrid Host runtime

Hybrid partitions the deterministic tile plan into consecutive cohorts. Each cohort runs the existing seven encoder stages, DiT input, 32 DiT blocks, scheduler, and seven decoder stages with at most one weighted model stage resident. Completed work is published as crash-safe checkpoint generations; production mode retains the newest two valid generations and only state required for resume. Fusion runs once after every cohort completes. CPU FP32 tensor mathematics and tile/fusion ordering are unchanged.

## Measured strategy

| Input | Tiles | Mode | Wall | Loads | Exact RGB parity |
|---|---:|---|---:|---:|---|
| 160×96 | 2 | TileMajor | 30.381 s median | 96 equivalent | reference |
| 160×96 | 2 | StageMajor | 45.803 s | 48 | yes |
| 160×96 | 2 | Hybrid K=2 | 43.338 s | 48 | yes |
| 224×224 | 4 | StageMajor | 71.536 s | 48 | reference |
| 224×224 | 4 | Hybrid K=2 | 76.389 s | 96 | yes |
| 224×224 | 4 | Hybrid K=4 | 71.844 s | 48 | yes |
| 256×256 | 9 | StageMajor | 131.796 s | 48 | reference |
| 256×256 | 9 | Hybrid K=4, process-exit + resume | 160.377 s | 144 | yes |
| 512×512 | 25 | StageMajor | 424.684 s | 48 | reference |
| 512×512 | 25 | Hybrid K=4, pipe-block recovery + resume | 484.875 s | 336 | yes |
| 1024×1024 | 121 | StageMajor | 1,699.100 s | 48 | reference |
| 1024×1024 | 121 | Hybrid K=4 | 2,058.990 s | 1,488 | yes |

The measured rule is therefore not “Hybrid is always faster.” TileMajor wins at two tiles. StageMajor is the latency choice through the largest measured medium task (25 tiles) when storage is sufficient. Hybrid K=4 is the bounded-disk/resumable choice for longer tasks, explicit resume, or when StageMajor's estimated peak cannot fit. Explicit user mode and cohort size always override Auto.

The 512 Hybrid task measured 425,728,469 bytes peak and about 56.9 MB final versus the older StageMajor conservative requirement of 2,635,497,472 bytes. Peak RSS was 1,477.52 MiB and output SHA matched the uninterrupted StageMajor result exactly.

The uninterrupted 1024 Hybrid task completed 121/121 tiles with exact StageMajor RGB SHA parity. It measured 596,448,430 bytes peak task disk versus the 10,275,165,815-byte StageMajor baseline, a 94.195% reduction, while wall time increased 21.181%. This confirms Hybrid as the bounded-disk long-task mode, not the latency winner; `reports/phase18d_runtime_mode_benchmark.json` records the measured calibration.

## Recovery and process I/O

The 256 task was terminated after a committed decoder-4 generation. Resume loaded only the final two decoder stages and fusion, then produced exact SHA parity. The 512 screen also exposed a Host CLI pipe deadlock: the CLI captured native stderr without draining it, and repeated ncnn layer warnings filled the pipe around the third cohort. A regression emitting 100,000 stderr lines reproduces the old timeout; native stdout/stderr now stream directly to task-local logs and the regression completes. The interrupted 512 task resumed from decoder-5 and finished bit-exact.

Crash injection covers before-write, half-write, pre-rename, pre-control-update, pre-GC and post-GC. Startup ignores temporary generations, rejects wrong identities and corrupt newest generations, and selects the newest valid generation without skipping work.

Directional seam statistics are structural diagnostics, not a perceptual score. Reports include valid/duplicate boundary counts, vertical/horizontal mean, P95 and max gradients, nearby non-seam reference gradients, zero-weight and uncovered pixels.

Xiaomi Pad 7 Pro calibration and execution remain **READY_FOR_DEVICE / NOT_RUN**. Complete Phase 18 remains **NOT READY**.
