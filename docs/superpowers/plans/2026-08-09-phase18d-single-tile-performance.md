# Phase 18D Single-Tile Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely reduce the real 128x128 single-tile SeedVR2 ncnn CPU FP32 wall time from the fixed 93.3-second Pad baseline while preserving numerical results, streaming residency, and shared Host/Android behavior.

**Architecture:** Add one shared profiling record around the existing production path, then optimize only measured hotspots. The first candidate is a persistent verified-manifest cache whose key binds the manifest digest and every declared file's path, size, and modification time; every mismatch or malformed cache performs the existing full SHA-256 verification. DiT remains strict one-block-at-a-time streaming and model precision remains FP32.

**Tech Stack:** C++17, ncnn CPU FP32/OpenMP, Android JNI/Kotlin, CMake/CTest, Gradle, JSON evidence files.

## Global Constraints

- Fixed formal BEFORE is 93.3 seconds on the real Pad single-tile run; do not redefine it.
- Execute Encoder, all 32 DiT blocks, scheduler, all Decoder stages, and frame 0 at 128x128 with five repeated input frames.
- Do not skip CFG/scheduler/blocks, reduce input, change sampling, or change FP32 precision.
- Keep at most one weighted DiT block resident and do not materially increase peak RSS.
- No silent fallback; Host and Android compile the same runtime sources.
- Preserve models, checkpoints, evidence, stashes, and untracked history; never use broad git add, reset, clean, or stash deletion.
- Do not push, tag, or open a pull request.

---

### Task 1: Formal BEFORE and profiling contract

**Files:**
- Create: `reports/phase18d_single_tile_perf_before.json`
- Create: `reports/phase18d_single_tile_profile_before.json`
- Modify: `android/native/include/minimal_video_pipeline.h`
- Modify: `android/native/src/minimal_video_pipeline.cpp`
- Modify: `android/native/tests/phase18d_real_hooks_test.cpp`

**Interfaces:**
- Produces a task-local `performance_profile.json` with top-level wall/IO/validation timings, Encoder load/forward, input/output stages, 32 independent DiT block load/forward/unload/RSS entries, scheduler, and seven Decoder stage entries.

- [ ] Write a failing host contract test that runs the real hooks against controlled tiny fixtures and asserts all required profiling fields and exactly 32 ordered block slots.
- [ ] Run the focused CTest target and confirm failure because the profile artifact/fields do not exist.
- [ ] Add monotonic-clock/RSS instrumentation without changing inference ordering or tensor values.
- [ ] Re-run focused tests and the existing Phase 18D host suite.
- [ ] Materialize the fixed 93.3-second BEFORE report, using null only where the preserved run has no evidence and marking those fields as pending instrumentation rather than guessing.
- [ ] Commit only Task 1 paths with `perf: profile Phase 18D single-tile runtime`.

### Task 2: Trusted manifest verification cache

**Files:**
- Modify: `android/native/include/model_manifest.h`
- Modify: `android/native/src/model_manifest.cpp`
- Modify: `android/native/tests/model_manifest_test.cpp`

**Interfaces:**
- `validate_model_manifest(root)` retains its JSON contract and adds validation mode/timing evidence.
- Persistent cache lives below the model root, is atomically replaced, and binds canonical root, manifest SHA-256, declared path, size, and mtime for all files.

- [ ] Write failing tests for first-run full SHA, unchanged cache hit, manifest mutation, file-size mutation, same-size mtime mutation, malformed cache, missing model, symlink rejection, and cache-root mismatch.
- [ ] Run the manifest test and confirm each new behavior fails for the expected missing-cache reason.
- [ ] Implement streaming SHA reads and atomic cache persistence; never accept cache when any identity field differs.
- [ ] Re-run manifest and real-hook tests, including a second validation of the 1366-file linked Host bundle.
- [ ] Benchmark full SHA and cache-hit validation separately and record bytes/files checked.
- [ ] Commit only manifest source/test paths with `perf: cache verified model manifest`.

### Task 3: Profile-driven runtime optimizations

**Files:**
- Modify only files named by the measured hotspot.
- Add focused tests beside the affected runtime component.

**Interfaces:**
- Preserve the existing Host/Android request and result JSON contracts and strict DiT streaming lifecycle.

- [ ] Rank the ten slowest measured operations and classify load, forward, SHA, checkpoint IO, and other IO totals.
- [ ] For the largest remaining low-risk hotspot, write one failing behavior/parity test before production changes.
- [ ] Apply one variable only (load lifecycle, per-stage thread count, allocator/lightmode/packing, or tensor-copy removal).
- [ ] Run parity, wall-time, RSS, and crash checks; remove the experiment if benefit is not material.
- [ ] Repeat only while a clear low-risk hotspot remains, committing each retained optimization separately.

### Task 4: Benchmark, parity, reports, and Android handoff

**Files:**
- Create: `reports/phase18d_single_tile_perf_after.json`
- Create: `reports/phase18d_single_tile_perf_comparison.json`
- Create: `reports/phase18d_tile_major_vs_stage_major.json`
- Create: `docs/PHASE18D_SINGLE_TILE_PERF_PROFILE.md`
- Create: `docs/PHASE18D_SINGLE_TILE_PERFORMANCE.md`

**Interfaces:**
- Reports use the same input/model/backend/checkpoint state and record run count, min/median/mean/max, parity, RSS, and device identity.

- [ ] Run the production Host single-tile path with one warm-up when justified and at least two formal repetitions.
- [ ] Compare final RGB plus key tensors using max_abs, mean_abs, RMSE, cosine, finite/NaN/Inf, and SHA only when bit-identical.
- [ ] Sweep 1/2/4/5/6/8 threads per stage where runtime permits and retain only measured improvements.
- [ ] Use measured load/forward ratios to benchmark or analytically bound 2- and 4-tile tile-major versus stage-major execution without running hundreds of tiles.
- [ ] Generate all required reports and answer the 34-item final handoff checklist.
- [ ] Run Host regressions and `:app:assembleDebug` for arm64-v8a.
- [ ] If Xiaomi Pad 7 Pro is connected, install the Debug APK and run the same real input/model BEFORE/AFTER benchmark; otherwise record `NOT_RUN` with the exact adb evidence.
- [ ] Run git protection, inspect explicit diff/status/stashes, and commit report/doc paths without push/tag/PR.
