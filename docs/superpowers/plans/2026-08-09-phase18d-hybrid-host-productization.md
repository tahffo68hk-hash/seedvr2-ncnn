# Phase 18D Hybrid Host Productization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver real arbitrary-size one-command Host enhancement with a measured Hybrid/Auto runtime, crash-safe low-disk checkpoints, and Android-ready metadata while preserving exact FP32 output.

**Architecture:** Preserve the current tile and per-stage math. Add a cohort-aware scheduler and generation checkpoint store as focused C++ modules, then route TileMajor/StageMajor/Hybrid/Auto through one selection contract. A Python CLI handles image formats and delegates all inference to the native runner.

**Tech Stack:** C++17, ncnn CPU FP32/OpenMP, CMake/MSBuild, Python 3 + Pillow, Kotlin/JVM, Android Gradle.

## Global Constraints

- Physical Xiaomi Pad 7 Pro gates stay `READY_FOR_DEVICE / NOT_RUN`.
- `peak_resident_model_stages <= 1`; no FP16 checkpoint conversion.
- Preserve four stashes and all ignored model/checkpoint/evidence directories.
- Never commit runtime images, models, checkpoints, APK/SO/A, or logs; never push/tag/create PR.

---

### Task 1: Formal current-runtime 1024 and arbitrary-size gates

**Files:** Create ignored fixtures/tasks; create `reports/phase18d_1024_full_photo_gate.json`, `docs/PHASE18D_1024_FULL_PHOTO_GATE.md`, and `reports/phase18d_arbitrary_size_full_photo_gate.json` after execution.

**Interfaces:** Consumes `phase18d_full_photo_runner`; produces immutable measured baseline metrics and output hashes.

- [ ] Generate EXIF-corrected 1024x1024 and 640x427 RGB8 from an existing natural photo and record input SHA-256.
- [ ] Run 1024 stage-major to 121/121 completion; capture stage totals, checkpoint IO, RSS, peak/final disk, fusion coverage, dimensions, finite/NaN/Inf, and SHA.
- [ ] Run the 640x427 gate and verify exact dimensions, edge padding/crop, all tiles, coverage, and finite output.
- [ ] Write measured reports/docs and commit only text evidence.

### Task 2: Runtime mode and Hybrid scheduler contract

**Files:** Create `android/native/include/phase18d_runtime_strategy.h`, `android/native/src/phase18d_runtime_strategy.cpp`, `android/native/tests/phase18d_runtime_strategy_test.cpp`; modify `android/CMakeLists.txt`, `minimal_video_pipeline.h/.cpp`.

**Interfaces:** Produces `RuntimeMode`, `RuntimeRequest`, `RuntimeSelection`, `parse_runtime_request()`, `select_runtime()`, and cohort ranges `[begin,end)`.

- [ ] Write failing tests for all four modes, invalid cohort sizes, deterministic consecutive cohorts, fixed one-stage residency, and stable error codes.
- [ ] Run the new test and confirm missing API failures.
- [ ] Implement the minimal parser/cohort scheduler and route Hybrid through the existing stage work hooks.
- [ ] Run new and existing scheduler/contract tests; commit the runtime contract.

### Task 3: Crash-safe checkpoint generations and GC

**Files:** Create `phase18d_checkpoint_store.h/.cpp` and `phase18d_checkpoint_store_test.cpp`; modify `phase18d_stage_major.cpp` and Hybrid runtime integration.

**Interfaces:** Produces `publish_generation()`, `load_newest_valid_generation()`, `reclaim_obsolete_generations()`, `CheckpointInventory`, and injected crash points.

- [ ] Write failing tests for interruption before write, partial write, pre-rename, pre-control update, pre-GC, post-GC, corrupt newest generation, and wrong identity.
- [ ] Verify RED, then implement temp-write/validate/atomic-publish/control-update/GC ordering.
- [ ] Add production/debug retention policies and category byte accounting.
- [ ] Verify resume never skips work and output state hashes remain exact; commit checkpoint lifecycle changes.

### Task 4: Resource estimator and Auto strategy

**Files:** Modify `phase18d_runtime_strategy.*`, `phase18d_tile_core.*`; extend their tests; create `reports/phase18d_common_photo_resource_estimates.json`.

**Interfaces:** Produces `estimate_task_resources(width,height,mode,K,calibration)` returning tile count, required/free margin, peak/final bytes, runtime, RSS, source label, and selection reason.

- [ ] Write failing overflow/resource tests for every required dimension from 1x1 through 4032x3024.
- [ ] Fit calibration only from measured gates and benchmark results; keep measured and estimated fields distinct.
- [ ] Implement Auto thresholds from measured crossover and disk constraints, with explicit override behavior.
- [ ] Verify stable errors and emit common-photo estimates; commit estimator/Auto.

### Task 5: Real Hybrid benchmark and parity

**Files:** Create ignored benchmark tasks; create `reports/phase18d_runtime_mode_benchmark.json` and `docs/PHASE18D_HYBRID_RUNTIME.md`.

**Interfaces:** Consumes RuntimeMode/CheckpointStore; produces selected calibration for Auto.

- [ ] Screen TileMajor, StageMajor, Hybrid K=2/K=4 on 2 and 4 tiles; add K=8 only for sufficiently large tasks.
- [ ] Select one or two valuable Hybrid configurations and execute real 256/512 benchmarks.
- [ ] Validate exact output SHA parity, pause/process-exit/resume, load count, stage timings, RSS, peak/final disk, and checkpoint IO.
- [ ] Record measured results and commit text evidence; reject low-value experimental code/configurations.

### Task 6: One-command Host photo CLI

**Files:** Create `tools/phase18d_enhance_photo.py`, `tests/phase18d_host_cli/test_phase18d_enhance_photo.py`; modify runner only if a stable machine-readable planning/control command is required.

**Interfaces:** CLI flags `--input`, `--output`, `--model-root`, `--task-root`, `--runtime-mode`, `--tile-cohort-size`, `--resume`, `--pause-after-dit-block`, `--cancel`; task report remains authoritative.

- [ ] Write failing smoke/identity/EXIF/JPEG/PNG/WebP/progress/error tests using a fake runner boundary but real image IO.
- [ ] Verify RED and implement collision-safe task IDs, identity validation, RGB conversion, invocation, progress, and same-size PNG output.
- [ ] Run format tests and one real enhancement command; verify resume and output SHA.
- [ ] Commit CLI and tests.

### Task 7: Seam and coverage analysis

**Files:** Create `phase18d_seam_metrics.h/.cpp` and tests; integrate into task report.

**Interfaces:** Produces directional seam mean/P95/max gradients, nearby reference gradients, coverage and bounds validity.

- [ ] Write failing synthetic fusion-field tests that distinguish a seam defect from natural gradients.
- [ ] Implement horizontal/vertical boundary sampling and quantiles without a perceptual score.
- [ ] Verify completed 256/512/1024/non-grid evidence reports where executed; commit metrics.

### Task 8: Android metadata and READY_FOR_DEVICE synchronization

**Files:** Modify `PhotoTask.kt`, `PhotoEnhanceService.kt`, `MainActivity.kt`, Kotlin tests, JNI contract, `docs/PHASE18D_PAD_FINAL_GATE.md`, and Pad scripts.

**Interfaces:** Android receives/prints selected mode, cohort size, checkpoint generation, tile/block counts, estimated/actual disk, and pause/resume state.

- [ ] Write failing JVM request/label/state tests, then implement metadata without synthetic progress.
- [ ] Run JVM tests plus Debug/Release builds and inspect APK ABI/content.
- [ ] Update Pad automation to collect runtime selection, RSS/swap/thermal/logcat/task evidence and restore device state.
- [ ] Commit Android-ready changes without claiming device PASS.

### Task 9: Final regression, protection, and handoff

**Files:** Update final reports/docs and `AGENTS.md`/`docs/NEXT_TASK.md` only with executed facts.

**Interfaces:** Produces the 47-field final handoff and Pad-only remaining gate list.

- [ ] Run all existing 12 Host tests plus Hybrid/GC/Auto/resource/arbitrary-size/CLI tests and large-artifact protection.
- [ ] Run Android JVM tests, Debug/Release builds, APK SHA/ABI inspection, and validate every JSON/PowerShell artifact.
- [ ] Confirm four stashes, clean worktree, explicit-path commits, and no push/tag/PR.
- [ ] Report PASS/READY_FOR_DEVICE/NOT_RUN strictly from evidence.
