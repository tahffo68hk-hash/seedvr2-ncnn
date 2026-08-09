# Phase 18D Stage-Major Full-Photo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a resumable, resource-bounded C++17 stage-major full-photo runtime shared by Host and Android, prove it with real two/four-tile and 256/512 image runs where resources permit, and leave only physical Xiaomi Pad 7 Pro gates unexecuted.

**Architecture:** Keep the existing C++ tile planner and weighted fusion as the single Host/Android implementation. Add a task scheduler that persists one validated tensor state per tile, loads one Encoder/DiT/Decoder stage at a time, executes all pending tiles through that loaded stage, atomically advances the task checkpoint, and releases the model before continuing. Kotlin owns only Android lifecycle/UI/service integration and consumes the native progress/checkpoint contract.

**Tech Stack:** C++17, ncnn CPU FP32, Kotlin/JVM Android 35, JNI, CMake/MSBuild, Gradle 8.9, PowerShell gate scripts.

## Global Constraints

- Preserve all four historical stashes, models, ignored evidence, checkpoints, reports, and Phase 8 history.
- Never skip Encoder, any of the 32 DiT blocks, scheduler, CFG path, or Decoder; never silently fall back.
- Keep at most one weighted model stage resident; use tile-state files to bound RAM.
- Model and task identities must be validated before resuming; checkpoint writes use temporary file plus atomic rename.
- Do not claim any Xiaomi Pad gate PASS without new physical-device evidence; use `READY_FOR_DEVICE` or `NOT_RUN`.
- Do not commit models, tensor checkpoints, task outputs, PNG/APK/SO/A files, device identifiers, or raw logs.
- Do not push, tag, or create a pull request.

---

### Task 1: Resource Guards, Disk Estimate, and Task Identity

**Files:**
- Modify: `android/native/include/phase18d_tile_core.h`
- Modify: `android/native/src/phase18d_tile_core.cpp`
- Modify: `android/native/tests/phase18d_tile_core_test.cpp`

**Interfaces:**
- Produces: `PhotoRuntimeLimits`, `PhotoStorageEstimate`, `validate_photo_limits(...)`, `estimate_photo_storage(...)`, and a deterministic `tile_plan_sha256(...)`.

- [ ] Write failing tests for the required size matrix, invalid/overflow dimensions, `max_pixels`, `max_tiles`, fusion allocation safety, deterministic plan SHA, and insufficient free space.
- [ ] Build and run `phase18d_tile_core_test`; verify failures are caused by the missing interfaces.
- [ ] Implement checked arithmetic and conservative storage estimates for RGB input/output, fusion buffers, per-tile encoder/DiT/decoder state, metadata, and atomic-write overhead.
- [ ] Rebuild and verify all tile-core tests pass.
- [ ] Commit only the three scoped files as `feat: guard full-photo resources and storage`.

### Task 2: Resumable Stage-Major State Machine

**Files:**
- Create: `android/native/include/phase18d_stage_major.h`
- Create: `android/native/src/phase18d_stage_major.cpp`
- Create: `android/native/tests/phase18d_stage_major_test.cpp`
- Modify: `android/native/tests/CMakeLists.txt`
- Modify: `android/CMakeLists.txt`

**Interfaces:**
- Consumes: `TilePlan`, `PhotoRuntimeLimits`, `tile_plan_sha256(...)`.
- Produces: `StageMajorCheckpoint`, `StageMajorProgress`, `load_stage_major_checkpoint(...)`, `persist_stage_major_checkpoint(...)`, `validate_stage_major_checkpoint(...)`, `next_stage_major_work(...)`, `mark_stage_major_work_complete(...)`.

- [ ] Write failing tests for a new task, Encoder interruption, DiT block 5/31 interruption, Decoder interruption, pre-fusion interruption, exact resume position, corrupted metadata/tensor rejection, wrong task/source/model/plan rejection, pause/cancel boundaries, and duplicate-start exclusion.
- [ ] Configure/build the new test and verify the expected missing-symbol failure.
- [ ] Implement schema-versioned JSON metadata, tensor SHA/shape/finite identity, atomic persistence, legal stage transitions, per-tile completion vectors, and cooperative pause/cancel states.
- [ ] Rebuild and verify the new test plus existing progress/tile tests pass.
- [ ] Commit scoped files as `feat: add resumable stage-major state machine`.

### Task 3: Real Stage-Major ncnn Executor

**Files:**
- Modify: `android/native/include/minimal_video_pipeline.h`
- Modify: `android/native/src/minimal_video_pipeline.cpp`
- Modify: `android/native/include/phase18d_stage_major.h`
- Modify: `android/native/src/phase18d_stage_major.cpp`
- Create: `android/native/tests/phase18d_stage_major_contract_test.cpp`
- Modify: `android/native/tests/CMakeLists.txt`
- Modify: `android/CMakeLists.txt`

**Interfaces:**
- Produces: `run_phase18d_stage_major_photo(...)` and execution hooks whose load/execute/unload counters prove one model load per stage, not per tile.
- Runtime order: 7 Encoder stages across N tiles; `vid_in`; DiT blocks 0..31 across N tiles; post-head/scheduler; 7 Decoder stages across N tiles; valid-crop output.

- [ ] Write failing hook-contract tests for two and four real RGB tiles, one load/unload per stage, per-tile checkpoint persistence, stage/tile progress, cancellation, resume without replay, and complete-chain flags.
- [ ] Build/run and verify failures are attributable to the absent scheduler.
- [ ] Extract a reusable loaded ncnn model wrapper from `run_model`, retain one `HybridDiTBlockRuntime` per block while processing pending tiles, persist minimal tile state between stages, then clear cache/runtime before the next stage.
- [ ] Wire production `run_phase18d_full_photo` to request `runtime_mode=stage_major`, while retaining explicit `tile_major` for benchmark comparison and refusing unknown modes.
- [ ] Rebuild and run stage-major contract, full-photo contract, single-tile contract, real-hooks, progress, and tile-core tests.
- [ ] Commit scoped files as `perf: add real stage-major multi-tile runtime`.

### Task 4: Real Fusion, Artifacts, and Resume Reports

**Files:**
- Modify: `android/native/src/minimal_video_pipeline.cpp`
- Modify: `android/native/src/phase18d_full_photo_artifacts.cpp`
- Modify: `android/native/tests/phase18d_full_photo_contract_test.cpp`
- Modify: `android/native/tests/phase18d_full_photo_artifacts_test.cpp`
- Modify: `android/native/tests/phase18d_tile_core_test.cpp`

**Interfaces:**
- Produces: `tile_plan.json`, `runtime_progress.json`, `task_report.json`, `diff.png`, and status fields for coverage, zero weights, seam metrics, finite/NaN/Inf, checkpoint bytes/IO, RSS, and stage profiles.

- [ ] Write failing tests that feed non-identity tile outputs through real weighted fusion, verify padded bytes never enter fusion, require same-size output, full coverage, positive weights, complementary overlaps, and artifact atomicity.
- [ ] Run tests and observe the expected missing metadata/artifact failures.
- [ ] Add task/report serialization and diff artifact generation; load completed tile outputs from task storage so resumed fusion does not require all tile RGB vectors resident.
- [ ] Rebuild and verify artifact/full-photo/tile-core regression tests.
- [ ] Commit scoped files as `feat: fuse resumable real tile outputs`.

### Task 5: Model Cache Hardening

**Files:**
- Modify: `android/native/src/model_manifest.cpp`
- Modify: `android/native/tests/model_manifest_test.cpp`

**Interfaces:**
- Produces: cache schema/model-set fields bound to canonical root and manifest digest.

- [ ] Add failing tests for unchanged hit, manifest miss, size mutation, mtime mutation, model-set/version mutation, corruption, schema mismatch, and copied cache from another root.
- [ ] Run `model_manifest_test` and verify the new schema/model-set cases fail.
- [ ] Persist and validate cache schema plus manifest model set while preserving the full SHA fallback on any mismatch.
- [ ] Run all  cache tests and the 1366-file local model-bundle validation.
- [ ] Commit scoped files as `test: harden verified model manifest cache`.

### Task 6: Android Background Task Ownership

**Files:**
- Create: `android/app/src/main/java/com/seededge/runtime/PhotoEnhanceService.kt`
- Create: `android/app/src/main/java/com/seededge/runtime/PhotoTaskState.kt`
- Create: `android/app/src/test/java/com/seededge/runtime/PhotoTaskStateTest.kt`
- Modify: `android/app/src/main/java/com/seededge/runtime/PhotoTask.kt`
- Modify: `android/app/src/main/java/com/seededge/runtime/MainActivity.kt`
- Modify: `android/app/src/main/AndroidManifest.xml`

**Interfaces:**
- Produces: foreground-service task ownership, notification channel/progress, partial WakeLock, one-task duplicate guard, activity reattachment, pause/resume/cancel commands, and disk preflight.

- [ ] Write failing JVM tests for request JSON runtime mode/limits, state restoration, duplicate start, stage-boundary pause/cancel, resume identity, progress fields/EMA fallback, and disk rejection.
- [ ] Run `testDebugUnitTest` and verify the expected failures.
- [ ] Implement the pure Kotlin state reducer first, then service/notification/WakeLock lifecycle and MainActivity binding; keep all tile planning in C++.
- [ ] Run JVM tests and `assembleDebug`; inspect APK ABIs and manifest service/permissions.
- [ ] Commit scoped files as `feat: prepare Android resumable full-photo service`.

### Task 7: Pad One-Command Gates

**Files:**
- Create: `scripts/phase18d_pad_after_benchmark.ps1`
- Create: `scripts/phase18d_pad_full_photo_gate.ps1`
- Create: `docs/PHASE18D_PAD_FINAL_GATE.md`

**Interfaces:**
- Produces: scripted adb identity/install/model/cache/benchmark/gallery/background/resume/MediaStore/log collection with device-state restoration.

- [ ] Add PowerShell parse/dry-run validation that fails if required APK/model/report paths or arm64 device checks are absent.
- [ ] Implement bounded scripts with explicit serial selection, Pad identity check, APK install, second-run cache assertion, report pull, crash/ANR scan, and `finally` restoration.
- [ ] Run parser and `-WhatIf`/offline paths; record all physical actions as `READY_FOR_DEVICE` or `NOT_RUN`.
- [ ] Write the eight concrete gates with command, expected output, PASS/FAIL, and evidence path.
- [ ] Commit scoped files as `docs: prepare Phase 18D Pad final gate`.

### Task 8: Real Host Benchmarks and Final Evidence

**Files:**
- Modify: `android/native/tests/phase18d_full_photo_runner.cpp`
- Create/modify: `reports/phase18d_two_tile_tile_major.json`
- Create/modify: `reports/phase18d_two_tile_stage_major.json`
- Create/modify: `reports/phase18d_two_tile_comparison.json`
- Modify: `reports/phase18d_tile_major_vs_stage_major.json`
- Create: `docs/PHASE18D_STAGE_MAJOR_RUNTIME.md`
- Create: `docs/PHASE18D_HOST_FULL_PHOTO_GATES.md`

**Interfaces:**
- Produces: real two-tile tile-major/stage-major, four-tile stage-major, 256x256 natural-image, 512x512 natural-image plus interruption/resume evidence, and a 1024 plan/partial decision.

- [ ] Extend the runner CLI test-first with explicit runtime mode, stop boundary, resume, report path, and no-replay counters.
- [ ] Build Release runner and run the exact same natural RGB inputs/model bundle in tile-major and stage-major modes.
- [ ] Run two-tile and four-tile benchmarks; record wall/load/forward/IO/RSS/SHA/finite metrics and strict output parity.
- [ ] Run 256x256 full-photo and verify every tile's 32 blocks, scheduler, decoder, weighted fusion, output dimensions, SHA, coverage, seam and finite gates.
- [ ] Run 512x512 once with a controlled stop after DiT block 10, resume without replay, and compare against an uninterrupted result; if a real resource ceiling is reached, capture free disk/RSS/time/error and stop only that run.
- [ ] After 256/512 PASS, perform 1024 planning and a bounded partial 4/8-tile estimate unless the full run is justified by measured time/resources.
- [ ] Run full Host regression, Android unit/build, arm64-only APK inspection, large-artifact protection, `git diff --check`, and stash preservation.
- [ ] Commit only source/tests/docs/small JSON evidence as `docs: record Phase 18D stage-major host evidence`.
