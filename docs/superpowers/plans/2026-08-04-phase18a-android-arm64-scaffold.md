# Phase 18A Android ARM64 Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, device-free Android ARM64 SeedEdge runtime scaffold that builds ncnn with Vulkan support, packages only `arm64-v8a`, and is ready for Phase 18B device execution.

**Architecture:** A minimal Kotlin Activity calls a focused JNI facade in `libseededge.so`. Native code separates runtime metadata, deterministic ncnn CPU validation, Vulkan capability probing, and external model-manifest validation. PowerShell scripts pin all Android tooling and caches to D:, build both APK variants, inspect package/ELF contents, run host-side negative tests, and generate evidence-backed JSON reports.

**Tech Stack:** Android Gradle Plugin/Kotlin/JDK 17, Gradle Wrapper, Android SDK, NDK r26d, CMake 3.22.1, official pinned ncnn source, C++17/JNI, Kotlin.

## Global Constraints

- Work from Phase 17 commit `13b2d14e945ea3d07246006d6c75840184f9448c` on `codex/phase18a-android-arm64-scaffold`.
- Use only `arm64-v8a`; use SDK/NDK/JDK/CMake/Ninja/Gradle caches under `D:\Android`.
- Build ncnn from a repository-pinned official commit with Vulkan enabled; do not commit generated libraries, APKs, caches, SDKs, models, or weights.
- Device execution, Adreno/Vulkan inference, VAE/DiT inference, and full restoration remain false and unclaimed.
- Final Phase 18A state may be `READY_FOR_DEVICE`, `PARTIAL`, or `BLOCKED`; full Phase 18 remains not ready.
- Create exactly one final local commit, `feat: prepare SeedVR2 Android ARM64 runtime scaffold`; do not push, tag, or create a PR.

---

### Task 1: Baseline, branch, pinned toolchain, and Gradle project

**Files:** create `android/settings.gradle.kts`, root/app Gradle files, wrapper files, manifest/resources/Kotlin Activity, CMake entrypoint, and environment scripts; modify `.gitignore`.

- [ ] Record the baseline status/diff and switch to the Phase 18A branch without discarding the existing Phase 18 setup files.
- [ ] Detect installed SDK platforms/build-tools and select stable explicit compile/target versions; record exact Gradle/AGP/Kotlin/JDK/NDK/CMake versions.
- [ ] Add a minimal single-activity project and wrapper with D-drive-only environment/bootstrap scripts.
- [ ] Run `gradlew.bat --version` and environment checks; require nonzero exits for missing SDK, NDK, JDK, wrong ABI, or wrong paths.

### Task 2: Test-first model manifest contract

**Files:** create `android/native/include/model_manifest.h`, `android/native/src/model_manifest.cpp`, `android/native/tests/model_manifest_test.cpp`, compact manifest fixtures, and host test CMake/script files.

- [ ] Write tests first for missing/invalid manifest, format version, missing file, size mismatch, SHA-256 syntax, absolute/traversal paths, and valid minimal resources; run and capture expected failures.
- [ ] Implement the smallest safe parser/validator API needed by those tests, with no silent fallback.
- [ ] Re-run host tests and record exact pass/fail counts.

### Task 3: Test-first runtime/JNI and synthetic ncnn CPU test

**Files:** create focused native runtime, JSON, ncnn synthetic-test, Vulkan-probe, and JNI source/header files plus tests; update app CMake and Kotlin bridge/UI.

- [ ] Write host-contract/static tests first for the five required JNI functions, parseable finite JSON, `device_tested=false`, deterministic CPU-test shape/error/repeat checks, and Vulkan no-device semantics; run and capture expected failures.
- [ ] Implement runtime/build/ncnn JSON, a deterministic CPU-only ncnn Net execution, explicit resource release, and a guarded Vulkan probe that reports actual device execution only when called on Android.
- [ ] Connect the Kotlin Activity buttons/status fields to all five JNI functions and the external-files model root.
- [ ] Re-run host tests/static contracts and record evidence without representing host tests as Android execution.

### Task 4: Reproducible ncnn Android ARM64 integration and APK builds

**Files:** create pinned ncnn fetch/build metadata/scripts and app native build wiring; generated sources/build outputs remain ignored.

- [ ] Identify the Phase 14-17 ncnn source commit/license/build configuration and pin the same official revision.
- [ ] Configure ARM64-only ncnn with Vulkan enabled and record OpenMP, exceptions, RTTI, and shader compilation options.
- [ ] Run clean, debug, and release builds with all caches/logs/output on D:; capture APK and native-library sizes.
- [ ] If a build fails, diagnose and fix it, adding a failing regression test before behavioral source fixes.

### Task 5: APK inspection, negative gates, and historical regressions

**Files:** create `android/scripts/inspect_apk.ps1`, `check_d_drive_paths.ps1`, negative-test drivers, and report generators.

- [ ] Test the inspector first with synthetic bad ZIPs containing a forbidden ABI/model entry and verify expected failures.
- [ ] Inspect produced APKs for required entries, AArch64 ELF, one ABI, no weights/secrets/absolute paths, SDK configuration, debug/release labeling, and reasonable size.
- [ ] Run all 15 required negative cases, JSON parse/finite checks, lightweight Phase 14-17 protections, conflict-marker scan, and large/sensitive tracked-file checks.
- [ ] Audit C:/D: paths read-only and report any observed limitations without deleting user files.

### Task 6: Reports, documentation, final verification, and commit

**Files:** create the eleven `reports/phase18a_*.json` files and `docs/android-phase18a-build-and-scaffold-report.md`; update `README.md`, `AGENTS.md`, `docs/NEXT_TASK.md`, `docs/CURRENT_STATUS.md`, `docs/CODEX_HANDOFF.md`, and `docs/TEST_MATRIX.md`.

- [ ] Generate UTF-8 strict JSON with `overall_pass`, `device_required`, and `device_executed`, preserving all device-only results as false.
- [ ] Update documentation to distinguish Phase 17 READY, Phase 18A device-free readiness, and Phase 18 overall NOT READY.
- [ ] Run clean rebuilds, all host/negative/regression/APK/git-protection checks, `git diff --check`, tracked binary scans, and inspect the final diff/status.
- [ ] Create the single requested local commit only if the evidence supports the reported Phase 18A status; do not push.
