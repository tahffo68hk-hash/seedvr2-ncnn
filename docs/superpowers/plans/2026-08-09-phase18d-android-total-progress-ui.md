# Phase 18D Android Total Progress UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Android Pad UI that explains Phase 18D photo tiling and shows whole-photo progress.

**Architecture:** Keep native inference unchanged. Add pure Kotlin helpers in `PhotoTask` for tile planning, tile-map text, stage progress fraction, and whole-photo progress. Consume those helpers from `MainActivity` to render plan and progress.

**Tech Stack:** Kotlin, Android Activity widgets, existing Gradle/JUnit test setup.

## Global Constraints

- Do not change native model execution, checkpointing, fusion, or runtime selection semantics.
- Do not package model files, checkpoints, APKs, or heavy artifacts into git.
- Pad execution remains NOT_RUN until tested on device.
- User-visible Phase 18D Android text should be readable Chinese, not mojibake.

---

### Task 1: Pure Tile Plan And Progress Helpers

**Files:**
- Modify: `android/app/src/main/java/com/seededge/runtime/PhotoTask.kt`
- Modify: `android/app/src/test/java/com/seededge/runtime/PhotoTaskTest.kt`

**Interfaces:**
- Produces: `PhotoTask.TilePlan`
- Produces: `PhotoTask.planTiles(width: Int, height: Int, runtimeMode: String = "auto", cohortSize: Int = 4, availableDiskBytes: Long = Long.MAX_VALUE): TilePlan`
- Produces: `PhotoTask.tileMapText(plan: TilePlan, completedTiles: Int, activeTileIndex: Int = completedTiles): String`
- Produces: `PhotoTask.tileWorkUnits(phase: String, ditBlockIndex: Int = 0, frameIndex: Int = 0): Int`
- Produces: `PhotoTask.overallProgressPercent(completedTiles: Int, totalTiles: Int, phase: String, ditBlockIndex: Int = 0, frameIndex: Int = 0): Int`

- [ ] **Step 1: Write failing tests**

Add tests that expect `640x427` to plan as `7x5=35 tiles`, `hybrid K=4`, `9 cohorts`; expect tile-map rows to mark completed/current/pending cells; expect whole-photo percent to advance inside `dit_streaming` without resetting to the current tile.

- [ ] **Step 2: Run focused JVM tests and confirm RED**

Run: `cd android; .\gradlew.bat testDebugUnitTest --tests com.seededge.runtime.PhotoTaskTest`

Expected: compile failure or assertion failure because the helper APIs do not exist yet.

- [ ] **Step 3: Implement minimal helper APIs**

Implement the helpers in `PhotoTask` using existing tile math and existing runtime estimates. Use 40 work units per tile: model check and manifest phases map to 0, VAE encoder to 1, DiT block to `2 + block`, VAE decoder to 35, writing frames to `36 + frame`, fusion/PNG/save are terminal whole-task stages.

- [ ] **Step 4: Run focused JVM tests and confirm GREEN**

Run: `cd android; .\gradlew.bat testDebugUnitTest --tests com.seededge.runtime.PhotoTaskTest`

Expected: PASS.

### Task 2: Android Screen Rendering

**Files:**
- Modify: `android/app/src/main/java/com/seededge/runtime/MainActivity.kt`
- Modify: `android/app/src/main/java/com/seededge/runtime/PhotoTask.kt`
- Modify: `android/app/src/test/java/com/seededge/runtime/PhotoTaskTest.kt`

**Interfaces:**
- Consumes: Task 1 helpers.

- [ ] **Step 1: Write failing label tests**

Update `fullPhotoStageLabelShowsTileProgress` to assert readable Chinese strings.

- [ ] **Step 2: Run focused JVM tests and confirm RED**

Run: `cd android; .\gradlew.bat testDebugUnitTest --tests com.seededge.runtime.PhotoTaskTest`

Expected: label assertions fail against the current mojibake strings.

- [ ] **Step 3: Update UI text and progress consumption**

Render the plan summary and tile map in `showEnhance`. In `startEnhancement`, set the progress bar max to 100 and the initial details to the plan summary. In `startPoller`, read `status.json` plus `runtime_progress.json`, compute whole-photo percent, and update the state/detail text with total progress, tile progress, cohort progress, current stage, elapsed time, backend, and message.

- [ ] **Step 4: Run focused JVM tests and confirm GREEN**

Run: `cd android; .\gradlew.bat testDebugUnitTest --tests com.seededge.runtime.PhotoTaskTest`

Expected: PASS.

### Task 3: Build And APK Handoff

**Files:**
- Output only: `android/app/build/outputs/apk/debug/app-debug.apk`
- Output only: `phase18d_pad_apk_handoff/seedvr2-phase18d-progress-ui-debug.apk`

- [ ] **Step 1: Run Android JVM suite**

Run: `cd android; .\gradlew.bat testDebugUnitTest`

Expected: PASS.

- [ ] **Step 2: Build debug APK**

Run: `cd android; .\gradlew.bat assembleDebug`

Expected: exit 0.

- [ ] **Step 3: Copy debug APK to handoff directory**

Copy the debug APK to `phase18d_pad_apk_handoff/seedvr2-phase18d-progress-ui-debug.apk`, write SHA-256, and report the install command.
