# Phase 18D Stage 1 Shared Tile Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one C++17 tile core and run `tile_0000` of a real photo through the full SeedVR2 chain without Phase 18C.4 checkpoint reuse.

**Architecture:** C++ owns tile planning, extraction/padding, SHA and checkpoint contracts, identity fusion, and native single-tile execution. Kotlin owns Photo Picker import, EXIF-corrected RGB8 buffer creation, task storage, PNG encoding, UI, and JNI. Host and Android compile the same C++ core source.

**Tech Stack:** C++17, CMake, ncnn CPU FP32/OpenMP, Android JNI, Kotlin, Gradle, CTest.

## Global Constraints

- Fixed contract: tile=128, overlap=32, stride=96, repeated T=5, target frame=0, RGB8 HWC sRGB, `value / 127.5 - 1`.
- Preserve models, Phase 18C.4 checkpoints, reports, output, worktree artifacts, and every stash.
- Do not commit photos, PNG output, tensor/checkpoint, APK, SO/A, device ID, or raw log; never use `git add .`.
- No C++20 upgrade, no duplicate Kotlin tile/padding/blending implementation, no push/tag/PR/reset/clean/stash deletion.

---

## File structure

- Create `android/native/include/phase18d_tile_core.h`: C++17 types and stable public core interface.
- Create `android/native/src/phase18d_tile_core.cpp`: planner, padding, SHA, checkpoint validation, state/error JSON, and identity fusion.
- Create `android/native/tests/phase18d_tile_core_test.cpp`: Host assertions for every core contract.
- Modify `android/native/tests/CMakeLists.txt` and `android/CMakeLists.txt`: link exactly that core source into Host and Android targets.
- Modify `android/native/include/minimal_video_pipeline.h` and `android/native/src/minimal_video_pipeline.cpp`: real tile runtime.
- Modify `android/native/src/jni_runtime.cpp` and `android/app/src/main/java/com/seededge/runtime/NativeRuntime.kt`: direct-buffer JNI contract.
- Create `android/app/src/main/java/com/seededge/runtime/PhotoTask.kt`: source copy, EXIF RGB conversion, hash/task metadata, and PNG handoff.
- Modify `android/app/src/main/java/com/seededge/runtime/MainActivity.kt` and `android/app/src/androidTest/java/com/seededge/runtime/RuntimeDeviceTest.kt`: real Stage-1 UI and instrumentation contract.

### Task 1: Shared planner, extraction, and padding

**Files:** create `phase18d_tile_core.h/.cpp` and `phase18d_tile_core_test.cpp`; modify both CMake files.

**Interfaces:**

```cpp
namespace seededge::phase18d {
struct TileConfig { int tile_size=128, overlap=32, stride=96; };
struct TileSpec { int tile_index,grid_x,grid_y,source_left,source_top,source_right,source_bottom,valid_width,valid_height,padding_left,padding_top,padding_right,padding_bottom; };
struct TilePlan { int image_width,image_height; TileConfig config; std::vector<TileSpec> tiles; };
struct PaddedTile { std::vector<uint8_t> rgb; TileSpec spec; std::string padding_mode_x,padding_mode_y; };
bool plan_tiles(int,int,const TileConfig&,TilePlan*,std::string*);
bool extract_and_pad_rgb8(const uint8_t*,size_t,int,int,const TileSpec&,PaddedTile*,std::string*);
}
```

- [ ] **Step 1: Write the failing test**

```cpp
for (auto [w,h] : {std::pair{1,1},{64,64},{128,128},{129,128},{256,256},{320,240},{512,512},{1024,768},{3200,2136}}) {
  TilePlan plan; std::string err;
  expect(plan_tiles(w,h,{},&plan,&err), "planner accepts size");
  expect(coverage_is_exact(plan,w,h), "coverage is exact");
}
```

Add 1x1, 1x64, 64x1, 64x64, 127x128, right, bottom, and corner padding tests that require 128*128*3 output, valid-pixel preservation, reflect only when valid length >1 and pad < length, otherwise replicate.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S android/native/tests -B build/phase18d-host && cmake --build build/phase18d-host --target phase18d_tile_core_test && build/phase18d-host/phase18d_tile_core_test`

Expected: FAIL because the target/core does not exist.

- [ ] **Step 3: Write minimal implementation**

Generate starts `0,96,...`, append `max(0,d-128)`, sort/dedupe, use exclusive bounds and row-major tiles. Implement manual reflect-period mapping and replicate fallback. Reject any configuration other than the fixed contract.

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: `phase18d_tile_core_test=PASS`.

- [ ] **Step 5: Commit**

```bash
git add android/CMakeLists.txt android/native/include/phase18d_tile_core.h android/native/src/phase18d_tile_core.cpp android/native/tests/CMakeLists.txt android/native/tests/phase18d_tile_core_test.cpp
git commit -m "feat: add shared full-resolution tile core"
```

### Task 2: SHA, checkpoint identity, errors/states, and identity fusion

**Files:** modify the Task 1 core/header/test files.

**Interfaces:**

```cpp
std::string sha256_hex(const uint8_t*,size_t);
struct CheckpointIdentity { std::string task_id,normalized_image_sha256,tile_input_sha256,model_manifest_sha256,dtype,checkpoint_sha256; TileSpec tile; std::vector<int> shape; size_t byte_count; bool finite; };
bool validate_checkpoint_identity(const CheckpointIdentity&,const CheckpointIdentity&,std::string* code,std::string* error);
struct FusionResult { std::vector<float> rgb,weight_sum; size_t uncovered_pixel_count,zero_weight_pixel_count; float max_abs_error; };
bool identity_fuse_rgb8(const TilePlan&,const uint8_t*,size_t,FusionResult*,std::string*);
```

- [ ] **Step 1: Write the failing test**

Assert SHA-256 empty and `abc` vectors; reject altered task/hash/tile/model/shape/byte-count/checkpoint hash/non-finite identity. For 64x64, 128x128, 129x128, 256x256, 320x240, 512x512, and 1024x768 deterministic RGB, require zero uncovered pixels, zero zero-weight pixels, and FP32 identity error <=1e-5. Test documented JSON error codes and forbidden state transitions.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/phase18d-host --target phase18d_tile_core_test && build/phase18d-host/phase18d_tile_core_test`

Expected: FAIL for missing hash/fusion/checkpoint functions.

- [ ] **Step 3: Write minimal implementation**

Implement SHA-256, strict identity equality, fixed error codes/states, and FP32 accumulation. Use complementary per-overlap weights `cos((pi/2)*(k+0.5)/n)^2` and `sin((pi/2)*(k+0.5)/n)^2`; derive actual neighbours and retain outer-edge weight one.

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: all core assertions PASS.

- [ ] **Step 5: Commit**

```bash
git add android/native/include/phase18d_tile_core.h android/native/src/phase18d_tile_core.cpp android/native/tests/phase18d_tile_core_test.cpp
git commit -m "feat: add tile checkpoint and fusion contracts"
```

### Task 3: Real native single-tile runtime

**Files:** modify `android/native/include/minimal_video_pipeline.h`, `android/native/src/minimal_video_pipeline.cpp`, and core tests.

**Interface:**

```cpp
std::string run_phase18d_single_photo_tile(const std::string& model_root,const std::string& task_root,const uint8_t* normalized_rgb,size_t bytes,int width,int height,const std::string& request_json);
```

- [ ] **Step 1: Write the failing test**

Require malformed JSON, wrong schema/config/tile index, bad buffer size, and mismatched normalized SHA to return documented errors before any model load. Require any request that names a Phase 18C.4 `frames` checkpoint path to be rejected.

- [ ] **Step 2: Run test to verify it fails**

Run the Host target. Expected: runtime entry point/error contract absent.

- [ ] **Step 3: Write minimal implementation**

Parse schema 1, validate model manifest identity, make full plan, select only tile 0, write task-local metadata/input bytes, and create new checkpoints only after actual stages. Reuse the validated VAE encoder, strict blocks 0..31 load/execute/unload, scheduler, all decoder stages, and frame 0 conversion. Crop to valid region; atomically write `output_valid.rgb`; return per-stage/finite evidence. Never read Phase 18C.4 checkpoints.

- [ ] **Step 4: Run tests to verify it passes**

Run Host negative tests. Expected: invalid paths/requests fail before model inference.

- [ ] **Step 5: Commit**

```bash
git add android/native/include/minimal_video_pipeline.h android/native/src/minimal_video_pipeline.cpp android/native/tests/phase18d_tile_core_test.cpp
git commit -m "feat: connect real photo tile to SeedVR2 runtime"
```

### Task 4: Kotlin import, direct JNI, and PNG output

**Files:** modify JNI and `NativeRuntime.kt`; create `PhotoTask.kt`; modify `MainActivity.kt` and instrumentation test.

**Interfaces:**

```kotlin
external fun nativePlanPhoto(normalizedRgb: ByteBuffer, width: Int, height: Int, requestJson: String): String
external fun nativeRunSinglePhotoTile(modelRoot: String, taskRoot: String, normalizedRgb: ByteBuffer, width: Int, height: Int, requestJson: String): String
```

- [ ] **Step 1: Write the failing Android test**

Use a direct RGB buffer for 129x128 and require a valid plan. Require heap, undersized direct buffer, bad SHA, and malformed JSON to return stable errors. Preserve a pre-call copy and assert native did not mutate buffer.

- [ ] **Step 2: Run test to verify it fails**

Run: `gradlew.bat :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.seededge.runtime.RuntimeDeviceTest`

Expected: JNI declarations are unresolved before implementation.

- [ ] **Step 3: Write minimal implementation**

Import into `files/tasks/<uuid>`, hash source bytes, apply EXIF orientation, write normalized RGB8 and hash it, then call native with direct buffer. JNI checks direct address/capacity and recomputes hash. On native success Kotlin validates output RGB byte count, encodes without resizing, reopens PNG, hashes it, and presents only verified native output as real; UI DEMO remains separately labelled.

- [ ] **Step 4: Run Android test and build**

Run focused instrumentation and `gradlew.bat :app:assembleDebug`. Expected: JNI contract passes and arm64-v8a Debug APK builds.

- [ ] **Step 5: Commit**

```bash
git add android/native/src/jni_runtime.cpp android/app/src/main/java/com/seededge/runtime/NativeRuntime.kt android/app/src/main/java/com/seededge/runtime/PhotoTask.kt android/app/src/main/java/com/seededge/runtime/MainActivity.kt android/app/src/androidTest/java/com/seededge/runtime/RuntimeDeviceTest.kt
git commit -m "feat: connect gallery photos to tiled SeedVR2 runtime"
```

### Task 5: Evidence and real Gate 3/4

**Files:** create/update `reports/phase18d_tile_planner_tests.json`, `reports/phase18d_tile_blending_tests.json`, `reports/phase18d_real_photo_single_tile.json`, `docs/PHASE18D_OVERNIGHT_RESULT.md`, and `reports/phase18d_overnight_summary.json` only from actual runs.

- [ ] **Step 1: Write failing report validation test**

Reject reports that omit dimensions, coverage, padding modes, blend metrics, hash fields, stage evidence, PNG verification, or provenance. Reject PASS when `gallery_photo=false`, blocks !=32, any stage false, output missing, or finite checks fail.

- [ ] **Step 2: Run test to verify it fails**

Run the Host test target. Expected: report validator rejects incomplete evidence.

- [ ] **Step 3: Execute gates**

Run Host and JNI tests, install Debug APK, then search the private app tree for the prior 3200x2136 imported photo. Use it if found; otherwise only a non-checkerboard legal fixture with `fixture_photo=true` and `gallery_photo=false`. Execute `tile_0000` in foreground and collect real progress.

- [ ] **Step 4: Verify and clean up**

Verify PNG dimensions, hashes, finite counts, 32-block completion, no input-byte copy, and only actual device facts. Then force-stop both packages, remove idle whitelist entry, and set stay-awake false after writes complete.

- [ ] **Step 5: Commit only permitted evidence source/docs**

```bash
git add docs reports tests android
git status --short
git commit -m "docs: record Phase 18D first-stage validation"
```

Remove generated images, APKs, tensors, checkpoints, logs, and device IDs from the explicit staging list before committing.

## Plan self-review

Tasks 1–2 cover the shared contracts; Task 3 covers no-legacy-checkpoint real tile execution; Task 4 covers Android import/JNI/PNG; Task 5 covers evidence and device gates. Stage 2 intentionally remains blocked behind a real Stage-1 tile. Interfaces are consistently C++17 pointer-plus-count and Kotlin direct `ByteBuffer`; all required design names and fixed dimensions are explicit.
