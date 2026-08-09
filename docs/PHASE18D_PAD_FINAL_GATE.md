# Phase 18D Xiaomi Pad 7 Pro final gate

Status: **READY_FOR_DEVICE / NOT_RUN**. Pad absence is not a development blocker and is not PASS.

1. Confirm `adb devices -l`, model identity, storage, battery, and thermal baseline.
2. Install the arm64-v8a APK and run `scripts/phase18d_pad_full_photo_gate.ps1`.
3. Run natural 256x256 then 512x512 photos; require all tiles and blocks, finite/no-fallback output, zero fusion holes, and complete PNG/diff/report artifacts.
4. During 512, background and lock the Pad, pause at a committed boundary, recreate the Activity, resume, and compare SHA with an uninterrupted run.
5. Verify notification Pause/Resume/Cancel, WakeLock release while paused/terminal, orientation and MediaStore save.
6. Collect battery/thermal/meminfo/logcat with `scripts/phase18d_pad_after_benchmark.ps1`.

No item becomes PASS without executed artifacts and threshold results.
