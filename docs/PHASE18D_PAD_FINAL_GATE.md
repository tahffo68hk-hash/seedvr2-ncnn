# Phase 18D Xiaomi Pad 7 Pro final gate

Status: **READY_FOR_DEVICE / NOT_RUN**. Pad absence is not a development blocker and is not PASS.

1. Confirm `adb devices -l`, model identity, storage, battery, and thermal baseline.
2. Install the arm64-v8a APK and run `scripts/phase18d_pad_full_photo_gate.ps1 -TaskId <id>` so task metadata is collected with the device snapshot.
3. Run natural 256x256 then 512x512 photos; require all tiles and blocks, finite/no-fallback output, zero fusion holes, and complete PNG/diff/report artifacts. Record requested/selected runtime, selection reason, cohort size, checkpoint generation, estimated/actual disk and real stage/tile/block progress.
4. During 512, background and lock the Pad, pause at a committed boundary, recreate the Activity, resume, and compare SHA with an uninterrupted run.
5. Verify notification Pause/Resume/Cancel, WakeLock release while paused/terminal, orientation and MediaStore save.
6. Collect battery/thermal/meminfo/swap/data-disk/power/notification/logcat and task generation evidence with `scripts/phase18d_pad_after_benchmark.ps1`.

Host calibration selects TileMajor for at most two tiles, StageMajor for 3–25 tiles when its estimated task disk fits, and Hybrid K=4 for longer tasks, resume, or bounded-disk fallback. This is implemented on Android but is **READY_FOR_DEVICE_RECALIBRATION / NOT_RUN**; do not treat Host thresholds as Pad performance evidence.

No item becomes PASS without executed artifacts and threshold results.
