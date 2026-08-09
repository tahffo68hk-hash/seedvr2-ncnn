# Phase 18D Hybrid Host Productization Design

## Scope and status rules

This design advances the existing real CPU FP32 full-photo implementation without replacing the validated tile planner, padding, single-tile runtime, or model artifacts. Host work may be marked PASS only from executed evidence. Xiaomi Pad 7 Pro work remains `READY_FOR_DEVICE` or `NOT_RUN`; complete Phase 18 remains `NOT READY`.

## Runtime architecture

The public runtime selector has four modes: `tile_major`, `stage_major`, `hybrid`, and `auto`. Hybrid partitions the deterministic tile plan into consecutive cohorts of `K` tiles. Each cohort executes the existing stage-major stage order, keeps at most one weighted model stage resident, publishes crash-safe checkpoint generations, fuses completed tile outputs only after all cohorts finish, then reclaims obsolete cohort state. Tile and output mathematics remain CPU FP32 and unchanged.

Auto selection consumes tile count, measured calibration data, resume requirement, available disk, estimated peak disk, and memory budget. Explicit user mode and cohort size always override Auto. Reports record requested mode, selected mode, cohort size, and a stable selection reason.

## Crash-safe checkpoint lifecycle

Each checkpoint generation is written to a temporary generation directory, closed, SHA/size/identity validated, atomically published, and referenced by the task control file before older generations are reclaimed. Startup ignores partial temporary generations, validates published generations newest-first, and resumes from the newest valid generation. Production mode retains only resume-required state, metadata, and completed tile output; debug/profile mode may retain diagnostic tensors.

## Host product entry point

`tools/phase18d_enhance_photo.py` accepts JPEG, PNG, and WebP, applies Pillow EXIF transpose, converts to RGB8 HWC, creates or validates a collision-safe task identity, invokes the native runner, streams real progress from task metadata, and writes a same-size PNG. It supports new, resume, pause-boundary, cancel, explicit runtime mode, Auto, and cohort size.

## Evidence and gates

Before runtime changes, execute the current long-task implementation on a natural 1024x1024 input as the formal baseline. Execute a non-grid natural image gate. After Hybrid/GC implementation, use small real tasks to screen cohort sizes, then benchmark the most useful configurations on 256 and 512 only. Re-run an expensive gate only when its core runtime/checkpoint/fusion behavior changed.

Reports distinguish `MEASURED` from `ESTIMATED`. Seam evidence reports coverage, invalid/duplicate bounds, horizontal and vertical boundary mean/P95/max gradients, and nearby non-seam reference gradients; it does not claim perceptual quality.

## Android handoff

Android request/task metadata gains runtime mode, cohort size, checkpoint generation, estimated/actual disk, and real stage/tile/block progress. Host-derived Auto calibration is implemented but remains `READY_FOR_DEVICE` on Android until Pad measurements recalibrate thresholds.

## Safety constraints

- Preserve all four stashes, model/checkpoint/evidence directories, and historical files.
- Do not commit models, checkpoints, task outputs, images, APK/SO/A, or logs.
- No lossy FP16 checkpoint conversion and no simultaneous residency of multiple weighted stages.
- No push, tag, PR, destructive reset, clean, or stash deletion.
