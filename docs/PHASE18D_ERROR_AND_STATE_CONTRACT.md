# Phase 18D error and state contract

## Response envelope

Every JNI operation returns JSON with `schema_version`, `ok`, `error_code`, and
`error_message`. Success uses `error_code="OK"` and an empty message. Failures
add the most precise available `stage`, `tile_index`, `block_index`, or decoder
stage. Unknown exceptions are caught at the JNI boundary and mapped to
`NATIVE_EXCEPTION`; `UNKNOWN_ERROR` is the final fallback.

```json
{
  "schema_version": 1,
  "ok": false,
  "error_code": "DIT_BLOCK_FAILED",
  "error_message": "block execution failed",
  "stage": "dit",
  "tile_index": 0,
  "block_index": 17
}
```

Stable codes are:

```text
OK
INVALID_REQUEST_JSON INVALID_SCHEMA_VERSION INVALID_TASK_ID
INVALID_MODEL_VERSION INVALID_IMAGE_DIMENSIONS INVALID_RGB_BUFFER
RGB_BUFFER_SIZE_MISMATCH NORMALIZED_IMAGE_SHA_MISMATCH
INVALID_TILE_CONFIG INVALID_TILE_INDEX INVALID_TILE_BOUNDS
INVALID_PADDING_CONTRACT
MODEL_ROOT_MISSING MODEL_MANIFEST_MISSING MODEL_MANIFEST_MISMATCH
MODEL_FILE_MISSING MODEL_FILE_CORRUPT
CHECKPOINT_METADATA_MISSING CHECKPOINT_METADATA_INVALID
CHECKPOINT_TASK_MISMATCH CHECKPOINT_INPUT_MISMATCH
CHECKPOINT_TILE_MISMATCH CHECKPOINT_MODEL_MISMATCH
CHECKPOINT_SHAPE_MISMATCH CHECKPOINT_SIZE_MISMATCH
CHECKPOINT_SHA_MISMATCH CHECKPOINT_NON_FINITE
ENCODER_FAILED DIT_BLOCK_FAILED SCHEDULER_FAILED DECODER_FAILED
OUTPUT_FRAME_FAILED OUTPUT_RGB_WRITE_FAILED OUTPUT_PNG_FAILED
CANCELLED THERMAL_PAUSED OUT_OF_MEMORY OUT_OF_STORAGE
NATIVE_EXCEPTION UNKNOWN_ERROR
```

## Task state machine

```text
CREATED -> PHOTO_IMPORTED -> PHOTO_NORMALIZED -> PLAN_READY
PLAN_READY -> TILE_PREPARED -> RUNNING_ENCODER -> RUNNING_DIT
RUNNING_DIT -> RUNNING_SCHEDULER -> RUNNING_DECODER
RUNNING_DECODER -> WRITING_OUTPUT -> COMPLETED
```

Running states may enter `CANCEL_PENDING`, then `CANCELLED` at a safe boundary.
Stage 2 running states may enter `THERMAL_PAUSED` after a checkpoint and resume
only after checkpoint validation. Any non-recoverable error enters `FAILED`.
`FAILED` and `CANCELLED` never transition directly to `COMPLETED`.

Tile states are `pending`, `preparing`, `encoder`, `dit`, `scheduler`,
`decoder`, `writing_output`, `completed`, `cancelled`, and `failed`. Skipping
forward, such as `pending -> decoder`, is invalid. A resume first validates the
complete checkpoint identity and then re-enters the stage after the checkpoint.

Status files are written through a temporary file followed by an atomic rename.
They contain the task and tile identity, current stage, completed block count,
per-stage booleans, output paths, finite counts, and structured error.

Kotlin maps stable codes to concise user messages and preserves native detail in
the task report. It does not infer success from a progress label.
