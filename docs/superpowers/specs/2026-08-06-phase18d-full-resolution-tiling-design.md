# Phase 18D full-resolution tiled enhancement design

Status: approved architecture, implementation pending.

## Objective and scope

Phase 18D adds an Android workflow that preserves a selected photo's complete
EXIF-corrected image and pixel dimensions, runs the existing fixed 128 x 128
SeedVR2 runtime over overlapping tiles, and eventually blends the tiles into an
output PNG with the same width and height. This is same-resolution restoration,
not 2x or 4x super-resolution.

The work is split into two sequential stages.

Stage 1 implements and validates the shared tile core, Android JNI contract,
checkpoint contract, and one real gallery-photo tile (`tile_0000`) through VAE
Encoder, all 32 DiT blocks, scheduler, and VAE Decoder. It plans the complete
photo but does not run or blend every model tile.

Stage 2 starts only after Stage 1 passes. It adds serial multi-tile execution,
foreground-service scheduling, pause/resume/cancel, full-image accumulation,
result UI, and MediaStore export. Identity fusion is never evidence of real
model execution.

## Architecture

The C++17 core is the single implementation of tile planning, extraction,
padding, valid-region calculation, blend weights, accumulation, checkpoint
validation, single-tile inference, and result validation. Host tests and the
Android library compile the same source files.

Kotlin owns Android integration: Photo Picker and `content://` access, private
source-file copy, source hashing, decoding, EXIF correction, conversion to a
direct contiguous RGB8 buffer, task lifecycle, UI, foreground service,
notifications, MediaStore, and JNI calls. Kotlin must not reimplement tile
coordinates, extraction, padding, valid regions, blend weights, or checkpoint
validation.

## Canonical data flow

```text
Photo Picker
-> Kotlin copies source and computes source_sha256
-> Kotlin decodes, applies EXIF, converts to contiguous RGB8 HWC
-> Kotlin computes normalized_image_sha256
-> JNI receives RGB buffer, dimensions, hashes, and task request
-> C++ recomputes normalized_image_sha256 and rejects disagreement
-> C++ TilePlanner creates the complete tile_plan
-> C++ extracts tile_0000 and pads it to 128 x 128
-> C++ computes tile_input_sha256
-> C++ normalizes RGB8 as value / 127.5 - 1 and repeats it for T=5
-> real VAE Encoder
-> real 32-block DiT streaming
-> scheduler endpoint
-> real VAE Decoder
-> target_frame_index=0
-> C++ crops to the tile's valid region and writes output_valid.rgb
-> Kotlin verifies byte count and encodes tile_0000_output.png
-> Kotlin records PNG SHA-256
```

The validated temporal contract is input `[C=3,T=5,H=128,W=128]`, decoder
output T=5, and target frame 0. The input photo tile is repeated identically
five times. This design does not substitute a centre frame.

## Shared C++ interfaces

The project remains C++17. Interfaces use pointer plus byte count rather than
upgrading the project for `std::span`.

```cpp
struct TileConfig {
    int tile_size = 128;
    int overlap = 32;
    int stride = 96;
};

struct TileSpec {
    int tile_index;
    int grid_x;
    int grid_y;
    int source_left;
    int source_top;
    int source_right;
    int source_bottom;
    int valid_width;
    int valid_height;
    int padding_left;
    int padding_top;
    int padding_right;
    int padding_bottom;
};

TilePlan plan_tiles(int image_width, int image_height,
                    const TileConfig& config);

PaddedTile extract_and_pad_rgb8(const uint8_t* image_rgb,
                                size_t byte_count,
                                int image_width,
                                int image_height,
                                const TileSpec& tile);

FusionResult identity_fuse(const TilePlan& plan,
                           TileProvider tile_provider);

SingleTileResult run_single_tile_seedvr2(
    const SingleTileRequest& request,
    ProgressCallback progress,
    CancelCallback cancelled);
```

Names may follow existing code style, but responsibilities and validation must
not be weakened.

## Kotlin and JNI interfaces

```kotlin
external fun nativePlanPhoto(
    normalizedRgb: ByteBuffer,
    width: Int,
    height: Int,
    requestJson: String
): String

external fun nativeRunSinglePhotoTile(
    modelRoot: String,
    taskRoot: String,
    normalizedRgb: ByteBuffer,
    width: Int,
    height: Int,
    requestJson: String
): String
```

The buffer is direct, RGB8 HWC, row-major, sRGB, without alpha, with capacity
exactly `width * height * 3`. Kotlin retains it for the synchronous JNI call.
Native reads it only during that call, never stores its pointer, and never
modifies it. Native writes `output_valid.rgb`; Kotlin encodes the PNG.

## Hash and model identity contract

`source_sha256` hashes every byte of the private copy of the selected source
file. `normalized_image_sha256` hashes the complete EXIF-corrected contiguous
RGB8 HWC byte stream. `tile_input_sha256` hashes the padded 128 x 128 x 3 RGB8
tile. Digests are lowercase 64-character hexadecimal strings.

Checkpoint identity binds task ID, normalized image hash, tile input hash,
model manifest SHA-256, tile index and bounds, shape, dtype, byte count,
checkpoint SHA-256, and finite status. The source hash is provenance, but JPEG
metadata differences do not invalidate pixel-identical recovery. The model
version comes from the verified manifest digest, never UI text.

## Planner, padding, and fusion

For each axis, starts are `0, 96, 192, ...`; the final start is anchored at
`max(0, dimension - 128)`, then values are sorted and deduplicated. Every source
pixel is covered, and no tile is out of bounds.

Tiles smaller than 128 use reflect padding per axis only when valid length is
greater than one and each padding extent is smaller than that length. An axis
otherwise uses replicate, including a length-one valid axis. The actual mode
for each axis is recorded. Only the valid region is returned from model output.

The blend window is separable. For overlap sample `k` of `n`, the leading
tile's trailing weight is `cos((pi/2)*(k+0.5)/n)^2` and the trailing tile's
leading weight is the complementary squared sine. Non-overlap and photo outer
edges have weight 1. Accumulation uses FP32 `weighted_sum` and `weight_sum`, then
divides by `max(weight_sum, epsilon)`. No valid pixel may have zero weight.
Stage 1 exercises this only with identity tiles.

## Files and persistence

```text
files/tasks/<task_id>/
|-- source.<extension>
|-- normalized_image.rgb
|-- input_metadata.json
|-- tile_plan.json
|-- runtime_progress.json
|-- task_report.json
`-- tiles/tile_0000/
    |-- status.json
    |-- input_padded.rgb
    |-- output_valid.rgb
    `-- tile_0000_output.png
```

Large tensor checkpoints are created only by real execution. Stage 1 must not
read the Phase 18C.4 checkerboard Block32, dec_up0, or dec_up1 checkpoints.

## Errors and state

All native responses contain `ok`, stable `error_code`, human-readable
`error_message`, and stage-specific fields. The stable codes and legal state
transitions are specified in `docs/PHASE18D_ERROR_AND_STATE_CONTRACT.md`.
Invalid or corrupted checkpoint metadata is rejected before tensors are read.

## Verification gates

Gate 1 runs shared Host tests for Planner coverage, padding and fallback,
hashing, checkpoint validation, identity fusion, state transitions, and error
serialization.

Gate 2 runs Android JNI contract tests for direct-buffer validation, capacity,
native hash recomputation, request parsing, tile-plan JSON, errors, and pointer
lifetime.

Gate 3 imports an existing real Pad photo when available, computes all three
hash levels, and generates its complete tile plan. A repository fixture may
exercise the technical path only when clearly reported as `fixture_photo=true`
and `gallery_photo=false`.

Gate 4 runs `tile_0000` through the complete real model and produces a decodable
PNG of `valid_width x valid_height`. PASS requires Encoder complete, exactly 32
DiT blocks complete, scheduler complete, Decoder complete, finite output,
NaN=0, Inf=0, and output bytes not equal to copied input bytes.

Only Gate 4 permits `Phase 18D Real Photo Tile = PASS`. Stage 2 gates and final
gallery-MVP status remain separate.

## Safety and repository policy

Existing models, checkpoints, reports, outputs, and stashes are preserved. User
photos, generated images, tensors, checkpoints, APKs, libraries, device IDs,
and raw logs remain untracked. No push, tag, or pull request is created. Full
Phase 18 remains NOT READY unless its separate video criteria are met.
