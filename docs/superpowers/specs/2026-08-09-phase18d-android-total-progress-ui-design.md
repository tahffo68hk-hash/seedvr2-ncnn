# Phase 18D Android Total Progress UI Design

## Goal

Make the Pad test UI explain how a selected photo is tiled and show whole-photo progress instead of a repeating per-stage progress bar.

## Approved Behavior

After a photo is selected, the enhancement screen shows the corrected image size, tile size, overlap, stride, grid columns, grid rows, total tile count, selected runtime mode, cohort count, estimated runtime, estimated peak disk, and required free disk. It also shows a compact text tile map where each tile is represented by a cell.

During execution, the primary progress bar represents the whole photo. It combines completed tiles from `status.json` with current native `runtime_progress.json` phase information, so a long tile still advances through model check, VAE encoder, DiT block 1..32, VAE decoder, frame writing, fusion, PNG encoding, and save. The detail line keeps the current stage visible.

User-facing Chinese strings are repaired where the Phase 18D Android screen currently displays mojibake. Native inference, checkpointing, fusion math, runtime selection, model validation, and artifacts are unchanged.

## Components

- `PhotoTask`: owns pure planning and progress math helpers that can be unit-tested without Android UI.
- `MainActivity`: displays the tile plan, text tile map, whole-photo progress bar, and readable Chinese labels.
- `PhotoTaskTest`: verifies tile plans, progress fractions, and labels.

## Testing

Use Android JVM unit tests for pure helper behavior, then run the app module unit test suite and assemble a debug APK. No Pad PASS claim is made from host-side tests.
