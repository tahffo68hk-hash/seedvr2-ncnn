# Registry–Dispatcher Integration (Phase 6.3)

## Runtime flow

The Release runner receives an artifact root, Case C FP32 input, PyTorch
reference, and output path. It scans the root, builds
`AttentionRequest{1,4,17,17,32,"fp32"}`, calculates metadata, derives
`CoreKey{window_t,window_h,window_w,grid_t,grid_h,grid_w}`, then calls
`ArtifactRegistry::lookup`. `CoreInfo::single_model()` supplies the only ncnn
param/bin paths used by the baseline runner. It partitions the
input, creates an additive key mask, extracts ncnn output, reverse/crops BTHWC,
writes it, and compares it to the PyTorch reference.

`AttentionRequest` requires `batch == 1`, `channels == 32`, positive T/H/W,
and `dtype == "fp32"`; each rejection has a specific error string. Missing
keys are fail-closed: no nearest or previous core is used. The error includes
the requested full key and every available key.

## Case C verification

Artifact root: `tests/attention_dispatcher/artifacts/attention_core`.
Case C `[1,4,17,17,32]` selected `core_444_grid_1x5x5` with window `4x4x4`
and grid `1x5x5`. Release execution reported `extract=PASS`, max absolute
error `1.49012e-07`, mean absolute error `1.57646e-08`, `nan_count=0`, and
`inf_count=0`.

The old compiled `CoreRegistry` remains only for existing dispatcher contract
tests; the production runner does not construct or call it. The prior default
build failure was a Debug runner against Release `ncnn.lib` CRT mismatch. This
phase uses `build/attention-dispatcher-release` with `CMAKE_BUILD_TYPE=Release`
and links successfully without changing ncnn.

Model caching, automated real inference for all five cases, and benchmark or
runtime-validation artifacts remain future work.
