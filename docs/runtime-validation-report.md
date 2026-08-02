# Adaptive Window Runtime Validation

## Scope
Release ncnn CPU FP32 validation of five static profiles.

## Runtime Architecture
ArtifactRegistry â†?dispatcher â†?model cache â†?host partition/mask â†?ncnn â†?reverse/crop.

## Artifact Registry
Discovered 5 cores under `D:\AIProjects\seedvr2-ncnn\tests\attention_dispatcher\artifacts\attention_core`.

## Model Cache
Loaded 5 models; steady-state benchmark excludes first-load time.

## Five-Case Matrix

| Case | Core | Extract | Max error | Mean error |
|---|---|---|---:|---:|
|Case A|core_444_grid_1x4x4|PASS|1.78814e-07|1.55931e-08|
|Case B|core_344_grid_1x4x4|PASS|1.49012e-07|1.60619e-08|
|Case C|core_444_grid_1x5x5|PASS|1.49012e-07|1.57646e-08|
|Case D|core_444_grid_2x8x8|PASS|1.78814e-07|1.55457e-08|
|Case E|core_444_grid_2x5x6|PASS|2.08616e-07|1.6552e-08|

## Performance Baseline
CPU FP32, ncnn threads=1, three warmups and ten cached runs per case. See `reports/runtime_validation.json` for per-stage mean/min/max.

## Negative Shape Handling
`[1,4,99,99,32]`: safely rejected.

## Known Build Constraint
Debug runner cannot link Release ncnn CRT; validation uses a separate Release build.

## Current Limitations
Only five static profiles are supported. Arbitrary resolutions, RoPE, Q/K Norm, text conditioning, varlen attention, Vulkan, and SeedVR2 weights are not implemented.

## Phase 6.3 Status
READY

## Next Recommended Milestone
Integrate the cached static-core runtime into the broader SeedVR2 pipeline.
