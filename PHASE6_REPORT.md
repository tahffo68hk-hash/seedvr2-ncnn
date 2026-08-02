# Phase 6 Runtime Report

Runtime Prototype: READY (Phase 6.3)

- ArtifactRegistry scans five manifest-backed FP32 static cores atomically.
- Dispatcher uses `AttentionRequest -> Metadata -> CoreKey -> ArtifactRegistry::lookup` and fails closed.
- Release CPU runtime uses a `NcnnModelCache`, host partition/mask/reverse-crop, and native ncnn attention graphs.
- Five real reference inputs A–E passed ncnn extraction and BTHWC comparison with max error below `1e-4`; exact measurements and 10-run cached benchmark statistics are in `reports/runtime_validation.json`.
- The negative shape `[1,4,99,99,32]` was rejected before model load/inference.

The only supported profiles are the five checked-in static window/grid variants.
RoPE, Q/K Norm, text conditioning, varlen attention, Vulkan, dynamic arbitrary
shape export, and SeedVR2 weight integration remain out of scope.
