# Dynamic RMS Hybrid Multi-Shape Runtime

Phase 9.5 validates one metadata-driven Dynamic RMS Hybrid runtime against real
PyTorch references. Dynamic A is `[1,4,8,8,32]` without padding; Dynamic B is
`[1,4,9,10,32]` with spatial padding `[0,3,2]`; Dynamic C is `[1,5,9,10,32]`
with temporal and spatial padding `[3,3,2]`.

All three profiles use variant `dynamic_rms_rope3d_hybrid` and the same per-window
`[64,32]` ncnn contract. Their CoreKeys differ by grid: A `[1,2,2]`, B `[1,3,3]`,
C `[2,3,3]`. B has 360 valid and 216 invalid tokens (13,824 masked entries);
C has 450 valid and 702 invalid tokens (44,928 masked entries).

The validation sequence is A → B → C → A. It recreates dynamic contracts per
request, verifies the final A is bit-identical, and records path-keyed cache
growth 5 → 10 → 15. See
[dynamic_rms_multi_shape_runtime_validation.json](D:\AIProjects\seedvr2-ncnn\reports\dynamic_rms_multi_shape_runtime_validation.json).

## Status

Phase 9.4 typed metadata/runtime routing is READY. Phase 9.5 multi-shape RMS
Hybrid Runtime is READY for these three FP32 CPU fixture profiles. Vulkan,
head_dim=128, and real SeedVR2 weights remain TODO.
