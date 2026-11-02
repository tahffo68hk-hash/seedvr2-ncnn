# SeedVR2 ncnn fixture guidance

Goal: validate a SeedVR2-style adaptive-window attention subsystem on ncnn CPU FP32. Current phase on this branch: Phase 9.5 READY. This is not a full SeedVR2 port.

Build ncnn from the pinned commit in `.github/workflows/cpu.yml` and [README.md](README.md) (`a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`). Install it to `build/ncnn-cpu/install` and pass that prefix as `CMAKE_PREFIX_PATH`. After a public C++ ABI or type-layout change, use a new clean validation build directory such as `build/phase95-dynamic-multishape-release`.

Do not modify external checkouts of penguin-vl-ncnn, ncnn, or SeedVR2 unless explicitly requested. Do not download official weights, alter model binaries, or claim PASS without execution. Do not silently fall back between variants or auxiliary modes. FP32 parity artifacts require `pnnx fp16=0`.

CoreKey is `(window, grid, variant)`; fixed artifacts use reference mask/RoPE and dynamic A/B/C use padding-validity plus DynamicMMRoPE3D. Cache is path-keyed. Next task: Phase 10 synthetic head_dim=128 shape stress validation.
