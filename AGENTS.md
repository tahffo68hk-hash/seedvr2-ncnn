# SeedVR2 ncnn fixture guidance

Goal: validate a SeedVR2-style adaptive-window attention subsystem on ncnn CPU FP32. Current phase: Phase 9.5 READY. Build with `D:\DevTools\VisualStudio\BuildTools\Common7\Tools\VsDevCmd.bat`, ncnn at `build/ncnn-cpu/install`, and clean validation build `build/phase95-dynamic-multishape-release`.

Do not modify `D:\AIProjects\penguin-vl-ncnn-6788`, `D:\AIProjects\third_party\ncnn`, or `D:\AIProjects\third_party\SeedVR2` unless explicitly requested. Do not download dependencies/weights, alter model binaries, or claim PASS without execution. Do not silently fall back between variants or auxiliary modes. FP32 parity artifacts require `pnnx fp16=0`.

CoreKey is `(window, grid, variant)`; fixed artifacts use reference mask/RoPE and dynamic A/B/C use padding-validity plus DynamicMMRoPE3D. Cache is path-keyed. Use a new clean build after public C++ ABI/type-layout changes. Next task: extract the real single SeedVR2 DiT block interface/shape contract from local `third_party/SeedVR2` (including the true MM-RoPE3D axis split) after resolving `rotary-embedding-torch` availability. Phase 10 synthetic head_dim=128 is PARTIAL; do not claim real-SeedVR2 RoPE parity.
