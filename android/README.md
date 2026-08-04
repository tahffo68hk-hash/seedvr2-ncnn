# SeedEdge Android Runtime (Phase 18A)

Device-free ARM64 scaffold for `com.seededge.runtime` / **SeedEdge Runtime**. It builds only `arm64-v8a`, keeps SeedVR2 models external under `<external-files-dir>/models/seedvr2/`, and does not claim Android device, Vulkan inference, VAE, DiT, or restoration execution.

Run `scripts/check_environment.ps1`, `scripts/prepare_ncnn.ps1`, then `scripts/build_debug.ps1`. All caches and build outputs remain on D:.
