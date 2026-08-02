# ncnn/pnnx Build Report

## ncnn source

- URL: https://github.com/Tencent/ncnn.git
- Path: `D:\AIProjects\third_party\ncnn`
- Branch: `master`
- pnnx is bundled at `tools\pnnx`; no separate pnnx clone was needed.

## pnnx source

- Path: `D:\AIProjects\third_party\ncnn\tools\pnnx`
- Relationship: ncnn repository subdirectory built through its CMake tree.

## Commit

- ncnn commit: `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`
- Commit date: 2026-07-22 14:13:06 +0800
- Working tree: clean at clone time.

## Build environment

- Generator: Ninja 1.12.1
- CMake: 3.31.6-msvc6
- MSVC: 19.44.35228 / toolset 14.44.35207
- Architecture: x64
- Configuration: Release
- Python/Torch environment: `D:\AIProjects\seedvr2-ncnn\.conda-cuda`, PyTorch 2.7.0+cu128

## CMake options

Actual ncnn options were confirmed in its top-level CMakeLists:

- `NCNN_VULKAN=OFF`
- `NCNN_BUILD_BENCHMARK=OFF`
- `NCNN_BUILD_TOOLS=ON`
- `NCNN_BUILD_EXAMPLES=ON`
- `NCNN_BUILD_TESTS=OFF`

Build directory: `D:\AIProjects\seedvr2-ncnn\build\ncnn-cpu`

## Build result

- ncnn CPU configure: PASS
- ncnn CPU Release build: PASS
- OpenMP: detected
- OpenCV: not found; examples requiring OpenCV were disabled
- Protobuf: not found; Caffe converter was disabled

## Generated binaries

- `D:\AIProjects\seedvr2-ncnn\build\ncnn-cpu\src\ncnn.lib`
- `D:\AIProjects\seedvr2-ncnn\build\ncnn-cpu\tools\ncnnoptimize.exe`
- `D:\AIProjects\seedvr2-ncnn\build\ncnn-cpu\tools\ncnn2mem.exe`
- `D:\AIProjects\seedvr2-ncnn\build\ncnn-cpu\tools\quantize\ncnn2int8.exe`
- `D:\AIProjects\seedvr2-ncnn\build\ncnn-cpu\tools\quantize\ncnn2table.exe`
- `benchncnn.exe` was not generated because `NCNN_BUILD_BENCHMARK=OFF`.

## Verification

- `ncnnoptimize.exe --help`: PASS; usage output displayed.
- Large benchmark: not run.
- `benchncnn --help`: SKIPPED; binary intentionally not built.
- ncnn Vulkan: not configured or built.
- CUDA: not enabled in ncnn.

## Problems

1. pnnx configuration found Torch 2.7.0 CMake files from the current CUDA PyTorch environment, but Torch's CMake configuration automatically required CUDA and the missing `CUDA::nvToolsExt` target caused generation failure.
2. pnnx's CMakeLists does not consume the manually supplied `USE_CUDA=OFF` variable.
3. pnnx was not built; no libtorch was downloaded. A CPU-only PyTorch/libtorch toolchain or a compatible standalone libtorch package must be evaluated before retrying. No package size was estimated or downloaded in this phase.

## Final Status

| Capability | Status |
|---|---|
| ncnn CPU | READY |
| pnnx | BLOCKED |
| ncnn tools | READY (selected tools) |
| Start model conversion | BLOCKED |
