# pnnx Build Report

## LibTorch

- Package: `libtorch-win-shared-with-deps-2.7.0+cpu.zip`
- Official URL: `https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.0%2Bcpu.zip`
- Downloaded size: `196,059,242` bytes (~187 MiB)
- Extracted size: `1,154,297,400` bytes (~1.08 GiB)
- Final path: `V:\AI\LibTorch\2.7.0+cpu`
- Key files verified:
  - `lib\torch.lib`
  - `lib\torch_cpu.dll`
  - `lib\c10.dll`
  - `share\cmake\Torch\TorchConfig.cmake`

## Source

- ncnn source: `D:\AIProjects\third_party\ncnn`
- pnnx source: `D:\AIProjects\third_party\ncnn\tools\pnnx`
- pnnx is bundled in the ncnn repository.

## Commit

- Branch: `master`
- Commit: `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`

## CMake Config

- Build directory: `D:\AIProjects\seedvr2-ncnn\build\pnnx-cpu`
- Generator: Ninja 1.12.1
- Compiler: MSVC 19.44.35228 / x64
- Configuration: Release
- Torch root: `V:\AI\LibTorch\2.7.0+cpu\share\cmake\Torch`
- CUDA was not detected during the successful configuration.
- TorchVision was not found; pnnx configured to build without TorchVision.
- Protobuf was not found; onnx-zero support was disabled.

## Build

- Configure: PASS
- Build: PASS
- Build time: approximately 179 seconds
- Warnings: optional TorchVision and Protobuf unavailable; no build errors.

## Binary

- `D:\AIProjects\seedvr2-ncnn\build\pnnx-cpu\src\pnnx.exe`
- Size: `4,024,832` bytes
- `pnnx --help`: PASS; usage displayed and no missing-DLL error occurred when LibTorch `lib` was temporarily added to PATH.
- System PATH was not permanently modified.

## Minimal Conversion Test

- Test model: `Linear(10,4) + ReLU`
- Script: `D:\AIProjects\seedvr2-ncnn\pnnx-minimal-test.py`
- Input shape: `[1,10]`
- TorchScript output shape: `(1,4)`
- pnnx conversion: PASS
- Generated files:
  - `linear_relu.pnnx.param`
  - `linear_relu.pnnx.bin`
  - `linear_relu.ncnn.param`
  - `linear_relu.ncnn.bin`
  - `linear_relu_pnnx.py`
  - `linear_relu_ncnn.py`

## ncnn Load Result

- Test source: `D:\AIProjects\seedvr2-ncnn\ncnn-load-test\main.cpp`
- Test build: `D:\AIProjects\seedvr2-ncnn\build\ncnn-load-test2`
- Result: `param_ret=0 model_ret=0`
- ncnn successfully loaded both generated `.ncnn.param` and `.ncnn.bin` files.

## Problems

- The first pnnx build directory retained a CUDA Torch cache and was not usable; a separate `pnnx-cpu` build directory was used without modifying source.
- TorchVision static library was unavailable, so optional TorchVision operator support is not built.
- Protobuf was unavailable, so optional onnx-zero support is not built.
- No SeedVR2 weights or video data were downloaded.
- `D:\AIProjects\penguin-vl-ncnn-6788` was not modified.

## Final Status

| Capability | Status |
|---|---|
| pnnx build | READY |
| pnnx help | READY |
| PyTorch to pnnx | READY |
| ncnn load result | READY |
