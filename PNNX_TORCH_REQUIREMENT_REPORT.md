# pnnx Torch Dependency Analysis

## Current pnnx version

- Source: `D:\AIProjects\third_party\ncnn\tools\pnnx`
- ncnn commit: `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`
- The current CMake requires `find_package(Torch REQUIRED)`.
- Minimum version enforced by CMake: PyTorch `>= 1.8`.
- PyTorch `>= 2.1` causes pnnx to use C++17 headers.

## Current PyTorch environment

- Python: `D:\AIProjects\seedvr2-ncnn\.conda-cuda\python.exe`
- PyTorch: `2.7.0+cu128`
- CUDA runtime: `12.8`
- Torch CMake path: `D:\AIProjects\seedvr2-ncnn\.conda-cuda\Lib\site-packages\torch\share\cmake`
- The installed Python wheel contains Torch headers, libraries, and CMake files, but its CMake configuration is CUDA-enabled.

## Required LibTorch

- Source builds require the C++ LibTorch distribution or a compatible PyTorch installation exposing a complete Torch CMake package.
- pnnx's `PNNXPyTorch.cmake` can inspect Python `torch.utils.cmake_prefix_path`, but this only locates the Torch CMake package; it does not remove the C++ library dependency.
- pnnx README additionally states that PyTorch and the TorchVision C++ library are installed before building. TorchVision is optional in the current CMake and falls back to building without it.

## CPU-only feasibility

- CPU-only LibTorch is supported by the official PyTorch distribution and is the appropriate choice for the requested CPU-only pnnx build.
- The current pnnx CMake has no project-defined `PNNX_USE_CUDA=OFF` or equivalent switch.
- Passing `-DUSE_CUDA=OFF` is not consumed by pnnx and did not prevent the CUDA Torch package from being loaded.
- A CPU-only LibTorch package should avoid the current `CUDA::nvToolsExt` generation failure.

## CUDA requirement

- pnnx itself does not require CUDA for a CPU conversion tool.
- Using the current `2.7.0+cu128` Python/Torch CMake package causes Caffe2/Torch CMake to probe CUDA and fail on the missing `CUDA::nvToolsExt` target in this environment.
- CUDA support is therefore not recommended for this pnnx build stage.

## Recommended download

- Official Windows CPU package matching the current Torch release:
  `libtorch-win-shared-with-deps-2.7.0+cpu.zip`
- Official URL:
  `https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.0%2Bcpu.zip`
- Official Windows CUDA alternative, not recommended for this stage:
  `libtorch-win-shared-with-deps-2.7.0+cu128.zip`

## Expected size

HEAD-only checks; no files were downloaded:

- CPU LibTorch compressed size: `196,059,242` bytes, about `187 MiB`
- CUDA LibTorch compressed size: `3,278,523,390` bytes, about `3.05 GiB`
- Extracted size was not measured because no archive was downloaded.

## Target path

Recommended target after confirmation:

`V:\AI\LibTorch\2.7.0+cpu`

## Next action

After explicit confirmation, download and extract only the official CPU LibTorch 2.7.0 archive to the target path, configure pnnx with `Torch_INSTALL_DIR`, and build without changing pnnx CMake or the existing repository.

## Sources

- [pnnx official repository](https://github.com/pnnx/pnnx)
- [ncnn official repository](https://github.com/Tencent/ncnn)
- [PyTorch C++/LibTorch installation](https://docs.pytorch.org/cppdocs/installing.html)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
