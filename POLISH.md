# Portfolio cleanup and tablet rerun

Date: 2026-10-24

## Branch and source

- Work is on local branch `polish-portfolio`, created from `codex/phase18d-progress-ui` at `cbcf72576a6e969d9936c8ae7d4a19134dd274a7`.
- No local `main` or remote `main` was found. `origin/HEAD` points to `codex/phase-7.5b-final` at `a01dc0d`, older than the selected experiment branch. GitHub read-only comparison showed `codex/phase18d-progress-ui` at the same `cbcf725` commit. Nothing was pushed.
- `git ls-remote` failed: `fatal: unable to access 'https://github.com/tahffo68hk-hash/seedvr2-ncnn.git/': schannel: failed to receive handshake, SSL/TLS connection failed`. The read-only branch/compare lookup was used to inspect upstream state.
- Preserved all four stashes and pre-existing untracked Phase 8.2 / Phase 18D files. Did not modify `penguin-vl-ncnn-6788` or the third-party ncnn checkout.

## Changes

- Rewrote `README.md` and `README.zh-CN.md` with the prototype scope, exclusions, Mermaid data flow, numerical alignment, performance tables, and desktop/Android build and run commands.
- Replaced hardcoded ncnn install paths in three fixture CMakeLists files with `find_package(ncnn CONFIG REQUIRED)`, configurable through `CMAKE_PREFIX_PATH` / `ncnn_DIR`.
- Added cosine similarity beside maximum absolute error in dynamic multi-shape and hybrid runtime JSON.
- Expanded the PyTorch reference trace with raw QKV, Q/K before and after RoPE, masked logits, merged heads, and projection tensors. Attention output computation is unchanged.
- Extended the dynamic multi-shape runner with selectable A/B/C profiles, three warm-ups, three timed iterations, JSON output, and Android `/proc/self/status` VmHWM reporting.
- Added `reports/tablet_dynamic_rms_multi_shape_validation.json`; refreshed `reports/dynamic_rms_multi_shape_runtime_validation.json` with the latest desktop run.
- Captured the four requested evidence images: actual Windows Terminal output for build, alignment, and desktop run; plus the tablet About screen from adb, cropped to exclude its personalized device-name row. The build image is cropped to the successful build summary to remove absolute build paths.
- Screenshot capture issue: the first whole-screen capture showed the unrelated foreground window and was discarded. Subsequent captures targeted the Windows Terminal window directly; the final build image is cropped to exclude local paths.
- Updated the Phase 17 Git-protection check to allow the four existing Phase 18C demo frames and the four named `demo-screenshots/*.png` portfolio evidence files; other binary-artifact protections remain in place.

## Desktop reproduction

- ncnn source commit: `a4d2ea1d4422`; installed package version: `20261024`.
- Toolchain: Visual Studio 2022 17.14.35, MSVC 19.44.35228.0, Windows x64, CPU FP32.
- Fresh ncnn configure/build/install succeeded in `build/ncnn-cpu`. The fixture configure/build succeeded in `build/polish-portfolio-release`.
- PyTorch reference-stage check passed: 1 test. Earlier runs exposed missing trace keys `qkv_raw` and `head0_q_rope`; the trace was completed and the next run passed.
- Before staging the local commit, the no-large-artifact check and the updated Phase 17 Git-protection check passed; the existing Phase 16 protection report also records PASS.
- Desktop A/B/C all passed. Each profile used 3 warm-ups and 3 timed iterations, one thread, FP32, Vulkan off. Peak process memory was not measured.

| Profile | Input BTHWC | Mean (ms) | Cosine | BTHWC max abs error | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| A | [1, 4, 8, 8, 32] | 1.105900000 | 1.0000000000 | 1.490116119e-7 | PASS |
| B | [1, 4, 9, 10, 32] | 2.176000000 | 1.0000000000 | 1.490116119e-7 | PASS |
| C | [1, 5, 9, 10, 32] | 3.324433333 | 1.0000000000 | 2.086162567e-7 | PASS |

Desktop peak memory remains unmeasured. This A/B/C run is shown in `demo-screenshots/02-alignment-result.png`.

## Android ARM64 tablet rerun

- `adb devices -l` first showed `unauthorized`. The user approved the device prompt; the next check showed status `device`.
- Device: Xiaomi Pad 7 Pro (name shown in the About screen); `ro.product.model=24091RPADC`.
- Chipset properties: `ro.soc.model=SM8635`; `ro.board.platform=pineapple`. The About screen labels it “第三代骁龙 8s 移动平台”.
- Android release: `16`; CPU ABI: `arm64-v8a`; `/proc/meminfo` `MemTotal=11787108 kB`.
- Built ncnn for Android ARM64 CPU FP32 with NDK `26.3.11579264`, Android API 24, Vulkan off. The fixture runtime uses `try/catch`, so ncnn was configured with `NCNN_DISABLE_EXCEPTION=OFF`. ncnn source itself was not changed.
- Build troubleshooting: the first validator configure did not find the ncnn CMake package; setting `ncnn_DIR` fixed it. The next build failed because the package propagated `-fno-exceptions`; rebuilding ncnn with exceptions enabled fixed the compile.
- Runtime used one ncnn thread. Each profile ran three warm-ups and three timed iterations in its own process. All three processes completed and emitted cosine, error, latency, and VmHWM, but each failed the existing `max_abs_error < 1e-4` check. The gate was not relaxed.

| Profile | Input BTHWC | Mean (ms) | Cosine | BTHWC max abs error | VmHWM (kB) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | [1, 4, 8, 8, 32] | 2.320485667 | 0.9999988949 | 0.001141190529 | 54056 | FAIL |
| B | [1, 4, 9, 10, 32] | 3.413993000 | 0.9999989426 | 0.001223027706 | 54680 | FAIL |
| C | [1, 5, 9, 10, 32] | 6.554218667 | 0.9999989158 | 0.001506924629 | 54632 | FAIL |

The runner printed these warnings before each profile, then completed normally:

```text
WARNING: linker: normalize_path - invalid input: "D", the input path should be absolute
WARNING: linker: Warning: unable to normalize "D" (ignoring)
```

The validator, fixtures, and temporary JSON files were confined to `/data/local/tmp/seedvr2/`. After pulling the reports and taking the screenshot, that exact directory was removed; `ls` confirmed it no longer exists. No app was installed, system setting changed, or root access used. The device screenshot documents the hardware, not terminal output, because the validator was invoked through `adb shell`.

## Still pending

- The Android numerical difference is not resolved: all tablet profiles exceed the existing `1e-4` maximum-error gate. The measurement is reported as FAIL without changing the gate.
- The Android linker still prints the `normalize_path` warning quoted above; execution is successful despite the warning.

## Resume description (under 150 Chinese characters)

端侧推理/ncnn：开发 SeedVR2 attention CPU FP32 原型；Pad 7 Pro（SM8635）A/B/C 2.32/3.41/6.55 ms，误差超1e-4；penguin-vl-ncnn 合入 PR #6834 full-width vision RoPE 修复。

Upstream reference: [Tencent/ncnn PR #6834](https://github.com/Tencent/ncnn/pull/6834), merged 2026-10-08. It adds full-width cos/sin cache handling for 2D/vision RoPE.
