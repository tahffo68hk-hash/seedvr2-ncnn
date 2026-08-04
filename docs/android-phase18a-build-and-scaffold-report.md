# Phase 18A Android ARM64 build and scaffold report

Status: **READY_FOR_DEVICE**. Full Phase 18 is **NOT READY**.

Phase 17 remains READY at `13b2d14`. Phase 18A created the device-free `android/` project for `com.seededge.runtime` (**SeedEdge Runtime**) and built Debug plus unsigned Release APKs for `arm64-v8a` only. The build uses Gradle 8.9, AGP 8.7.3, Kotlin plugin 2.0.21, JDK 17.0.20, NDK 26.3.11579264, CMake 3.22.1, and SDK 35.

ncnn commit `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a` is compiled from source with Vulkan enabled and fixed glslang commit `fe88f421038e1bb0a25cd5c1b2dfe505db82d08f`; OpenMP, C++ exceptions, and RTTI are disabled. `libseededge.so` statically links ncnn and is packaged as AArch64. The five JNI methods return structured JSON. The shared synthetic ncnn CPU logic passed on the Windows host (max absolute error 0, finite, repeat-consistent) and is compiled into the Android binary, but has not run on an Android device. Host-side manifest tests passed 9/9.

APK inspection found only `arm64-v8a`, no SeedVR2 weights/models, no keystore, and no sensitive local paths. The Release APK is unsigned and is not described as production signed. Core tools, caches, temporary data, and build outputs are on D:; the pre-existing small `C:\Users\<user>\.android` adb/config directory was observed read-only and not modified or removed.

No Xiaomi Pad was connected. No Android ARM64 execution, Vulkan device probe/inference, real VAE/DiT inference, or full video restoration was performed. Phase 18B begins after USB connection with `adb devices -l`, followed by install/device contract tests.
