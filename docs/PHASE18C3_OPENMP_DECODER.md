# Phase 18C.3 OpenMP decoder

The Android arm64-v8a build enables the pinned ncnn OpenMP path. With NDK 26.3.11579264, the ncnn source selects `-fopenmp -static-openmp`; no copied or dynamically loaded OpenMP shared object is used.

On Xiaomi Pad 7 Pro, runtime reports OpenMP compiled and available, eight logical CPUs, and a five-core ncnn balanced capacity. This validates build/runtime capability only. A real ncnn stage benchmark must still demonstrate worker participation and choose the final 2/4/6/8-thread policy.
