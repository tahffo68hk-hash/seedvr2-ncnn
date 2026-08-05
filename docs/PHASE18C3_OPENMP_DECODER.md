# Phase 18C.3 OpenMP decoder

The Android arm64-v8a build enables the pinned ncnn OpenMP path. With NDK 26.3.11579264, the ncnn source selects `-fopenmp -static-openmp`; no copied or dynamically loaded OpenMP shared object is used.

On Xiaomi Pad 7 Pro, runtime reports OpenMP compiled and available, eight logical CPUs, and a five-core ncnn balanced capacity. A real FP32 ncnn `vae/decoder/dec_conv_in` stage was measured with one warmup and three fresh Net/Extractor load-and-execute runs at 1/2/4/5/6/8 threads: 519.667 / 310.667 / 159.667 / 149.667 / 178.333 / 145.667 ms. `/proc/self/task/*/stat` CPU tick sampling observed 1/2/4/5/6/8 active CPU threads and all outputs were finite and bit-identical to the one-thread reference. Five threads are selected as balanced: 3.472x faster than one thread and within 2.7% of the eight-thread minimum without exceeding the reported balanced capacity. See `reports/phase18c3_thread_benchmark.json`.

Decoder-only now reports all seven real decoder stages. The first checkpoint-resume attempt completed `dec_conv_in` (213 ms), `dec_mid` (24,291 ms), and `dec_up0` (77,478 ms); `dec_up1` is the confirmed bottleneck and did not complete. No PNG was generated. Xiaomi froze the instrumentation process when it was backgrounded; bringing SeedEdge foreground resumed it. This is a real device limitation, not a completion claim.
