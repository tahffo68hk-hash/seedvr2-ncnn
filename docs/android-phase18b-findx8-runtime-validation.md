# Phase 18B Find X8 runtime validation

Status: **PASS** for the temporary Find X8 runtime gate. Complete Phase 18 remains **NOT READY**. Find X8 temporary device validation is not Xiaomi Pad 7 Pro final validation.

Executed on PKB110, Android 16 / API 36, arm64-v8a. No device serial is retained. The pinned ncnn build is 20260730 at `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`.

The original CPU result was finite data, not NaN/Inf. Its `4.88311052e-5` error came from ncnn defaults enabling FP16 packed/storage/arithmetic. The old implementation then mislabeled the combined failure as `finite=false`, short-circuited the second run, and compared two max-error scalars instead of outputs. The replacement forces CPU FP32, one thread, disables packing/FP16/INT8, creates a fresh Net/Extractor/Mat for each of three runs, and reports finite, expected parity, bitwise equality and numeric repeat separately. Find X8 completed 20/20 executions; all 60 inference runs produced `[-0.200000003,-0.100000001,2,3]`, max error 0 and repeat error 0.

Vulkan selected Mali-G925-Immortalis MC12 (vendor 5045), API raw 4206870 and driver raw 205524992. The test creates the ncnn device, blob/staging allocators, explicitly uploads the same input, executes the Vulkan graph, downloads output and fails instead of falling back. An intermediate SIGSEGV exposed that a weightless graph still needs `load_model()` to create its Vulkan pipelines; that lifecycle defect was fixed before acceptance. The final 10/10 executions passed, CPU/GPU max and mean absolute error were 0, `silent_cpu_fallback=false`, cleanup passed, and the final crash buffer was empty.

The Activity now executes JNI work on a single background executor, atomically rejects overlap, disables both buttons while busy, posts immutable bounded text to the main thread, restores controls after success/failure, converts thrown JNI errors to JSON, and shuts the executor down with the Activity. The existing Android 16 plain-string/16K/`BufferType.NORMAL` crash fix remains intact.

Evidence is in `reports/phase18b_findx8_*.json`. Next: real VAE, real DiT, streaming and minimal video device gates, followed by final Xiaomi Pad 7 Pro acceptance.
