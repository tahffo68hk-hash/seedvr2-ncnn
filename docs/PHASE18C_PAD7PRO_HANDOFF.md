# Phase 18C Pad 7 Pro handoff

Branch: `phase18c-android-minimal-video-demo`
HEAD at start: `2437856000a3532eb49303f2c6a09e7ccf4618c6`

Device: Xiaomi Pad 7 Pro candidate (`product:muyu`, model `24091RPADC`, Android 16/API 36, arm64-v8a, Adreno 735). Find X8 PKB110 evidence is separate and remains only the earlier block-9 partial result.

Completed on Pad 7 Pro: APK and instrumentation installation after user approval; host tests; CPU synthetic 20/20; Vulkan synthetic 10/10 with explicit upload/command/download and FP32 parity; real VAE encoder; strict CPU FP32 DiT streaming reached blocks 1, 2, 4 and 32 with gate tensors. The real-model Vulkan VAE attempt crashed on Adreno 735, so the real pipeline was changed to explicit CPU FP32 rather than silently falling back.

Not completed: decoder, decoded frames, MediaCodec/MediaMuxer, MP4, numeric parity, or second stability run. The long instrumentation remained `mFinished=false` for approximately two hours and produced no MP4; it was force-stopped after external timeout 124. Phase 18C is `PARTIAL_BLOCKED`; full Phase 18 is `NOT READY`.

Recovery for a future device/run:

```powershell
Set-Location D:\AIProjects\seedvr2-ncnn-phase18c
git stash list
git status --short
```

Keep the existing `phase18c-partial-before-device-swap` stash entries and `codex-preserve-phase8-local-before-phase18b`; do not drop, reset, clean, or overwrite them. Rebuild/install, redeploy the current manifest if needed, then rerun `RuntimeDeviceTest#phase18cMinimalVideoDemo` with the device kept awake. A new device needs fresh capability, manifest, real VAE, all-32-block, decoder, frame, encode, MP4, and two-run stability gates; do not inherit Find X8 results.
