# Phase 18D stage-major runtime

The real CPU FP32 full-photo path groups work by model stage: 7 VAE encoder stages, DiT input, 32 weighted DiT blocks, scheduler/head, 7 decoder stages, then fusion. Each stage is loaded once, applied to every pending tile, unloaded, and atomically checkpointed. Peak resident weighted model stages is one.

Checkpoint identity binds task/model/source/normalized SHA-256, tile count, and tile-plan SHA-256. Tensor state records shape, finite status, and SHA-256. Pause occurs only at a committed boundary; resume dispatches the first incomplete item; cancel is terminal.

| Input | Tiles | Mode | Wall | Peak RSS | Loads | Result |
|---|---:|---|---:|---:|---:|---|
| 160x96 | 2 | tile-major | 30.381 s median | not sampled | 96 equivalent | PASS |
| 160x96 | 2 | stage-major | 45.803 s | not validly sampled | 48 | PASS, bit-exact |
| 224x224 | 4 | stage-major | 71.536 s | 1468.93 MiB | 48 | PASS |
| 256x256 | 9 | stage-major | 131.796 s | 1477.39 MiB | 48 | PASS |
| 512x512 | 25 | stage-major | 424.684 s | 1476.32 MiB | 48 | PASS |

Tile-major wins two-tile latency. Stage-major is retained for long resumable tasks. The 512 gate paused after block 10, resumed at `dit_11`, and exactly matched uninterrupted SHA-256 `03ee69e4af40e2949f5e5472a98eb093bd56fbc671f05f299ca4da6a366ce194`.

Xiaomi Pad 7 Pro performance/thermal/background/notification/gallery gates remain **READY_FOR_DEVICE / NOT_RUN**.
