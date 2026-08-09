# Phase 18D arbitrary-size full-photo gate

A natural 640×427 PNG completed through the one-command Host CLI and the real CPU FP32 stage-major runtime. The non-square input plans 35 tiles; its final horizontal and vertical anchors are not limited to square, 128-multiple, or stride-multiple dimensions. All 35 tiles completed the seven VAE encoder stages, 32/32 DiT blocks, scheduler, seven decoder stages, valid crop, weighted fusion, and same-size PNG publication.

Measured wall time was 524.986 s, peak RSS was 1,473.22 MiB, peak task disk was 2,972,980,657 bytes, and final task disk was 78,902,392 bytes. Fusion reported zero uncovered pixels, zero zero-weight pixels, valid unique seam bounds, finite output, NaN=0, Inf=0, and no silent fallback. The final RGB SHA-256 is `89d85b72d4786639368e0bfa5de27cc5d7bd5dec267145f06e34a7ef3842943b`.

The executed command used `tools/phase18d_enhance_photo.py` with `--runtime-mode stage_major`; task-local runtime artifacts and natural images remain git-ignored. Exact structured evidence is in `reports/phase18d_arbitrary_size_full_photo_gate.json`.

Host arbitrary-dimension full photo is **PASS** for reasonable positive dimensions within the documented pixel, tile, disk, and memory guards. This does not claim unlimited dimensions. Xiaomi Pad 7 Pro remains **READY_FOR_DEVICE / NOT_RUN**.
