# Phase 18D single-tile performance

## Outcome

The fixed **93.3 s** formal BEFORE now measures **30.381 s**
median on the exact preserved 160x96 input/model/two-tile plan: **3.07x**
faster, saving **62.919 s**. Both AFTER runs produced the
same full RGB SHA-256 as BEFORE: `cb8ca128082adc3a655f3d32993f95d5430466c8f362c0066812d6fba90184dd`.

For the requested 128x128 single-tile shape, three Host 5/5/5-thread runs were
15.361 / 15.578 /
15.542 / 15.688 s
(min/median/mean/max). The same preserved Pad input improved from 18.269 s at
5/1/5 threads to 15.357 s at 5/5/5, with bit-identical output.

## Retained changes

1. Stream SHA-256 with a fixed 64 KiB buffer.
2. Persist a verified manifest identity cache binding canonical model root,
   manifest digest, file count, and every declared path digest/size/mtime.
   Any mismatch or malformed cache performs full SHA again.
3. Record model validation, VAE stage load/forward/unload, all 32 DiT blocks,
   scheduler, checkpoint IO, RGB IO, and observed RSS.
4. Support threads per stage. Host sweep best was 8; Android requests the
   existing balanced 5 threads for Encoder/DiT/Decoder.

No allocator, lightmode, packing, precision, sampling, CFG, scheduler, block,
or model algorithm change was retained. No silent fallback was introduced.

Host manifest/tile/full-photo/artifact/real-hook/runtime-progress regressions
passed, `model_cache_tests` passed, Android `:app:testDebugUnitTest` passed,
and `:app:assembleDebug` rebuilt an arm64-v8a-only APK. The generic large
artifact protection check passed. The historical Phase 17-only protection
script is not applicable to this Phase 18 branch because it rejects already
tracked Phase 18 PNG evidence; its attempted report rewrite was restored.

## Device boundary

`adb get-state` returned `no devices/emulators found`, so the rebuilt APK and
Pad AFTER run are separate verification gates. Historical Pad vs current Host
output is tolerance-consistent (RMSE 1.1661, cosine 0.999925), not bit-identical;
the actual Pad before/after must use the rebuilt APK on the same connected device.

## Multi-tile direction

Measured model loading remains a material share of one tile. Analytical 2/4-tile
costs favor stage-major, but no stage-major prototype is claimed. The next gate
is a 2-tile stage-major implementation preserving one resident weighted DiT
block, checkpoint/resume, RSS, and numerical parity.
