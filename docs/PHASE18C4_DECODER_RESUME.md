# Phase 18C.4 decoder resume and visible output

The runtime persists and verifies:

- `decoder_after_dec_up0.f32`: `[512,3,32,32]`, 6,291,456 bytes, SHA-256 `7a5a8d...b412`.
- `decoder_after_dec_up1.f32`: `[512,5,64,64]`, 41,943,040 bytes, SHA-256 `e1b015...76ae`.

Each JSON sidecar records float32 CTHW layout, shape, element/byte count, SHA-256, finite/NaN/Inf, min/max/mean/std, backend, thread count, and exact write/reload validation. Both survived App restarts and APK reinstalls. Metadata can be refreshed after validation with zero decoder-stage recomputation.

The successful final route was:

`dec_up1 checkpoint → dec_up2 (24.755 s) → dec_up3 (6.748 s) → dec_out (0.892 s) → RGB → PNG`.

Pad wrote `/storage/emulated/0/Android/data/com.seededge.runtime/files/output/phase18c/frame_000.png` (128×128 RGBA with fully opaque alpha, three RGB color channels, 34,338 bytes). The pulled file and generated visual comparison live under the git-ignored `device_outputs/phase18c4/` directory.

This is a decoder-only checkpoint recovery result derived from the real Block32 checkpoint, scheduler bridge, and VAE decoder. It is not the general user gallery-to-enhanced-photo workflow; Phase 18D image enhancement MVP remains not ready.
