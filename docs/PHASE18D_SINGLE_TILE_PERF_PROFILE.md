# Phase 18D single-tile performance profile

The fixed formal BEFORE remains **93.3 s**. Its preserved task is
`host-before-160x96`; inspection shows that historical input planned two tiles.
The unchanged 128x128 legacy core was therefore instrumented separately rather
than silently relabelling the old task as one tile.

## Where the time went

- Full 1366-file / 14.341 GB SHA validation: **56.398 s**.
- Unchanged-model identity-cache validation: **0.206 s**.
- Instrumented legacy 128x128 core: **16.859 s**.
- Encoder: **1.693 s**.
- 32 DiT blocks: **10.509 s**; P50
  **298.001 ms**, P90 **414.025 ms**.
- Decoder: **4.354 s**.
- Model load: **5.775 s**; forward **10.020 s**.
- Checkpoint IO: **7.513 ms**; other IO **4.355 ms**.

The three slowest operations were `dec_up2` (1432.323 ms),
`dec_up1` (1233.664 ms), and `dec_up3`
(976.151 ms). The full top ten and all 32 block records are
in `reports/phase18d_single_tile_profile_before.json`.

## Main conclusion

Repeated full SHA was the dominant avoidable cost. The remaining single-tile
core is compute/load bound, not checkpoint-IO bound. DiT load remains material,
so stage-major is the preferred next multi-tile architecture, subject to a
separate 2-tile RSS/parity prototype.
