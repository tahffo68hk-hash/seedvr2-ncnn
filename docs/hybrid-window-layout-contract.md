# Hybrid window layout contract

The staged hybrid runtime accepts BTHWC FP32 data.  For the validated fixture the
shape is `[1,4,8,8,32]`; window is `[4,4,4]`; grid is `[1,2,2]`; padding is
zero.  `pack_hybrid_windows` emits windows in this order:

`batch, grid_t, grid_h, grid_w, local_t, local_h, local_w, channel`.

Each window is then stored as ncnn 2-D `Mat(32,64)`, which corresponds to
logical `[64,32]`.  `unpack_hybrid_windows` is the exact inverse and crops the
declared padding.  The Release adapter test compares all 8,192 fixture values
against `reference/windows.bin` exactly (`max_abs_error=0`) and also verifies
the pack/unpack round trip.

This ordering is part of the artifact contract: changing it changes token
identity before attention and cannot be compensated by a different model key.
