# Phase 18D single-tile runtime contract

## Request

The request schema is version 1 and contains task ID, source and normalized
hashes, verified model-manifest hash, tile index 0, tile configuration, five
temporal frames, target frame 0, normalization identifier, padding identifier,
and configured threads. Malformed, missing, or extra contract-changing values
are rejected rather than silently defaulted.

The native call validates the direct RGB buffer, recomputes the normalized hash,
builds the complete photo plan, extracts and pads tile 0, and records its input
hash before loading any model.

## Real model stages

The padded RGB8 tile is converted to FP32 with `rgb8 / 127.5 - 1` in C,T,H,W
order and repeated identically for T=5. The runtime executes the existing real
FP32 VAE encoder, maps the latent using the validated scale and layout, executes
blocks 0 through 31 in strict load/execute/unload order with at most one weighted
block resident, runs the existing scheduler endpoint, then all seven VAE decoder
stages. It selects decoder frame 0.

No Phase 18C.4 checkerboard Block32 or decoder checkpoint is a valid input. Any
new checkpoint is written below the current task and carries the complete
identity contract. A checkpoint from another photo, tile, task, or manifest is
rejected.

## Output

Native validates finite decoder values, converts frame 0 to RGB8, crops padding,
and atomically writes `output_valid.rgb`. Its byte count is exactly
`valid_width * valid_height * 3`. Kotlin verifies the count, encodes PNG without
resizing, decodes the PNG for verification, and records its digest.

The report records all three input hashes, bounds, padding modes, per-stage
backend and thread count, Encoder completion, completed DiT count, scheduler and
Decoder completion, output statistics, finite/NaN/Inf counts, and errors.

## PASS boundary

Planner, padding, and identity-fusion tests do not imply model success.
`Phase 18D Real Photo Tile = PASS` requires a real gallery-photo tile, real
Encoder, 32 real blocks, real scheduler and Decoder, a decodable PNG of the valid
size, finite=true, NaN=0, Inf=0, and output bytes different from a direct input
copy. A repository photo fixture is reported as technical validation only and
cannot produce Gallery PASS.
