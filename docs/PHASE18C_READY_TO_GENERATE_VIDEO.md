# READY_TO_GENERATE_VIDEO gate

`READY_TO_GENERATE_VIDEO` remains false until all prerequisite evidence exists: real-device progress UI, manifest/checkpoint inspection, Block 32 bridge reference and parity, verified decoder latent, decoder single frame, four decoded frames, cancellation/persistence, and MediaCodec preflight. This phase adds the truthful gate plumbing and Decoder-only bridge entry point; it does not claim video readiness or MP4 output.
