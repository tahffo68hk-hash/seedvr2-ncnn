# Block 32 decoder bridge

The bridge contract is now explicit and shared by the full pipeline and Decoder-only entry point:

`gate_dit_block_32_vid.f32` (shape `[128,2560]`) and the matching text gate are checkpoint inputs, not VAE latents. The bridge applies the Phase 17 head path: per-token RMS normalization with `norm_weight`, Ada modulation from `emb` using `out_shift`/`out_scale`, `dit/vid_out` projection to 64 patch channels, THWC unpatchify, Euler endpoint against the stored noise tensor, then division by the VAE scaling factor `0.9152` and conversion to VAE decoder layout `[16,2,16,16]`.

The Android implementation is CPU FP32 and reports `decoder_bridge_verified` only after all finite checks and the decoder latent have been produced. Checkpoint presence alone never enables Decoder-only or `READY_TO_GENERATE_VIDEO`. Host/Pad numerical parity remains unverified until a device run produces the reference tensors.
