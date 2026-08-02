# Next task prompt

Phase 10 real HD128 MM-RoPE3D = READY. Phase 11 real single DiT block interface/shape contract = PARTIAL and committed: `NaMMSRTransformerBlock` (SeedVR2 `e4de8c2`), call chain, token layout (T-major W-fastest; per-window `[vid_win; txt]` with txt repeat/mean), `vid_freqs[l:l+f, :h, :w]` offset (`l` = text length, `f` = post-patch frame count), AdaSingle modulation, NaSwinAttention, SwiGLU MLP, residual order - see `docs/seedvr2-single-dit-block-contract-report.md` and `tests/seedvr2_dit_block_contract/`. Do not re-run Phase 11 extraction.

Task: make Phase 11 READY by running the real random-weight forward probe (`tests/seedvr2_dit_block_contract/probe_real_block.py`) in an environment that provides the pinned non-model deps (`einops==0.7.0`, `diffusers==0.33.1` or 0.29.1, `flash-attn==2.5.9.post1`, `apex==0.1`, `rotary-embedding-torch==0.5.3`; CUDA/GPU expected for flash-attn). Do not install them into the repo `.conda-cuda` without approval; do not substitute a hand-written implementation. Probe profiles: A `(1,4,4)`, B `(5,32,32)`, C asymmetric `(7,45,64)` with l=2/8/11; CPU FP32 eval, no_grad, fixed seed, stage hooks, negative tests, identity-tail check, determinism. If the probe passes, mark Phase 11 READY.

Then Phase 12 (do not start before Phase 11 READY): synthetic single DiT block PyTorch -> ncnn staged fidelity using the confirmed contract.
