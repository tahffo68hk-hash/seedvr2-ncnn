# Next task prompt

Phase 10.1 confirmed the real SeedVR2 MM-RoPE3D contract (`rotary-embedding-torch==0.5.3`, effective rotary dim 126 of configured 128, axis_dims=[42,42,42], identity tail 2) and Phase 10 is READY. Do not re-run Phase 10.1.

Task: integrate a single real SeedVR2 DiT block interface (read `D:\AIProjects\third_party\SeedVR2` read-only, no weights, no download): QKV projection shapes, head count (20) and head_dim (128) from `configs_3b/main.yaml`, the video-window T-axis text-length offset l (`vid_freqs[l:l+f, :h, :w]` in `models/dit_v2/rope.py` `NaMMRotaryEmbedding3d.get_freqs`), text-token RoPE (`txt_freqs[:l].repeat(1, 3)`), padding-validity semantics, and the output projection/merge contract. Extend the existing real-contract fixture with l>0 and text tokens, then validate on CPU FP32 with the documented thresholds. Do not start implementation before the contract is written down with file/symbol references.
