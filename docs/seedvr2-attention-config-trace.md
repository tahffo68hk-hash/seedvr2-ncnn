# SeedVR2 attention configuration trace

Static repository search found `configs_3b/main.yaml` and `configs_7b/main.yaml`
set `head_dim: 128` and `qk_norm: fusedrms`. The 3B configuration explicitly
sets `rope_type: mmrope3d` and `rope_dim: 128`.

`models/dit_v2/normalization.py:get_norm_layer` supports Identity, `layer`,
`rms` (Diffusers), `fusedln`, and `fusedrms` (Apex). `NaMMAttention` creates
separate `norm_q` and `norm_k` with `dim=head_dim`. The Phase 7/7.2 small
fixture instead uses LayerNorm, C=32, heads=4 and D=8 as a dependency-free
mathematical migration probe; it is not a weight/configuration match to 3B/7B.

Direct execution of `fusedrms` and `NaMMRotaryEmbedding3d` remains unverified:
the local environment intentionally lacks Apex, Diffusers and
`rotary_embedding_torch`; no package was downloaded.
