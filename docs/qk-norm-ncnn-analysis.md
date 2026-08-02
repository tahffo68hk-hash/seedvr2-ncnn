# QK Norm → ncnn analysis

The direct call path is `NaMMAttention.forward` / `NaSwinAttention.forward` in
`mmattn.py`: `self.norm_q(vid_q, txt_q)` and `self.norm_k(vid_k, txt_k)`, after
the `[L,3,H,D]` split and before RoPE. The norm factory resides in
`normalization.py`; chosen configurations may be LayerNorm, RMSNorm, or Apex
fused variants.

For the deterministic no-weight fixture, LayerNorm on `[windows*tokens,heads,D]`
is used with independent Q and K affine parameters. pnnx emitted ncnn `LayerNorm`
for `qk_norm.pt`; Release Extractor error was max `4.76837e-07`, mean
`4.95070e-08`.

| Item | Classification | Result |
|---|---|---|
| LayerNorm, FP32 | A native graph | pnnx + ncnn Extractor PASS |
| RMSNorm | E unknown for this build | not tested; depends on actual SeedVR2 config |
| Apex fused norms | B rewrite | use mathematically equivalent native LN/RMS graph, never import Apex into ncnn |
| Separate Q/K parameters | A native graph | two LayerNorm nodes in combined graph |
