# Phase 7.3 head-merge ncnn analysis

The Phase 7.2 vectorized model is retained as `vectorized_4d_merge = KNOWN_FAIL`.
Its blob 89 is produced by `reshape_26`, consuming post-AV `permute_8` blob 88;
the AV producer is `matmul_35` blob 87. It attempts `[W,H,N,D] -> [W,N,C]` via
4-D Permute then Reshape, and Release ncnn returns `-1` at blob 89.

Phase 7.3 adds `PerHeadBranchRewrite`: Q/K/V are statically sliced into four
`[W,N,D]` branches; every branch performs LayerNorm, host RoPE, QK MatMul, mask
Add, Softmax and AV. Python parity is below `1e-6`, including each head AV and
explicit head0/head1/head2/head3 channel order.

The first per-head ncnn graph used `cat_0` to concatenate four AV tensors; that
Concat returned `-1`. A second mathematically equivalent graph decomposes
`concat(heads) -> output Linear` into four fixed input-column Gemm contributions
plus the original output bias. It removes post-AV 4-D Permute and Concat, but
the current Release ncnn still returns `-1` for final `out0`. No custom layer was
added; this is retained as an unresolved ncnn multi-branch layout issue.

| Implementation | AV representation | Merge/output form | ncnn |
|---|---|---|---|
| Vectorized | `[W,H,N,D]` | 4-D Permute/Reshape | FAIL at blob 89 |
| Per-head concat | four `[W,N,D]` | four-input Concat | FAIL at `cat_0` |
| Per-head decomposed projection | four `[W,N,D]` | 4 Gemm contributions + Add | final `out0` FAIL |
