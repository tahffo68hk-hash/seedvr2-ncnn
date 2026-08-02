# Per-head attention pnnx graph

The per-head artifact contains one QKV `Gemm`, static Slice nodes for each Q/K/V
head range, four independent LayerNorm/RoPE/MatMul/Add/Softmax/MatMul paths, and
an output-projection equivalent written as four 2-D Gemm input-column blocks
plus BinaryOp additions. The rewrite has no post-AV 4-D Permute.

The graph is mathematically equivalent to concatenating head outputs in channel
order and applying the original output projection. pnnx conversion succeeds;
Release ncnn loading and all `Extractor::input` calls succeed, but final
`Extractor::extract("out0")` returns `-1` in the current ncnn build.
