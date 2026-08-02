# Full fidelity pnnx graph

Artifact: `tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity/core_fidelity_ln_rope444_grid_1x2x2`.

`model.ncnn.param` contains flattened projection boundaries as `Gemm`, two
`LayerNorm` nodes, native RoPE slice/elementwise/concat nodes, `MatMul` for QK,
`BinaryOp add` for the mask, `Softmax`, `MatMul` for AV, and the output `Gemm`.
Neither projection lowered to 3-D InnerProduct.

| Stage | Release ncnn result |
|---|---|
| QKV + Q/K Norm + RoPE | PASS |
| QK MatMul | PASS |
| mask Add + Softmax | PASS |
| AV MatMul | PASS |
| head merge Permute/Reshape | FAIL (`extract` blob 89 = -1) |
| output Gemm/final output | blocked by head merge |

All four `Extractor::input` calls return zero. The blocker is therefore the
generated post-AV 4-D head merge, not input loading or masking.
