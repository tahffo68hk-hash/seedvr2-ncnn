# ncnn Attention Debug

## Test cases

All tests use CPU float32 tensors with logical shape `[B=1, heads=4, tokens=32, dim=8]`. The fixed batch is represented by the test case; the ncnn input is `[heads,tokens,dim]` as `Mat(w=8,h=32,c=4)`.

| Case | PyTorch operation | ncnn output shape |
|---|---|---|
| A: MatMul-only | `QK^T / sqrt(8)` | `[4,32,32]` |
| B: MatMul+Softmax | A then softmax over final axis | `[4,32,32]` |
| C: complete attention | B then `attention @ V` (`V=Q`) | `[4,32,8]` |

## Results

| Case | pnnx param/bin | ncnn load | ncnn extract | Max abs error |
|---|---|---|---|---:|
| A | PASS | PASS | PASS | 0.0 |
| B | PASS | PASS | PASS | 1.1920929e-07 |
| C | PASS | PASS | PASS | 4.7683716e-07 |

All three models returned `extract ret=0`. The former fixture failure (`-1073741819`) is therefore not caused by ncnn `MatMul`, `Softmax`, or the basic attention `AV` multiplication.

## Failed operator

No isolated attention operator failed.

The failed pnnx fixture graphs have additional structure not present here: `Split` fan-out, `InnerProduct` Q/K/V projections, repeated reshape/permute paths, and concatenation. The two crashing variants have different head handling but both contain projected multi-branch QKV graphs. This narrows the next investigation to pnnx lowering/runtime interaction around the QKV projection/branch/shape graph, not to attention math itself.

## Possible cause

The current evidence supports a graph-composition issue rather than an ncnn attention primitive failure. The full BTHWC graph also has independently reported unsupported 8-rank transforms. The core crash must be further isolated with a `Linear-only`, then `Linear+Split`, then `Linear+reshape/permute` sequence before naming a single failed ncnn layer.

## Custom layer impact

Basic fixed-shape scaled dot-product attention does **not** need a custom ncnn layer: pnnx generated and ncnn executed it with small float32 error.

For real SeedVR2, a custom layer remains justified by dynamic window slicing/reverse indices, multimodal text repetition, 3D RoPE, Q/K norms, and variable-length attention. It should not be introduced merely to replace ordinary fixed-shape MatMul/Softmax/AV attention.
