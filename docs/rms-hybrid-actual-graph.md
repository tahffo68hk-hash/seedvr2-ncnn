# Phase 8.1 RMS hybrid actual graph

The checked-in production RMS artifact is
`hybrid_fidelity_rms_rope444_grid_1x2x2`. Its four ncnn head params are
generated with pnnx `fp16=0` and use the D rewrite:

```text
square (BinaryOp mul)
-> Reduction mean over the final head_dim=8 axis
-> add epsilon 1e-5
-> UnaryOp sqrt
-> UnaryOp reciprocal
-> multiply Q/K
```

There is no `rsqrt` operator. The fixture's Q/K affine weights are independent
parameters, no bias is present, and their current deterministic first/last
values are both `1.0`; pnnx therefore folds identity weight multiplication
from the ncnn graph without changing the mathematical result. Input rows are
`[64,8]`, mean-square is `[64,1]`, and reduction never crosses token/window
boundaries.

Full head0 comparison candidates, all converted with `fp16=0`:

| rewrite | final ncnn RMS operators | max abs error |
|---|---|---:|
| B | `sqrt` + `reciprocal` | 1.34110e-7 |
| C | `BinaryOp pow(-0.5)` | 1.49012e-7 |
| D | `sqrt` + `reciprocal` | 1.34110e-7 |

B and D lower to the same stable ncnn RMS subgraph; C remains `pow(-0.5)`.
None is fused to the known-failing `rsqrt` graph. D is the artifact's declared
rewrite because it is explicit at source level.
