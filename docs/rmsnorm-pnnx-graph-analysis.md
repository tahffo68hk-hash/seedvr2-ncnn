# Phase 8 RMSNorm pnnx/ncnn graph analysis

The independent fixture uses the mathematical affine RMSNorm contract:

`y = x * rsqrt(mean(x*x, dim=-1, keepdim=True) + 1e-5) * weight`

The fixture has eight channels, affine weight enabled, and no bias. Four
equivalent TorchScript rewrites were exported and converted with the local
pnnx executable. ncnn numerical behavior differs for the native reciprocal
square-root path.

| Rewrite | TorchScript | pnnx | Release ncnn extract | Max error |
|---|---|---|---|---:|
| A: `rsqrt` | PASS | PASS | FAIL (known numerical limitation) | 4.31776e-04 |
| B: `1/sqrt` | PASS | PASS | PASS | 4.76837e-07 |
| C: `pow(-0.5)` | PASS | PASS | PASS | 4.76837e-07 |
| D: explicit square/mean/add/sqrt/reciprocal | PASS | PASS | PASS | 4.76837e-07 |

The attention artifact uses the explicit primitive form required by the
current ncnn graph conversion. This is not a custom ncnn layer and does not
claim fused Apex parity. The standalone runner selects D and reports the real
A failure while requiring B/C/D to pass.

The remaining blocker is downstream full-head numerical drift: the same
primitive RMSNorm inside Q/K attention currently produces approximately
`3.47e-04` maximum error at the merged-head stage, exceeding the Phase 8
`<1e-4` acceptance threshold.
