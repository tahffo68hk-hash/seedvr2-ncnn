# Phase 7.5-B hybrid runtime integration

Status: **READY** for the fixed FP32 CPU fidelity fixture.

`ArtifactRegistry::scan()` collects the five baseline roots plus the fidelity
root.  An `AttentionRequest` derives metadata, builds a full `CoreKey`
(window, grid, variant), and performs `ArtifactRegistry::lookup()`.  The
selected staged payload supplies four head model pairs and one projection model
pair; `NcnnModelCache` loads them once and later calls reuse all five nets.

The validated request is `[1,4,8,8,32]`, `fp32`, variant
`fidelity_ln_rope3d_hybrid`.  It selected
`hybrid_fidelity_ln_rope444_grid_1x2x2` and ran pack, four ncnn head extractors,
host head-major merge, ncnn projection, and unpack/crop.  The final BTHWC error
was max `1.49012e-07`, mean `2.10550e-08`, NaN `0`, Inf `0`.

Mask source is the artifact `reference/mask.bin`, exactly four `[64,64]`
FP32 matrices, with `valid_tokens=60`.  A zero mask remains finite but produces
max BTHWC error `0.0796393`; it is therefore not a fallback.  Incorrect mask
length rejects with `expected_floats=16384 actual_floats=1`.  RoPE cos/sin come
from the artifact reference files, each logical float `[64,4]` and ncnn
`Mat(4,64)`; their exact 256-float bounds are checked before extraction.

An explicit `baseline` request is rejected before any hybrid cache load, and an
unknown hybrid key is rejected without fallback.  The old compile-time registry
is not on this runtime path.  Release is intentional: all targets use the
installed Release `ncnn.lib` and MSVC runtime; Debug/Release CRT mixing is a
build-boundary issue, not dispatcher behaviour.

The validation performs three cached warmups and ten cached CPU FP32,
single-thread iterations, emitting actual stage timing means/min/max in
`reports/hybrid_fidelity_runtime_validation.json`.  First bundle load timing is
reported separately from cached inference.  Remaining work is model-cache
policy refinement and broader multi-shape runtime validation/benchmarking.
