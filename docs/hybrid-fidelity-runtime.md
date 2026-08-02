# Phase 7.4 staged hybrid fidelity runtime

## Status

- Single-graph full fidelity: **BLOCKED**. The preserved vectorized-4D merge, per-head Concat, and multi-branch contribution-Gemm artifacts remain `KNOWN_FAIL`.
- Staged hybrid full fidelity: **READY** for the fixed LayerNorm/RoPE3D fixture `[1,4,8,8,32]`, window `[4,4,4]`, grid `[1,2,2]`.

Artifact: `tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_fidelity_ln_rope444_grid_1x2x2`.

## Execution and result

Four static single-head ncnn nets and one output-projection net load once per runner process. Each head uses 2-D `windows [64,32]` and `mask [64,64]`; host assembles four `[4,64,8]` results then performs head-major merge, exactly equivalent to `torch.cat(..., dim=-1)`. The shared reference `Linear(32,32)` executes in ncnn as a 2-D Gemm over `[256,32]`.

| Stage | Max abs error | Mean abs error | NaN / Inf |
| --- | ---: | ---: | --- |
| head0 | 2.08616e-07 | 2.74903e-08 | 0 / 0 |
| head1 | 1.49012e-07 | 2.58412e-08 | 0 / 0 |
| head2 | 1.63913e-07 | 2.40115e-08 | 0 / 0 |
| head3 | 1.78814e-07 | 2.58722e-08 | 0 / 0 |
| host merge | 2.08616e-07 | 2.58038e-08 | 0 / 0 |
| 2-D Gemm | 1.49012e-07 | 2.12798e-08 | 0 / 0 |
| windows / BTHWC | 1.49012e-07 | 2.10550e-08 | 0 / 0 |

No custom layer is required. Phase 7.5-B now drives this staged path through
`ArtifactRegistry`, keeps the five models in `NcnnModelCache`, validates the
reference mask/RoPE contracts, and records a 3-warmup/10-iteration Release CPU
benchmark in `reports/hybrid_fidelity_runtime_validation.json`. See
`hybrid-runtime-integration-report.md` for the runtime contract.
