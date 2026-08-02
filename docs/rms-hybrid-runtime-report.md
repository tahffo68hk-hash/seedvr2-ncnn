# RMS hybrid runtime report

The new artifact `hybrid_fidelity_rms_rope444_grid_1x2x2` is strict-schema
scannable and resolves as variant `fidelity_rms_rope3d_hybrid`. It contains
four independently converted head models, one 2-D output projection, and the
same artifact-owned mask/RoPE reference contract as the LayerNorm fixture.

`artifact_registry_tests` passes 34 cases with five baseline and four fidelity
artifacts. The variant lookup does not fall back to LayerNorm.

The Registry-driven runtime loads five models and completes window packing,
four head extracts, host merge, projection, reverse/crop, and cache reuse. The
current RMS runtime is **PARTIAL/BLOCKED** because the final head errors exceed
`1e-4`:

| stage | max abs error |
| --- | ---: |
| head0 | 2.95371e-4 |
| head1 | 3.47316e-4 |
| head2 | 3.22342e-4 |
| head3 | 3.12507e-4 |
| merged | 3.47316e-4 |

Outputs remain finite and repeated cached runs are deterministic. LayerNorm
hybrid and baseline A-E validation remain unchanged and are not replaced.
