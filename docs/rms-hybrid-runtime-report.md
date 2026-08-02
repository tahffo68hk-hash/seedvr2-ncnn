# RMS hybrid runtime report

The new artifact `hybrid_fidelity_rms_rope444_grid_1x2x2` is strict-schema
scannable and resolves as variant `fidelity_rms_rope3d_hybrid`. It contains
four independently converted head models, one 2-D output projection, and the
same artifact-owned mask/RoPE reference contract as the LayerNorm fixture.

`artifact_registry_tests` passes 34 cases with five baseline and four fidelity
artifacts. The variant lookup does not fall back to LayerNorm.

The Registry-driven runtime loads five models and completes window packing,
four head extracts, host merge, projection, reverse/crop, and cache reuse. The
artifact is now **PASS** after every ncnn model was regenerated with pnnx
`fp16=0`; the earlier `fp16=1` default quantized the QKV/projection weights.

| stage | max abs error |
| --- | ---: |
| head0 | 1.93715e-7 |
| head1 | 1.49012e-7 |
| head2 | 1.78814e-7 |
| head3 | 1.49012e-7 |
| merged | 1.93715e-7 |

Final BTHWC is `1.78814e-7`, with zero NaN/Inf and zero repeat-run difference.
LayerNorm hybrid and baseline A-E validation remain unchanged and are not
replaced.
