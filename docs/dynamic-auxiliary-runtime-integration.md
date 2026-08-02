# Dynamic Auxiliary Runtime Integration

`run_hybrid_attention` routes only from the Registry's typed `AuxiliaryInputConfig`.
It never infers behavior from artifact names or file existence, and dynamic
generation has no reference fallback. Dynamic A (`[1,4,8,8,32]`) regenerates a
zero-padding key mask and fixture-compatible MMRoPE3D each request while reusing
the five existing FP32 ncnn model files through hard links.

Validation report: `reports/dynamic_rms_hybrid_runtime_validation.json`.
Fixed artifacts retain `artifact_reference` routing.

The Dynamic A report records actual validity, mask, RoPE cos/sin, each head,
merge, projection, window, and BTHWC comparisons. Dynamic auxiliary vectors
are recreated for each request; only ncnn model bundles are cached.

## Phase 9.4 final acceptance

`build/phase94-final-reference-validation-release` records strict JSON-number,
reference-negative, precise-error-context, and atomic-scan checks in the dynamic
runtime report. Metadata/runtime routing is READY; Phase 9 multi-shape remains
PARTIAL until Dynamic B/C are implemented.

Phase 9.5 adds Dynamic B spatial padding and Dynamic C temporal-plus-spatial
padding using the same typed `padding_validity` and `dynamic_mmrope3d` route;
see [dynamic-multi-shape-runtime-report.md](D:\AIProjects\seedvr2-ncnn\docs\dynamic-multi-shape-runtime-report.md).

Phase 9.5 is READY for the tested A/B/C fixtures; profile generation beyond
those shapes remains TODO.
