# Artifact Registry Report

## Read files and artifact root

Read the dispatcher headers/implementation, metadata definitions and tests,
CMake file, static-core generator, dispatcher design/runtime architecture
documents, and `PHASE6_REPORT.md`. The artifact root is
`tests/attention_dispatcher/artifacts/attention_core`; it contains exactly five
core directories and manifests.

## Implementation

`ArtifactRegistry` is declared in `tests/attention_dispatcher/artifact_registry.h`
and implemented in `artifact_registry.cpp`. It exposes `scan`, `lookup`,
`list`, and `size`. A small strict recursive-descent JSON parser is used; it
accepts only strings, integers, arrays, and objects needed by this schema,
rejects malformed JSON and type mismatches, and reports manifest field names.

Validation covers all required fields, positive dimensions, token-product
consistency, non-empty input shape, `fp32`, safe relative param/bin paths, and
existing files. Scanning is one level deep, ignores ordinary files and
directories without `metadata.json`, sorts successful results by name, and
detects duplicate window/grid keys. Results are accumulated in a temporary
vector and committed only after the complete scan succeeds, preserving the
previous registry on failure.

The five existing manifests originally said `float32`, while the requested
strict contract allows only `fp32`; the five manifest values were normalized to
`fp32` so the real artifact directory satisfies the stated contract.

## Tests

`artifact_registry_tests.cpp` exercises real five-core scanning, size and
lookup, unknown keys, name sorting, missing root, malformed JSON, missing
fields, wrong types, token mismatch, missing param/bin, traversal paths,
duplicate keys, failure atomicity, and recovery after fixing a temporary
directory. The test executable was compiled with the existing MSVC/Ninja setup
and ran successfully:

`artifact_registry_tests=PASS cases=17`

Phase 6.3 now consumes this registry through the dispatcher and Release runtime.
See `registry-dispatcher-integration.md` and `runtime-validation-report.md` for
the five-case cached inference and benchmark results.
