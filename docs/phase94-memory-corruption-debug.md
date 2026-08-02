# Phase 9.4.1 Windows memory-corruption triage

## Scope

This note records the investigation of the Windows exceptions first observed
after the Phase 9.4 `ArtifactRegistry::CoreInfo` metadata extension.  The
attention mask, RoPE, RMSNorm, metadata schema, and Dynamic B/C work were not
changed during this triage.

## Original symptom

The pre-existing incremental Release directory
`build/attention-dispatcher-release` produced different failures from the
same source checkout:

| Executable | Result |
| --- | --- |
| `dispatcher_tests.exe` | `0xC0000005` (access violation) |
| `runtime_validation.exe` | `0xC0000374` (heap corruption) |
| `model_cache_tests.exe` | PASS |

The failures occurred after `CoreInfo` gained a typed `std::variant` member
for auxiliary-input metadata.  Such a public C++ object-layout change is ABI
incompatible with translation units compiled against the old header.

## Clean-build result

A new, previously unused Release build directory was configured from the
current source:

```text
cmake -S tests/attention_dispatcher -B build/phase94-memory-release-clean -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/phase94-memory-release-clean --target artifact_registry_tests dispatcher_tests runtime_validation --config Release
```

All three executables passed.  The clean directory was then used for the
following stability and runtime regression checks:

| Check | Result |
| --- | --- |
| `dispatcher_tests.exe` | PASS, 20 consecutive runs |
| `runtime_validation.exe <attention_core>` | PASS, 10 consecutive runs |
| `artifact_registry_tests.exe` | PASS (`cases=34`, 5 baseline and 5 fidelity artifacts) |
| `model_cache_tests.exe` | PASS |
| `dynamic_padding_mask_tests.exe` | PASS |
| `dynamic_rope3d_tests.exe` | PASS |
| `hybrid_window_adapter_tests.exe` | PASS |
| fixed LN hybrid ncnn validation | PASS, BTHWC max abs error `1.49012e-07` |
| fixed RMS hybrid ncnn validation | PASS, BTHWC max abs error `1.78814e-07` |
| Dynamic A RMS hybrid ncnn validation | PASS, BTHWC max abs error `1.49012e-07` |

Each hybrid validation also reported a zero second-run difference and a model
cache size of 5.

## Conclusion and remediation

The clean build does not reproduce either Windows exception.  The supported
root-cause classification is therefore **stale ABI build output / obsolete
object reuse in the former incremental directory**, not a demonstrated
source-level parser or runtime memory defect.

The current CMake target graph already links the consumers through
`artifact_registry_lib`; this investigation found no missing source-level
link dependency to change.  The corrective action is operational: after any
public C++ header layout change (especially `ArtifactRegistry::CoreInfo`),
configure and build in a new directory before interpreting a Windows heap or
access-violation failure as a runtime defect.  Do not reuse objects produced
against the previous header layout.

AddressSanitizer was intentionally not run: the required source-level
reproduction disappeared in a clean Release build, so there is no first
illegal access to attribute to the current source.  If the symptom recurs in
a newly configured directory, create the requested ASan build and report its
first diagnostic before changing implementation code.

## Minimal reproduction boundary

1. Build an executable that includes the old `artifact_registry.h` layout.
2. Change the `CoreInfo` layout and rebuild only a subset of the dependent
   objects/libraries in the same build directory.
3. Link or run the mixed output: this is the environment in which the original
   `0xC0000005` and `0xC0000374` were observed.
4. Repeat from a new CMake build directory: no crash reproduces.

The mixed-object state is not retained as a supported build configuration;
the clean-build procedure above is the reproducible fix.

The Phase 9.4 final validation repeated this rule in
`build/phase94-final-validation-release`; all targets that include the public
Registry and dynamic auxiliary-input headers were rebuilt there. Removing an
individual object file is not a supported substitute for a clean build.
