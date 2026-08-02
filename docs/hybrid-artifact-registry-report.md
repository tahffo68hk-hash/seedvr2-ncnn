# Phase 7.5-A Hybrid ArtifactRegistry

`CoreInfo` now carries a type-safe payload:

```cpp
std::variant<SingleModelArtifact, StagedHeadMergeArtifact>
```

`single_model()` and `staged_head_merge()` are the public discriminator accessors. Baseline code reads only `single_model()->model.param_path/bin_path`; it cannot accidentally select one hybrid head as a complete attention model.

The strict in-tree JSON parser represents lowercase `true`/`false` as a real boolean. Baseline manifests retain the legacy defaults (`variant=baseline`, `SingleModel`). A staged manifest must instead declare `execution=staged_host_head_merge` and a non-empty, non-`baseline` variant. It validates four uniquely named heads (`head0` through `head3`), output projection, feature types, positive token product, and safe existing relative paths.

`scan(vector<path>)` collects every core into a temporary candidate list, checks full keys (window, grid, variant) for duplicates, sorts by name, and commits only on success. Thus a bad later root leaves the previous registry intact.

`NcnnModelCache` explicitly rejects `StagedHeadMergeArtifact`, reporting core name, variant, and execution type; failed param/bin loads are not cached. The baseline runtime forwards `AttentionRequest::variant` into its lookup key and rejects a non-single artifact before model cache or inference. The compatibility runner has likewise migrated away from removed top-level model-path fields.

Validation used clean Phase 7.5-A build directories. Debug runs Registry and Dispatcher contract tests only. Release uses the installed Release `ncnn.lib` and built all Registry, Dispatcher, cache, runner, and runtime-validation targets. Release executions passed:

- `artifact_registry_tests=PASS cases=31` (`baseline_count=5`, `fidelity_count=3`); the target hybrid key resolves to staged execution with four heads and one projection.
- `dispatcher_tests=PASS`.
- `model_cache_tests=PASS`, including reuse, missing param/bin, staged rejection, unchanged cache, and `clear()`.
- `runtime_validation=PASS cases=5`, cache size 5. Cases A--E selected the expected baseline cores; maximum absolute errors were `1.49012e-07`--`1.78814e-07`, mean errors `1.57646e-08`--`1.69706e-08`, all below `1e-4` with no reported NaN/Inf.

The global migration search finds no direct `CoreInfo::param_path` or `CoreInfo::bin_path` access. Remaining `param_path/bin_path` occurrences are correctly nested payload accesses (and controlled test-only payload fault injection).

Phase 7.5-B adds `HybridModelBundle`, staged cache acquisition, and the
ArtifactRegistry-driven hybrid runtime. The cache owns five Release ncnn nets
(four heads plus projection); a baseline payload is rejected by the hybrid
executor before loading. RMSNorm, Vulkan, and generalized production shapes
remain out of scope.
