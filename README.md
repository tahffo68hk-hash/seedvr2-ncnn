# SeedVR2 ncnn Adaptive Window Prototype

Metadata-driven Adaptive Window Attention runtime prototype for ncnn.

## Current status

Phase 7.5-B remains READY for five manifest-backed baseline FP32 CPU profiles
and the LayerNorm staged-hybrid fixture. Phase 8 mathematical RMSNorm is
PARTIAL/BLOCKED at the complete ncnn attention-head accuracy gate. This is not
a complete SeedVR2 port.

```mermaid
flowchart LR
  R[AttentionRequest] --> D[Host dispatcher]
  D --> K[CoreKey]
  K --> G[ArtifactRegistry]
  G --> C[NcnnModelCache]
  C --> P[Partition and mask]
  P --> N[Static ncnn core]
  N --> O[Reverse crop BTHWC]
```

Supported profiles: `[1,4,16,16,32]`, `[1,3,16,16,32]`, `[1,4,17,17,32]`,
`[1,8,32,32,32]`, and `[1,5,19,21,32]`.

## Build and validate

Configure a Release Ninja build of `tests/attention_dispatcher`, then build
`model_cache_tests` and `runtime_validation`. Run:

```text
runtime_validation tests/attention_dispatcher/artifacts/attention_core
```

It writes [runtime_validation.json](D:\AIProjects\seedvr2-ncnn\reports\runtime_validation.json) and
[runtime-validation-report.md](D:\AIProjects\seedvr2-ncnn\docs\runtime-validation-report.md).

## Limits

Only the registered profiles are supported; arbitrary resolutions require a
matching static core. The hybrid fidelity fixture supplies artifact-owned mask
and RoPE data and is documented in
[hybrid-runtime-integration-report.md](D:\AIProjects\seedvr2-ncnn\docs\hybrid-runtime-integration-report.md).
Host code owns dynamic control flow, while ncnn executes the attention math.
Vulkan, SeedVR2 weight loading, generalized runtime validation, and direct
Apex `fusedrms` dependency parity remain out of scope or blocked as documented
in [rms-hybrid-runtime-report.md](D:\AIProjects\seedvr2-ncnn\docs\rms-hybrid-runtime-report.md).
