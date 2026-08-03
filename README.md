# SeedVR2 ncnn Adaptive Window Prototype

Metadata-driven Adaptive Window Attention runtime prototype for ncnn.

## Current status

Phase 7.5-B remains READY for five manifest-backed baseline FP32 CPU profiles
and the LayerNorm staged-hybrid fixture. Phase 8 mathematical RMSNorm is READY
for this fixture's complete ncnn attention-head accuracy gate. Phase 9.5
dynamic multi-shape RMS hybrid runtime is READY. Phase 10 head_dim=128 is READY:
the exact SeedVR2 MM-RoPE3D contract (rotary-embedding-torch==0.5.3, effective
rotary dim 126 of configured 128, 2-dim identity tail, axis_dims=[42,42,42])
was pinned, oracle-verified, and validated end-to-end (real-contract S/M/P),
with the Phase 10 synthetic fixture kept as an independent historical
regression. Phase 11 real single DiT block interface/shape/execution contract is
READY (superseded by Phase 12 staged ncnn fidelity): `NaMMSRTransformerBlock` (SeedVR2 `e4de8c2`) was executed with the real
random-weight forward in the isolated conda env `seedvr2-block-probe` (torch
2.4.0+cpu, einops 0.7.0, diffusers 0.33.1, rotary-embedding-torch 0.5.3) at full
2560 width (profiles A/B/C/D) with runtime-hook evidence for `[vid_win; txt]`
packing, `vid_freqs[l:l+f]` offset, identity tail 2, six-segment modulation,
SwiGLU, residual order, last-layer vid_only, window reassembly, negatives 17/17;
only flash_attn/apex fused-CUDA libs are replaced by test-harness CPU compat
(parity < 1e-6; CUDA fidelity not validated). See
[seedvr2-single-dit-block-contract-report.md](D:/AIProjects/seedvr2-ncnn/docs/seedvr2-single-dit-block-contract-report.md).
Phase 12 synthetic single DiT block staged ncnn fidelity is READY: the real
`NaMMSRTransformerBlock` topology runs at scaled C=512/H=4/HD=128 through a
staged ncnn + host hybrid runtime (`HybridDiTBlockRuntime`, seed 20260802, MLP
hidden 1536 from the real SwiGLUMLP formula); profiles A/B/C/D all PASS with
final vid/txt < 1e-4, A->B->C->D->A bit-identical, last-layer vid_only PASS,
registry/cache 19, negatives 24+7+registry, and Phase 9.5/10/11 regressions
PASS (batch>1 txt indexing quirk of the real `repeat_concat_idx` reproduced
exactly). See [synthetic-seedvr2-dit-block-ncnn-report.md](D:/AIProjects/seedvr2-ncnn/docs/synthetic-seedvr2-dit-block-ncnn-report.md).
Phase 13 real-width single-block feasibility + weight mapping is READY: the real
`NaMMSRTransformerBlock` at C=2560/H=20/HD=128 (MLP 6912, seed 20260803, random
weights) is exported to 19 FP32 ncnn stage models (`pnnx fp16=0`) and passes the
minimal RW-A profile end-to-end on the existing `HybridDiTBlockRuntime`
(final vid/txt ~1e-5, bit-identical determinism, cache 19, peak working set
~685 MB, cached forward ~31 ms), with a full committed state_dict->ncnn weight
mapping contract (29/29, AdaSingle runtime-only, qk-norm all-ones folded,
last-layer reuses all 19 models). See
[realwidth-seedvr2-block-feasibility-report.md](D:/AIProjects/seedvr2-ncnn/docs/realwidth-seedvr2-block-feasibility-report.md)
and [seedvr2-block-weight-mapping-report.md](D:/AIProjects/seedvr2-ncnn/docs/seedvr2-block-weight-mapping-report.md).
Phase 14 real-weight single-block hybrid visual milestone is **READY**: the
official `ByteDance-Seed/SeedVR2-3B` checkpoint (5/5 sha256 verified, official
source only) is safely loaded (`weights_only=True`), schema-validated (32 blocks,
block 16 = floor(32/2) middle, 635 fp32 keys), q/k RMSNorm forced real-affine
(not all-ones; Phase 13 0-byte reuse forbidden), mapped via the 15-key
state_dict->ncnn contract (mapped_numel == source_numel, QKV/SwiGLU bit-exact),
exported to 19 FP32 stages (`pnnx fp16=0`, git-ignored `.phase14_local_artifacts/`),
and the ncnn `HybridDiTBlockRuntime` with real weights passes the RW-A O(1)
fixture (final vid/txt < 2e-4) and the official-context block-16 parity
(rel < 1e-3), plus a hybrid visual milestone: 4-frame 128x128 official pipeline
with one block replaced by ncnn, pixel PSNR 52.5 dB / SSIM 0.9975, adapter
evidence (ncnn=1, original=0, no silent fallback), negatives 29/29, regressions
22/22. See [real-weight-single-block-report.md](D:/AIProjects/seedvr2-ncnn/docs/real-weight-single-block-report.md)
and [hybrid-visual-milestone-report.md](D:/AIProjects/seedvr2-ncnn/docs/hybrid-visual-milestone-report.md).
This is a SeedVR2-style attention subsystem with a single ncnn block, not a
complete SeedVR2 port.

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
`[1,8,32,32,32]`, `[1,5,19,21,32]`, Dynamic A/B/C `[1,4,8,8,32]` /
`[1,4,9,10,32]` / `[1,5,9,10,32]`, and synthetic/real-contract HD128 S/M/P `[1,4,4,4,512]` /
`[1,4,8,8,512]` / `[1,5,9,10,512]`.

## Build and validate

Configure a Release Ninja build of `tests/attention_dispatcher`, then build
`model_cache_tests` and `runtime_validation`. Run:

```text
runtime_validation tests/attention_dispatcher/artifacts/attention_core
```

It writes [runtime_validation.json](D:\AIProjects\seedvr2-ncnn\reports\runtime_validation.json) and
[runtime-validation-report.md](D:\AIProjects\seedvr2-ncnn\docs\runtime-validation-report.md).

Phase 10 synthetic HD128 validation:

```text
synthetic_hd128_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

It writes [synthetic_hd128_runtime_validation.json](D:\AIProjects\seedvr2-ncnn\reports\synthetic_hd128_runtime_validation.json) and is documented in
[synthetic-hd128-runtime-report.md](D:\AIProjects\seedvr2-ncnn\docs\synthetic-hd128-runtime-report.md).

Phase 10.1 real SeedVR2 MM-RoPE3D contract validation (new clean build `build/phase10_1-real-mmrope-release`):

```text
real_mmrope_contract_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

It writes [real_seedvr2_mmrope_contract_validation.json](D:\AIProjects\seedvr2-ncnn\reports\real_seedvr2_mmrope_contract_validation.json) and is documented in
[real-seedvr2-mmrope-contract-report.md](D:\AIProjects\seedvr2-ncnn\docs\real-seedvr2-mmrope-contract-report.md).

Phase 11 real single DiT block contract: extraction (source/AST, no ncnn build
needed) and Phase 11.1 real random-weight forward execution:

```text
D:\AIProjects\seedvr2-ncnn\.conda-cuda\python.exe tests\seedvr2_dit_block_contract\extract_seedvr2_block_contract.py
conda run -n seedvr2-block-probe python tests\seedvr2_dit_block_contract\probe_real_block.py
conda run -n seedvr2-block-probe python run_phase11_1_probe.py
```

`probe_real_block.py` runs the three suites in-process and writes
`tests/seedvr2_dit_block_contract/probe_result.json` (PASS). The runner
`run_phase11_1_probe.py` executes the three test files as subprocesses and
writes [seedvr2_real_block_forward.json](D:\AIProjects\seedvr2-ncnn\reports\seedvr2_real_block_forward.json).
Extraction writes [seedvr2_single_dit_block_contract.json](D:\AIProjects\seedvr2-ncnn\reports\seedvr2_single_dit_block_contract.json)
and `tests/seedvr2_dit_block_contract/seedvr2_block_contract.json`. Documented
in [seedvr2-single-dit-block-contract-report.md](D:\AIProjects\seedvr2-ncnn\docs\seedvr2-single-dit-block-contract-report.md).

## Limits

Only the registered profiles are supported; arbitrary resolutions require a
matching static core. The hybrid fidelity fixture supplies artifact-owned mask
and RoPE data and is documented in
[hybrid-runtime-integration-report.md](D:\AIProjects\seedvr2-ncnn\docs\hybrid-runtime-integration-report.md).
Host code owns dynamic control flow, while ncnn executes the attention math.
Vulkan, SeedVR2 weight loading, generalized runtime validation, and direct
Apex `fusedrms` dependency parity remain out of scope or blocked as documented
in [rms-hybrid-runtime-report.md](D:\AIProjects\seedvr2-ncnn\docs\rms-hybrid-runtime-report.md).

## Phase 9.4 status

`Phase 9.4 metadata/runtime routing = READY`. Fixed LayerNorm and RMSNorm use
artifact-owned mask/RoPE inputs; Dynamic A uses generated padding-validity mask
and dynamic MM-RoPE3D with natural `valid_tokens=64`. Any public C++ type-layout
change requires a new clean build directory. `Phase 9 multi-shape validation =
PARTIAL` until Dynamic B/C are implemented.

## Current result

This is a metadata-driven, SeedVR2-style adaptive-window attention subsystem for ncnn, not complete SeedVR2 inference. It includes baseline single-model execution; staged four-head Hybrid execution; Q/K LayerNorm and mathematical RMSNorm; local-window 3D RoPE; dynamic padding-validity masks; host head merge; ncnn projection; process-level model cache; variant-aware artifacts; Dynamic A/B/C FP32 CPU parity around `1e-7`; and a head_dim=128 fixture (S/M/P) under both the documented synthetic contract and the real SeedVR2 MM-RoPE3D contract (rotary-embedding-torch 0.5.3) with sub-`1e-6` RoPE parity, sub-`1e-4` stage parity, and exact validity/mask agreement.

All FP32 parity/reference artifacts must use `pnnx fp16=0`. After public C++ type-layout changes (for example `CoreInfo` or `std::variant` payloads), use a new clean build directory and rebuild all dependent targets.

Phase 7.5-B LayerNorm Hybrid Runtime = READY; Phase 8 RMSNorm Hybrid Runtime = READY; Phase 8.1 FP32 conversion precision fix = READY; Phase 9.1 DynamicAttentionContract = READY; Phase 9.2 Dynamic padding mask = READY; Phase 9.3 Dynamic MM-RoPE3D = READY; Phase 9.4 Typed auxiliary metadata/runtime routing = READY; Phase 9.5 Dynamic multi-shape RMS Hybrid Runtime = READY; Phase 10 head_dim=128 = READY (real SeedVR2 MM-RoPE3D contract confirmed from rotary-embedding-torch==0.5.3 and validated; synthetic contract kept as historical regression).

Phase 14 = READY (real SeedVR2 3B checkpoint, real-weight single-block hybrid visual milestone; see reports above).

Phase 13 real-width validation (clean build `build/phase13-real-width-release`):

```text
build/phase13-real-width-release/realwidth_dit_block_validation.exe .phase13_local_artifacts reports/realwidth_seedvr2_block_feasibility.json
build/phase13-real-width-release/realwidth_artifact_inventory_tests.exe .phase13_local_artifacts reports/realwidth_artifact_inventory.json
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/aggregate_realwidth_feasibility.py
```

Heavy real-width artifacts stay in the git-ignored `.phase13_local_artifacts/`
directory; only manifests, the mapping contract, scripts, tests and reports are
committed (see `tests/seedvr2_weight_mapping/check_no_large_tracked_artifacts.py`).

TODO (Phase 15, not started - explicit authorization required): full 32-block real-weight DiT streaming on ncnn, Android/Vulkan targets, VAE/full video restoration in ncnn, and direct fused-kernel parity. Phase 14 committed evidence: real-weight single-block RW-A parity (final vid/txt < 2e-4) and official-context block-16 parity (rel < 1e-3), hybrid visual milestone (pixel PSNR 52.5 dB, SSIM 0.9975), negatives 29/29, regressions 22/22, git protection 8/8.
