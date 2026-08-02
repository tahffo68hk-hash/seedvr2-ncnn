# Full-fidelity ncnn status

`Single graph full fidelity = BLOCKED`: the pre-existing vectorized 4-D merge fails at blob 89, the per-head graph Concat fails at `cat_0`, and the contribution-Gemm single graph has no usable final output. Those artifacts and reports are retained as failure evidence.

`Hybrid staged full fidelity = READY`: the ArtifactRegistry-driven Release
runtime executes four 2-D static head nets, explicit host merge, one ncnn 2-D
output Gemm, and reverse/crop. Its final BTHWC maximum absolute error was
`1.49012e-07`, with mean `2.10550e-08`, zero NaN, and zero Inf values.

The first attempted tuple-head core is also preserved under the hybrid artifact's legacy `attention_heads/model.ncnn.*`; extracting `out0` returned shape `[64,64,8]` (32,768 elements), while its required shape is `[4,64,8]` (2,048 elements). The ready path replaces that invalid rank-3 batched layout with four valid static 2-D head artifacts.

Phase 8 adds a separate mathematical RMSNorm path. Its standalone ncnn
fixture passes for the sqrt/reciprocal, pow, and explicit rewrites, but the
complete RMS hybrid head currently exceeds the `1e-4` gate; it is not promoted
to READY and does not change the LayerNorm result above.

## Phase 9.4 metadata/runtime routing

`Phase 9.4 metadata/runtime routing = READY`: fixed LayerNorm, fixed RMSNorm,
and Dynamic A select their typed metadata routes without cross-profile fallback.
Dynamic A uses natural `valid_tokens=64`; fixed RMS retains the artificial
fidelity override `valid_tokens=60`. Phase 9 multi-shape validation remains
`PARTIAL` because Dynamic B/C are not implemented.

## Phase 9.5 multi-shape status

`Phase 9.5 Dynamic multi-shape RMS Hybrid Runtime = READY`. Dynamic A/B/C use one
metadata-driven route with grids `[1,2,2]`, `[1,3,3]`, `[2,3,3]`; their BTHWC
maximum errors are `1.49012e-7`, `1.49012e-7`, and `2.08616e-7`. This remains a
CPU FP32 fixture subsystem, not complete SeedVR2 inference.

## Phase 10 synthetic head_dim=128 status

`Phase 10 synthetic head_dim=128 = PARTIAL`. A shape-only synthetic fixture
(heads=4, head_dim=128, channels=512, rope_dim=128, window [4,4,4], `pnnx
fp16=0`) validates the C=512 projection, four 128-wide heads, Q/K RMSNorm,
`rope_dim=128` local-window RoPE, dynamic padding-validity mask, HostHeadMerge,
reverse/crop, registry/dispatcher/cache, and single/multi/padded profiles.
RoPE cos/sin max `5.96e-8`; heads/merge/projection/BTHWC max `8.6e-7..1.55e-6`
(all `<1e-4`); validity/mask exact; S->M->P->S bit-identical; negatives 19/19;
cache 15; all 10 Phase 7.5-9.5 regressions PASS. The state is PARTIAL because
the real SeedVR2 MM-RoPE3D axis split is not uniquely resolvable locally
(`rotary-embedding-torch` absent); a documented synthetic axis contract
`[44,42,42]` / `[1,0.01,1e-4]` / theta=10000 is used instead. Real weights,
complete DiT block, Vulkan, and VAE remain TODO. See
[synthetic-hd128-runtime-report.md](synthetic-hd128-runtime-report.md).
