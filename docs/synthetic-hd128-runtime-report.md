# Phase 10: synthetic head_dim=128 / rope_dim=128 RMS Hybrid Attention

## Status

`Phase 10 (historical synthetic record) = PARTIAL` (synthetic contract is self-consistent and fully validated on CPU FP32; the real SeedVR2 MM-RoPE3D axis split was not uniquely resolvable from local sources at the time, so no real-SeedVR2 RoPE contract claim was made).

Historical regression record for the Phase 10 synthetic contract. Phase 10.1
subsequently confirmed the real SeedVR2 MM-RoPE3D contract
(`rotary-embedding-torch==0.5.3`, effective rotary dim 126 of configured 128,
axis_dims=[42,42,42], identity tail 2) and Phase 10 is READY; this synthetic
fixture is kept unchanged as an independent regression. See
[real-seedvr2-mmrope-contract-report.md](real-seedvr2-mmrope-contract-report.md).
The original status line ("PARTIAL ... not uniquely resolvable from local
sources") is retained below as the historical record.

This is a shape-only synthetic fixture. It is not real SeedVR2 weights, not a complete SeedVR2 DiT block, and not a mobile-performance conclusion.

## Configuration

```text
heads         = 4
head_dim      = 128
channels      = 512
rope_dim      = 128
window        = [4,4,4]
tokens/window = 64
dtype         = fp32
pnnx fp16     = 0
```

Variant: `synthetic_hd128_dynamic_rms_rope3d_hybrid`.

## RoPE contract

- Local SeedVR2 sources read: `configs_3b/main.yaml` (`heads:20`, `head_dim:128`, `rope_type:mmrope3d`, `rope_dim:128`), `models/dit_v2/rope.py` (`NaMMRotaryEmbedding3d(dim)` -> `MMRotaryEmbeddingBase(dim, rope_dim=3)` -> `RotaryEmbedding(dim=dim//3=42, freqs_for="lang", theta=10000)`), `models/dit_v2/nablocks/attention/mmattn.py` (`NaSwinAttention` passes `window_shape`).
- The concrete per-axis pair split depends on `rotary-embedding-torch`, which is not available locally (`.conda-cuda` and pip cache both lack it). The T/H/W pair partition therefore cannot be uniquely determined.
- Decision: use a documented synthetic axis contract that satisfies the task rules: `axis_dims=[44,42,42]` (all even, sum = 128 = rope_dim), `axis_scales=[1, 0.01, 1e-4]`, `theta=10000`. Python reference and C++ `generate_dynamic_rope3d` implement the identical float32 contract.

## Artifacts

```text
tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_synthetic_hd128_rope444_grid_1x1x1  (HD128-S, [1,4,4,4,512])
tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_synthetic_hd128_rope444_grid_1x2x2  (HD128-M, [1,4,8,8,512])
tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_synthetic_hd128_rope444_grid_2x3x3  (HD128-P, [1,5,9,10,512])
```

Exported by `export_dynamic_rms_profile.py --hd128`; converted by `convert_attention_heads_rms.py --hd128` with `pnnx fp16=0`. Grids 1x2x2/2x3x3 reuse the 1x1x1 ncnn files via hard links; each artifact keeps its own reference/JSON metadata.

## Results

RoPE cos/sin max error: `5.96e-8` (gate `<1e-6`). Validity/mask parity exact (0 error).

| Stage | HD128-S | HD128-M | HD128-P |
|---|---:|---:|---:|
| head0 max | 5.36e-7 | 7.07e-7 | 1.31e-6 |
| head1 max | 4.62e-7 | 1.07e-6 | 1.43e-6 |
| head2 max | 6.24e-7 | 8.94e-7 | 9.54e-7 |
| head3 max | 5.96e-7 | 6.18e-7 | 1.55e-6 |
| merged max | 6.24e-7 | 1.07e-6 | 1.55e-6 |
| projection max | 3.58e-7 | 4.17e-7 | 1.01e-6 |
| windows max | 3.43e-7 | 3.87e-7 | 9.54e-7 |
| bthwc max | 3.43e-7 | 3.87e-7 | 8.64e-7 |

All gates `<1e-4`; NaN=0, Inf=0. HD128-P validity: valid=450, invalid=702, min=2, max=64, total window tokens=1152, masked key entries=44928 (matches expected padding profile).

## Determinism

S -> M -> P -> S run order is bit-identical (last S equals first S). Second S run reports all cache hits. Cache grows 0->5->10->15; `cache_size_after=15`.

## Negative tests

19/19 pass, covering: channels/heads/head_dim/rope_dim contract violations, rope_dim vs metadata mismatch, odd axis_dims, invalid axis_scales, grid-vs-shape mismatch, wrong padding metadata, unknown HD128 grid, phase 9.5 variant requested for HD128 shape, HD128 variant requested with C=32 input, input element count +/-1, reverse/crop out of bounds, mask/rope Mat shape errors, and cache non-pollution. Every failure carries stage context ("axis"/"cache"/"validity"/"rope"/"partition").

## Memory

```text
ncnn model files         = 13,681,985 B (~13.68 MB)
explicit runtime buffers = 14,110,720 B (~14.11 MB)
process working set      = 147,013,632 B (~147 MB)
process peak working set = 307,126,272 B (~307 MB)
```

## Latency

CPU FP32, ncnn threads=1, 3 warmups + 10 cached iterations per profile.

```text
HD128-S total mean 3.845 ms (min 3.399, max 4.798)  heads 2.668 ms
HD128-M total mean 12.775 ms (min 11.526, max 14.219) heads 9.171 ms
HD128-P total mean 49.922 ms (min 46.546, max 53.362) heads 39.515 ms
```

Heads dominate; projection ~3.7 ms worst case. These are fixture timings, not mobile performance conclusions.

## Regressions

All 10 base regressions PASS on the Phase 10 build: `artifact_registry_tests`, `dispatcher_tests`, `model_cache_tests`, `runtime_validation` (baseline A-E), `dynamic_padding_mask_tests`, `dynamic_rope3d_tests` (incl. HD128 golden parity), `ncnn_mat_contract_tests` (incl. `Mat(w=64,h=64)`), `hybrid_window_adapter_tests`, `hybrid_runtime_validation` (fixed LN/RMS), `dynamic_multi_shape_validation` (Dynamic A 1.49e-7, B 1.49e-7, C 2.086e-7). Stability repeats: `synthetic_hd128_validation` x5, `dynamic_multi_shape_validation` x3, `runtime_validation` x3, `dispatcher_tests` x5, all PASS.

## JSON report

`reports/synthetic_hd128_runtime_validation.json`: parses with Python stdlib, zero NaN/Inf leaves, exactly 3 HD128 cases, all accuracy/latency/memory numbers are numbers (not strings), `real_seedvr2_weights=false`, `overall_status=PASS`.

## Limits and next step

Real SeedVR2 weights, complete DiT block, Vulkan, and VAE/full pipeline remain TODO. Next: extract the real single DiT block interface/shape contract from local SeedVR2 sources (including the true MM-RoPE3D axis split once `rotary-embedding-torch` is available) before claiming real-SeedVR2 RoPE parity.
