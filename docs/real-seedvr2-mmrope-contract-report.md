# Phase 10.1: real SeedVR2 MM-RoPE3D contract RMS Hybrid Attention

## Status

`Phase 10 = READY` (real-contract scope). The exact `rotary-embedding-torch`
contract used by SeedVR2 was pinned, downloaded, source-audited, reproduced as a
tensor oracle, implemented in the runtime, and validated end-to-end on CPU FP32.
The Phase 10 synthetic fixture remains committed as an independent historical
regression and is not modified.

This is still not real SeedVR2 weights, not a complete SeedVR2 DiT block, and
not a mobile-performance conclusion.

## Version evidence (provenance)

- SeedVR2 repo: `D:\AIProjects\third_party\SeedVR2`, HEAD
  `e4de8c24441a67e1b7df56abea10645059bb1185` (remote ByteDance-Seed/SeedVR, clean).
- `requirements.txt:13` -> `rotary-embedding-torch==0.5.3` (exact pin).
- `environment.yml:181` -> `rotary-embedding-torch==0.5.3` (exact pin).
- `configs_3b/main.yaml:19-20,35-36` -> `heads:20`, `head_dim:128`, `rope_type:mmrope3d`, `rope_dim:128`.

## Downloaded artifact (limited, temporary)

```text
rotary_embedding_torch-0.5.3-py3-none-any.whl  SHA256 7c0297a21301fbd0d20cfdcba33be5d1ad0a46e33454168af08413dbf3e07457
einops-0.7.0-py3-none-any.whl                 SHA256 0f3096f26b914f465f6ff3c66f5478f9a5e380bb367ffc6493a68143fbbf1fd1
beartype-0.22.9-py3-none-any.whl              SHA256 d16c9bbc61ea14637596c5f6fbff2ee99cbe3573e46a716401734ef50c3060c2
```

- Source file `rotary_embedding_torch/rotary_embedding_torch.py`
  SHA256 `80a45656e8d6aa8540c96a3fe97faae643da88182e5ae61d0e71c64c9ef61c1a`.
- License: MIT (`dist-info/LICENSE`). Temporary directory `.phase10_1_dependency_tmp/` was removed before commit.

## Audited upstream symbols

`RotaryEmbedding.__init__`, `RotaryEmbedding.forward`, `RotaryEmbedding.get_axial_freqs`,
`apply_rotary_emb`, `rotate_half` in rotary-embedding-torch 0.5.3, plus SeedVR2
`models/dit_v2/rope.py` (`NaMMRotaryEmbedding3d`, `MMRotaryEmbeddingBase`) and
`models/dit_v2/nablocks/attention/mmattn.py` (window rope call).

## Real MM-RoPE3D contract

SeedVR2 builds `RotaryEmbedding(dim=128 // 3 = 42, freqs_for="lang", theta=10000)`.

```text
configured_rope_dim   = 128   (SeedVR2 config value, recorded for documentation)
effective_rotary_dim  = 126   (3 x 42 per-axis blocks, actually rotated)
identity_tail_dim     = 2     (dims 126..127 unchanged, cos=1/sin=0)
head_dim              = 128
axis_order            = T, H, W  (T block first, then H, then W)
per-axis dims         = [42, 42, 42]  (21 pairs each)
freq formula          = 1 / (theta ** (arange(0, dim, 2)[:dim // 2] / dim)), dim=42, theta=10000
per-axis pair k       = theta ** (-2k / 42) = theta ** (-k / 21)
frequency layout      = each frequency duplicated for the interleaved pair (r=2)
partial rotary        = apply_rotary_emb rotates dims [0:126] only (interleaved rotate_half)
identity tail         = t[..., 126:] returned unchanged
token flatten         = T-major, W-fastest: (lt * window_h + lh) * window_w + lw
window coordinates    = per-axis arange(0..L-1); the real DiT pipeline offsets the
                        video T axis by the text length l (vid_freqs[l:l+f, :h, :w]);
                        l=0 in this standalone window fixture
```

Runtime metadata encodes the effective value: `rope_dim=126`, `theta=10000`,
`axis_dims=[42,42,42]`, `axis_scales=[1,1,1]`, `freq_mode="lang_rope"` (new
optional contract field; default `"global_pairs"` preserves the Phase 9.5 and
Phase 10 synthetic contracts byte-for-byte). The configured value 128 is
recorded in the oracle manifests, JSON report, and this document only.

## Upstream oracle

`tests/seedvr2_attention_fidelity/seedvr2_mmrope_oracle.py` imports the exact
0.5.3 package and writes reproducible references under
`tests/seedvr2_attention_fidelity/artifacts/mmrope_oracle/{a,b,c}/`:

```text
Oracle A window [1,1,1]  (all-zero freqs -> identity rotation)
Oracle B window [4,4,4]  (standard window)
Oracle C window [2,3,5]  (asymmetric: proves T/H/W axis order and W-fastest)
```

Each case: `freqs.bin` [L,126], `cos.bin`/`sin.bin` [L,64] (63 active pairs +
1 identity pair), `q_input.bin`/`k_input.bin` [L,4,128], `q_rotated.bin`/
`k_rotated.bin` [L,4,128], and `manifest.json` (package version, archive/source
SHA256, SeedVR2 commit, configured/effective/identity dims, axis order, window,
tensor shapes, dtype, random seed). Frequencies verified: per-axis block at
coordinate 1 begins `[1, 1, 0.6449, 0.6449, 0.4159, 0.4159]`; pair duplicates
exact; W-fastest reshape exact.

## C++ / Python contract alignment

- `Rope3DConfig` and `DynamicMMRoPE3DConfig` gain an optional `freq_mode`
  (`"global_pairs"` default = legacy; `"lang_rope"` = upstream per-axis lang
  formula). No hard-coded fixture tables; the extension is general.
- `generate_dynamic_rope3d` (lang_rope) computes per-axis local pair k:
  `inv_freq = theta^(-k / axis_pairs)`, angle = coordinate * axis_scale * inv_freq,
  identity pairs beyond `rope_dim` stay cos=1/sin=0.
- `host_lang_rope3d_cos_sin` in `rope3d_reference.py` mirrors the C++ exactly
  and matches the upstream oracle cos/sin within float32 rounding.
- Phase 9.5 `rope_dim=6` golden and Phase 10 synthetic `rope_dim=128` golden are
  unchanged and re-verified.

## Parity results (new clean build `build/phase10_1-real-mmrope-release`)

Oracle parity (`dynamic_rope3d_tests`):

```text
cos parity        max 1.19e-7  (gate <1e-6)
sin parity        max 8.94e-8  (gate <1e-6)
rotated Q parity  max 4.77e-7  (gate <1e-6)
rotated K parity  max 4.77e-7  (gate <1e-6)
identity tail     exact (pair 63 cos=1/sin=0; rotated dims 126..127 unchanged)
```

End-to-end real-contract HD128 S/M/P (`real_mmrope_contract_validation`, variant
`seedvr2_hd128_mmrope3d_hybrid`):

| Stage | HD128-S | HD128-M | HD128-P |
|---|---:|---:|---:|
| rope cos/sin max | 5.96e-8 | 5.96e-8 | 5.96e-8 |
| head0 max | 6.41e-7 | 1.19e-6 | 1.79e-6 |
| head1 max | 5.66e-7 | 1.06e-6 | 1.67e-6 |
| head2 max | 6.85e-7 | 1.10e-6 | 1.35e-6 |
| head3 max | 5.36e-7 | 9.54e-7 | 1.55e-6 |
| merged max | 6.85e-7 | 1.19e-6 | 1.79e-6 |
| projection max | 3.43e-7 | 4.32e-7 | 1.31e-6 |
| windows max | 4.17e-7 | 4.32e-7 | 1.43e-6 |
| bthwc max | 4.17e-7 | 4.32e-7 | 8.49e-7 |

All gates `<1e-4`; validity/mask exact; NaN=0, Inf=0; HD128-P validity
valid=450/invalid=702, masked key entries=44928. S->M->P->S bit-identical,
second S all cache hits, cache 0->5->10->15. Negative tests 22/22. Stability:
real-contract x5, synthetic x3, multi-shape x3, runtime x3, dispatcher x5, all PASS.

## Artifacts

New variant `seedvr2_hd128_mmrope3d_hybrid` (Option B; the synthetic variant
stays as an independent regression):

```text
tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_seedvr2_hd128_mmrope444_grid_1x1x1  (HD128-S)
tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_seedvr2_hd128_mmrope444_grid_1x2x2  (HD128-M)
tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_seedvr2_hd128_mmrope444_grid_2x3x3  (HD128-P)
```

ncnn head/projection files and conversion provenance are hard-linked from the
synthetic S artifact (weights are RoPE-agnostic; cos/sin are model inputs).
Reference and metadata are regenerated for the real contract via
`export_dynamic_rms_profile.py --hd128-real`. `pnnx fp16=0` remains enforced.

## Regressions

New clean build; all targets PASS: `artifact_registry_tests` (13 hybrids incl.
real variant + freq_mode parse negatives), `dispatcher_tests`, `model_cache_tests`,
`runtime_validation` (baseline A-E), `dynamic_padding_mask_tests`,
`dynamic_rope3d_tests` (Phase 9.5 golden + synthetic golden + real oracle parity),
`ncnn_mat_contract_tests`, `hybrid_window_adapter_tests`, `hybrid_runtime_validation`
(fixed LN/RMS), `dynamic_multi_shape_validation` (A 1.49e-7, B 1.49e-7,
C 2.086e-7), `synthetic_hd128_validation` (S/M/P), `real_mmrope_contract_validation`
(S/M/P). Historical reports (`runtime_validation.json`,
`dynamic_rms_multi_shape_runtime_validation.json`, `hybrid_fidelity_runtime_validation.json`,
`synthetic_hd128_runtime_validation.json`) are unchanged byte-for-byte.

## JSON report

`reports/real_seedvr2_mmrope_contract_validation.json`: Python-stdlib parseable,
no NaN/Inf, exactly 3 cases all PASS, `overall_status=PASS`, numeric fields are
numbers, includes SeedVR2 commit, dependency version, evidence, configured 128 /
effective 126 / identity 2, axis contract, cos/sin and per-stage errors,
regressions, memory, and latency.

## Limits and next step

Remaining real-DiT-block integration items: video-window T-axis offset by text
length l (`vid_freqs[l:l+f, :h, :w]`, l=0 in this fixture), text token RoPE,
real weights, complete DiT block, Vulkan, VAE/full pipeline.
