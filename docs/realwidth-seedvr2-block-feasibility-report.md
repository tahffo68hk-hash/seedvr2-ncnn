# Real-Width SeedVR2 Single Block Feasibility Report (Phase 13)

Status: **READY** (commit on `codex/phase13-real-width-weight-map`, base `c05a5a4`
Phase 12 READY). Scope: extend the Phase 12 scaled staged runtime to the REAL
SeedVR2 3B single-block width `C=2560 / H=20 / HD=128` with deterministic random
weights, export all stage artifacts with `pnnx fp16=0`, run the minimal RW-A
profile end-to-end on ncnn, measure resources, prove last-layer reuse, run all
historical regressions, and commit a full machine-readable weight mapping
contract. No real SeedVR2 checkpoint was downloaded; 32-block DiT, Android and
Vulkan are NOT started.

## 1. Provenance

- Repo branch: `codex/phase13-real-width-weight-map`; base commit `c05a5a4` (Phase 12 READY).
- SeedVR2: `e4de8c2` (`D:\AIProjects\third_party\SeedVR2`, clean worktree, unmodified).
- ncnn: `a4d2ea1d` (`D:\AIProjects\third_party\ncnn`, unmodified; install `build/ncnn-cpu/install`, cmake version `20260730`).
- pnnx: built from the same ncnn tree (`build/pnnx-cpu/src/pnnx.exe`), `fp16=0` on every export; runs against libtorch `2.7.0+cpu` (`V:\AI\LibTorch\2.7.0+cpu\lib`).
- Extraction/probe env: isolated conda `seedvr2-block-probe` (Python 3.11, torch `2.4.0+cpu`, rotary-embedding-torch 0.5.3, einops, diffusers).
- Block: real `NaMMSRTransformerBlock` (`models.dit_v2.nablocks.mmsr_block`, SeedVR2 `e4de8c2`); random seed `20260803`, FP32, CPU, `eval()`, `torch.no_grad()`, `real_seedvr2_weights=false`.

## 2. Real block configuration

```
vid_dim=2560 txt_dim=2560 emb_dim=15360 heads=20 head_dim=128
QKV output 7680, effective rotary dim 126 (configured 128, identity tail 2)
axis_dims=[42,42,42], window=[4,3,3], norm_eps=1e-5, ada=single
mlp_type=swiglu, mlp_hidden_dim=6912 (256*ceil(2*2560*4/3/256))
shared_weights=false, is_last_layer=false, window_method=720pwin_by_size_bysize
```

## 3. Parameter counts and sizes (measured)

| Item | Value |
|---|---|
| state_dict keys | 29 (28 params + buffer `attn.rope.rope.freqs` [21]) |
| numel (incl. rope freqs buffer) | 158,633,493 |
| theoretical FP32 bytes | 634,533,972 (~634.5 MB) |
| `block_state.pt` on disk | 634,542,851 bytes |
| ncnn artifacts (19 models, 38 files) | 634,413,578 bytes |
| mapping coverage | 29/29 entries, `mapped_numel == source_numel` |
| unmapped / duplicate keys | `[]` / `[]` |

Per-stage ncnn bins: `qkv_*` 78,643,204 B; `swiglu_*` 70,778,884 B; `attn_proj_*`
26,224,648 B; the four q/k-norm stages export 0-byte bins because the real
qk-norm RMSNorm weights are all-ones and pnnx folds them
(`pnnx_folded_identity_weight=true`, documented in the inventory manifest);
norm/rope/head/swiglu_act stages are weight-free (0-byte bins, same as Phase 12).

## 4. RW-A ncnn end-to-end execution

Profile RW-A: `batch=1, vid_shape=[1,4,4], text_length=2, hidden_dim=2560,
heads=20, head_dim=128`. Sequence `RW-A -> RW-A` (two complete forwards in one
process, 3 separate processes).

- Every captured stage passes its Phase 13 threshold (RMSNorm/Ada-in `<1e-5`,
  Linear/QKV/rotated Q/K `<2e-5`, RoPE cos/sin `<1e-6`, packed
  attention/projection `<1e-4`, SwiGLU product `<2e-4`, residuals/finals
  `<2e-4`). Worst measured final error: vid `1.00e-5`, txt `6.4e-6`.
- Determinism: bit-identical across both forwards and across the 3 processes
  (`determinism_stable_across_runs=true`).
- Zero NaN / Inf on every stage.

## 5. Resources (measured)

| Metric | Value |
|---|---|
| first-load (19 models) | 236-258 ms |
| cached forward latency | run1 ~33.9-35.5 ms, run2 ~30.3-31.7 ms |
| working set before load | ~5.6 MB |
| working set after all 19 models loaded | ~618 MB |
| peak working set | ~685 MB |
| working set after two forwards | ~639 MB |
| ncnn model cache | 19 unique models (20 heads share one weight-free `head` model) |

## 6. 32-block estimates (ESTIMATES ONLY, not measured)

- 32 blocks = 31 regular blocks + 1 last-layer block (last-layer is the same
  parameter set minus 9 txt-side MLP/AdaSingle entries; ~105.5M params).
- Total params ~5.023B; resident FP32 ~20.09 GB; FP16 ~10.05 GB.
- ncnn artifact disk (no cross-block sharing): ~20.3 GB FP32.
- Streaming per-block would keep ~0.64-0.69 GB resident per block plus
  activations, which is the only realistic large-model path.

## 7. Pad 7 Pro pre-conclusion (ESTIMATES ONLY, no Android work done)

- FP32 single block (~634 MB weights, ~0.69 GB peak working set on PC) is
  plausibly loadable on a tablet-class device with >=8 GB RAM, but this was not
  validated on device.
- FP32 32-block all-resident (~20 GB) is unrealistic on a tablet.
- FP16 theoretical weight volume ~10.05 GB still requires per-block streaming.
- Block-by-block streaming and/or weight sharing / chunked loading is required
  before Vulkan; the largest remaining memory problem is 32-block resident
  weights plus activations.
- Accurate status: PC CPU real-width single-block feasibility = validated;
  mobile deployment = not validated.

## 8. Last-layer reuse

- Base (29 keys) vs last-layer (20 keys): attention + RoPE keys bit-identical;
  the 9 txt-side MLP/AdaSingle entries are absent (vid_only control flow);
  the remaining 20 keys strict-load into the last-layer block.
- All 19 base stage models are reused; no second real-width export.
- Real-width numeric last-layer execution was NOT repeated (Phase 12 scaled
  last-layer numeric validation already covers numerics) - documented in
  `reports/realwidth_last_layer_reuse_validation.json`.

## 9. Negatives, protection, regressions

- Weight-mapping / artifact negatives: 25/25 PASS (controlled failures, no
  crash, no silent fallback).
- Git protection: no tracked/staged checkpoint, no real-width `.bin`, no
  large binary, both Phase 13 local dirs ignored - PASS.
- Mapping contract validation: 12 round-trip checks x5 runs, all PASS and
  stable.
- Inventory validation: 38 entries x3 runs, SHA256/size/fold markers PASS and
  stable.
- Historical regressions: Phase 12 (5 targets, fresh build) + 12 core
  dispatcher suites (Phase 10.1 build) + 4 Phase 11 Python suites: 21/21 PASS.
  Historical report baselines were restored byte-identical after the runs.

## 10. Flags

`real_seedvr2_weights=false`, `realwidth_block=true`, `full_32_block_dit=false`,
`android_vulkan=false`, `large_artifacts_committed=false`.

## 11. Reports and artifacts

- `reports/realwidth_seedvr2_block_feasibility.json` (aggregated x3 runs).
- `reports/realwidth_artifact_inventory.json` + validation; `reports/realwidth_negative_tests.json`.
- `reports/seedvr2_block_weight_mapping.json` + `reports/seedvr2_block_weight_mapping_validation.json`.
- `reports/realwidth_last_layer_reuse_validation.json`, `reports/realwidth_phase13_regressions.json`,
  `reports/realwidth_git_protection.json`.
- Local heavy artifacts (git-ignored): `.phase13_local_artifacts/realwidth_seedvr2_dit_block_c2560_h20_hd128_random/`.
- Scripts: `tests/seedvr2_weight_mapping/`, `tests/seedvr2_dit_block_ncnn/export_realwidth_dit_block.py`,
  `tests/seedvr2_dit_block_ncnn/generate_realwidth_dit_block_reference.py`,
  `tests/seedvr2_dit_block_ncnn/realwidth_dit_block_validation.cpp`,
  `tests/seedvr2_dit_block_ncnn/realwidth_artifact_inventory_tests.cpp`.

## 12. Status

PC CPU real-width single-block feasibility: **validated**. Mobile deployment,
real SeedVR2 weights, 32-block DiT, Android/Vulkan: **not validated / TODO**
(Phase 14+).
