# SeedVR2 Block Weight Mapping Contract Report (Phase 13)

Status: **READY**. Machine-readable contract of the REAL SeedVR2 3B single-block
state dict -> ncnn staged-artifact mapping, generated from a real object
enumeration (`block.state_dict()`, random seed `20260803`) and verified by 12
round-trip checks x5 runs. The committed JSON is
`tests/seedvr2_weight_mapping/seedvr2_block_weight_mapping.json`; a byte-identical
copy is committed at `reports/seedvr2_block_weight_mapping.json`.

## 1. Contract schema

Per entry: `state_dict_key`, `source_module`, `source_module_class`,
`semantic_role`, `branch`, `shape`, `numel`, `dtype`, `requires_grad`, `kind`
(weight/bias), `target_stage`, `target_artifact`, `target_blob`, `conversion`,
`transpose`, `reshape`, `split`, `concat`, `expected_target_shape`,
`shared_or_independent`, `is_last_layer_applicable`.

`conversion` uses only the enum `identity | transpose_2d | split_qkv |
reshape_per_head_norm | pnnx_managed | runtime_only | not_applicable`.

## 2. Coverage (verified)

| Metric | Value |
|---|---|
| entries | 29 (28 params + buffer `attn.rope.rope.freqs`) |
| mapped_numel | 158,633,493 |
| source_numel | 158,633,493 |
| unmapped_keys | `[]` |
| duplicate_keys | `[]` |
| shape_mismatches | `[]` |
| validation | 12 checks x5 runs, all PASS, deterministic |

## 3. Mapped families

- **Attention pre-norm**: `attn_norm.vid/.txt` RMSNorm weight `[2560]`, eps from
  config (`1e-5`), no bias; stage `attn_norm`.
- **QKV**: `attn.proj_qkv.vid/.txt` `[7680,2560]`, Q/K/V row split in order
  q/k/v (split dim 0, concat dim 0 rebuilds the source exactly), no bias,
  ncnn weight orientation left to pnnx (`pnnx_managed`, `transpose=false`);
  stages `qkv_vid`/`qkv_txt`.
- **Q/K norm**: `attn.norm_q/.norm_k.vid/.txt` `[128]` per-head dim, all-ones
  affine (folded by pnnx into no-affine RMSNorm; 0-byte bins are legit);
  stages `q_norm_*`/`k_norm_*`.
- **Attention output**: `attn.proj_out.vid/.txt` `[2560,2560]` + bias `[2560]`;
  stages `attn_proj_vid`/`attn_proj_txt`.
- **MLP pre-norm**: `mlp_norm.vid/.txt` RMSNorm `[2560]`; stage `mlp_norm`.
- **SwiGLU**: `mlp.vid/.txt.proj_in_gate` and `proj_in` `[6912,2560]` (gate/up
  independent), `proj_out` `[2560,6912]`, all with bias `[6912]`/`[2560]`;
  stages `swiglu_gate_*`, `swiglu_up_*`, `swiglu_out_*` (+ shared weight-free
  `swiglu_act`).
- **AdaSingle**: no trainable parameters in the state dict; it only consumes
  the dynamic `emb` input (`state_dict_parameters=[]`,
  `runtime_dynamic_input=emb`). The six AdaSingle params (`ada.{vid,txt}.
  {attn,mlp}_{shift,scale,gate}`) are runtime-only; the local `params/ada_*.bin`
  match the source block bit-exact (verified).
- **RoPE**: `attn.rope.rope.freqs` buffer `[21]` (runtime-only cos/sin
  generation); MM-RoPE3D effective rotary dim 126, identity tail 2.

## 4. Last-layer

- Parameter sets identical with base for every shared module (attention, vid
  MLP, AdaSingle vid, RoPE); `is_last_layer` is control-flow only.
- The 9 txt-side MLP/AdaSingle entries are `is_last_layer_applicable=false`.
- All 19 base stage models are reused (`artifact_reuse`); no second export.
- Verified: last-layer state dict = 20 keys, attention/RoPE bit-identical,
  common keys shape-compatible, base subset strict-loads into the last-layer
  block (`reports/realwidth_last_layer_reuse_validation.json`).

## 5. Validation evidence

- `validate_weight_mapping_contract.py`: 12 checks - full coverage, no
  unknown/duplicate/missing keys, target shapes, identity bit-exact, transpose
  reversible, exporter-module weights bit-exact, QKV split/concat round-trip,
  last-layer rules, stdlib-JSON parseability, key-order independence.
- `test_weight_mapping_roundtrip.py`: x5 stable runs -> PASS.
- `negative_tests.py`: 25/25 controlled-failure negatives PASS.
- Reports: `reports/seedvr2_block_weight_mapping_validation.json`,
  `reports/realwidth_negative_tests.json`.

## 6. Scripts

`tests/seedvr2_weight_mapping/` - `extract_real_block_state_dict.py`,
`generate_weight_mapping_contract.py`, `validate_weight_mapping_contract.py`,
`test_weight_mapping_roundtrip.py`, `negative_tests.py`,
`generate_realwidth_artifact_inventory.py`, `validate_realwidth_artifact_inventory.py`,
`check_no_large_tracked_artifacts.py`, `verify_realwidth_last_layer.py`,
`aggregate_realwidth_feasibility.py`, `run_phase13_regressions.py`,
`_realwidth_common.py`, `README.md`.
