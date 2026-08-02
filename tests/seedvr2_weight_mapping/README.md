# Phase 13: real-width SeedVR2 single block feasibility + weight mapping contract

Extends the Phase 12 scaled fixture to the real SeedVR2 3B single-block width
(C=2560 / H=20 / HD=128) with deterministic random weights (seed 20260803) and
establishes a complete, machine-readable, verifiable mapping contract from a
real `NaMMSRTransformerBlock` state_dict to the ncnn stage artifacts consumed
by `HybridDiTBlockRuntime`.

All heavy binaries (random state_dict, exported real-width FP32 stage models,
RW-A reference bins) live in the git-ignored `.phase13_local_artifacts/`
directory; only scripts, manifests, the committed mapping contract JSON, tests
and reports are committed.

## Files

- `extract_real_block_state_dict.py` - instantiates the REAL block at
  C=2560/H=20/HD=128 (seed 20260803, FP32 CPU eval/no_grad) and enumerates the
  full state_dict from the real object graph -> local inventory JSON.
- `generate_weight_mapping_contract.py` - derives the committed
  `seedvr2_block_weight_mapping.json` (target stage/artifact/blob, conversion
  enum, QKV split, RMSNorm/SwiGLU/AdaSingle/last-layer semantics).
- `validate_weight_mapping_contract.py` - 12 round-trip checks: full coverage,
  no unknown/duplicate keys, exact numel, target shapes, bit-exact identity /
  exporter-module weights, QKV split/concat, last-layer rules, stdlib-JSON,
  key-order independence, ada params vs params dir.
- `test_weight_mapping_roundtrip.py` - runs the validation 5x (stability) and
  writes `reports/seedvr2_block_weight_mapping_validation.json`.
- `check_no_large_tracked_artifacts.py` - Git protection gate (no
  .pt/.pth/.ckpt/.safetensors, no real-width .bin, no binary > 10 MiB,
  `.phase13_*` ignored) -> `reports/realwidth_git_protection.json`.
- `generate_realwidth_artifact_inventory.py` / `validate_realwidth_artifact_inventory.py` -
  committed inventory manifest (38 entries: sizes + SHA256 + parameter counts +
  `pnnx_folded_identity_weight`) and x3 stability validation.
- `verify_realwidth_last_layer.py` - base vs last-layer state_dict comparison
  (attention/RoPE bit-identical, txt MLP/ada absent, 19-model reuse) ->
  `reports/realwidth_last_layer_reuse_validation.json`.
- `negative_tests.py` - 25 controlled-failure negatives ->
  `reports/realwidth_negative_tests.json`.
- `aggregate_realwidth_feasibility.py` - runs the real-width C++ validation +
  inventory tests x3 and aggregates `reports/realwidth_seedvr2_block_feasibility.json`.
- `run_phase13_regressions.py` - Phase 12 + core + Phase 11 Python regression
  runner -> `reports/realwidth_phase13_regressions.json`.

## Run order

1. `extract_real_block_state_dict.py`
2. `generate_weight_mapping_contract.py`
3. `test_weight_mapping_roundtrip.py` (validates x5)
4. `check_no_large_tracked_artifacts.py` (before commit)

Python: `C:\Users\35287\.conda\envs\seedvr2-block-probe\python.exe`
SeedVR2: `D:\AIProjects\third_party\SeedVR2` at `e4de8c2` (read-only).
