# Project memory index

- `PROJECT_CONTEXT.md`: scope and environment.
- `ARCHITECTURE.md`: runtime ownership and modes.
- `CURRENT_STATUS.md`: validated state and next task.
- `DECISIONS_AND_GOTCHAS.md`: non-obvious constraints.
- `TEST_MATRIX.md`: commands and acceptance gates.
- `CODEX_HANDOFF.md`: entry point for a new agent.
- `NEXT_TASK.md`: copyable next-task prompt (Phase 13 real-width artifact feasibility + real weight mapping contract).
- `synthetic-hd128-runtime-report.md`: Phase 10 synthetic head_dim=128 report (historical regression, READY superseded).
- `real-seedvr2-mmrope-contract-report.md`: Phase 10.1 real SeedVR2 MM-RoPE3D contract report (Phase 10 READY).
- `seedvr2-single-dit-block-contract-report.md` + `reports/seedvr2_single_dit_block_contract.json`: Phase 11 real single DiT block interface/shape/execution contract (READY; real forward executed in isolated conda env `seedvr2-block-probe`).
- `reports/seedvr2_real_block_forward.json`: Phase 11.1 runner report (subprocess suites, env + wheel SHA256, disclaimers).
- `synthetic-seedvr2-dit-block-ncnn-report.md` + `reports/synthetic_dit_block_ncnn_validation.json` + `reports/synthetic_dit_block_last_layer_validation.json`: Phase 12 synthetic single DiT block staged ncnn fidelity (READY; `tests/seedvr2_dit_block_ncnn/` runtime/artifacts/reference generator).
- `tests/seedvr2_dit_block_contract/`: Phase 11 extract/probe scripts (`probe_real_block.py` orchestrator, `run_phase11_1_probe.py` runner at repo root, `test_*.py` suites, `compat/` CPU replacements), `seedvr2_block_contract.json`, README.
- `realwidth-seedvr2-block-feasibility-report.md` + `reports/realwidth_seedvr2_block_feasibility.json` + `reports/realwidth_artifact_inventory.json` + `reports/realwidth_last_layer_reuse_validation.json` + `reports/realwidth_phase13_regressions.json`: Phase 13 real-width C=2560/H=20/HD=128 staged ncnn feasibility (READY; local random weights, 19 models, RW-A parity, resources, 32-block/mobile estimates).
- `seedvr2-block-weight-mapping-report.md` + `tests/seedvr2_weight_mapping/seedvr2_block_weight_mapping.json` (copy in `reports/seedvr2_block_weight_mapping.json`) + `reports/seedvr2_block_weight_mapping_validation.json` + `reports/realwidth_negative_tests.json` + `reports/realwidth_git_protection.json`: Phase 13 real checkpoint -> ncnn weight mapping contract (29/29, 12 checks x5 PASS, negatives 25/25, git large-artifact protection).
- `tests/seedvr2_weight_mapping/`: Phase 13 extraction/mapping/inventory/negative/protection/aggregation scripts (`_realwidth_common.py`, `extract_real_block_state_dict.py`, `generate_weight_mapping_contract.py`, `validate_weight_mapping_contract.py`, `test_weight_mapping_roundtrip.py`, `negative_tests.py`, `generate_realwidth_artifact_inventory.py`, `validate_realwidth_artifact_inventory.py`, `check_no_large_tracked_artifacts.py`, `verify_realwidth_last_layer.py`, `aggregate_realwidth_feasibility.py`, `run_phase13_regressions.py`, README).
- `tests/seedvr2_dit_block_ncnn/export_realwidth_dit_block.py`, `generate_realwidth_dit_block_reference.py`, `realwidth_dit_block_validation.cpp`, `realwidth_artifact_inventory_tests.cpp`: Phase 13 stream export, RW-A reference generation and C++ validation targets (heavy outputs in git-ignored `.phase13_local_artifacts/`).
