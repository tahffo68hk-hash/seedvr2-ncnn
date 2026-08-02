# Project memory index

- `PROJECT_CONTEXT.md`: scope and environment.
- `ARCHITECTURE.md`: runtime ownership and modes.
- `CURRENT_STATUS.md`: validated state and next task.
- `DECISIONS_AND_GOTCHAS.md`: non-obvious constraints.
- `TEST_MATRIX.md`: commands and acceptance gates.
- `CODEX_HANDOFF.md`: entry point for a new agent.
- `NEXT_TASK.md`: copyable next-task prompt (Phase 12 synthetic single DiT block staged fidelity).
- `synthetic-hd128-runtime-report.md`: Phase 10 synthetic head_dim=128 report (historical regression, READY superseded).
- `real-seedvr2-mmrope-contract-report.md`: Phase 10.1 real SeedVR2 MM-RoPE3D contract report (Phase 10 READY).
- `seedvr2-single-dit-block-contract-report.md` + `reports/seedvr2_single_dit_block_contract.json`: Phase 11 real single DiT block interface/shape/execution contract (READY; real forward executed in isolated conda env `seedvr2-block-probe`).
- `reports/seedvr2_real_block_forward.json`: Phase 11.1 runner report (subprocess suites, env + wheel SHA256, disclaimers).
- `tests/seedvr2_dit_block_contract/`: Phase 11 extract/probe scripts (`probe_real_block.py` orchestrator, `run_phase11_1_probe.py` runner at repo root, `test_*.py` suites, `compat/` CPU replacements), `seedvr2_block_contract.json`, README.
