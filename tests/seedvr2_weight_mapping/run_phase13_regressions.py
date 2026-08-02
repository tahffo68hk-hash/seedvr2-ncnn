"""Phase 13: full historical regression runner (Phase 9.5/10/11/12).

Runs the Phase 12 staged-DiT-block targets from the fresh Phase 13 build
(no stale ABI), the 12 core attention-dispatcher regressions from the
Phase 10.1 build, and the four Phase 11 Python suites in the isolated
probe environment.  Writes `reports/realwidth_phase13_regressions.json`.

Historical report files that these tools regenerate are intentionally NOT
committed from this run: the caller restores the tracked baselines with
`git restore` after verifying PASS (per the Phase 13 task).
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _realwidth_common as rw  # noqa: E402

REPO = rw.REPO
P13_BUILD = REPO / "build" / "phase13-real-width-release"
P101_BUILD = REPO / "build" / "phase10_1-real-mmrope-release"
PY = r"C:\Users\35287\.conda\envs\seedvr2-block-probe\python.exe"
TMP = rw.EXPORT_TMP / "regressions"
CORE_FIDELITY = "tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity"
CORE_BASELINE = "tests/attention_dispatcher/artifacts/attention_core"
DIT_ARTIFACTS = "tests/seedvr2_dit_block_ncnn/artifacts"


def run(argv, timeout_s=1800, cwd=REPO):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, cwd=cwd, timeout=timeout_s)
        tail = (p.stdout or "")[-300:] + (p.stderr or "")[-200:]
        return {"exit": p.returncode, "pass": p.returncode == 0, "tail": tail}
    except subprocess.TimeoutExpired:
        return {"exit": -1, "pass": False, "tail": "TIMEOUT"}


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "phase13",
        "target": "phase13_historical_regressions",
        "phase12_build": "build/phase13-real-width-release",
        "core_build": "build/phase10_1-real-mmrope-release",
        "phase12": {},
        "core": {},
        "phase11_python": {},
        "overall_pass": True,
        "overall_status": "PASS",
        "real_seedvr2_weights": False,
        "realwidth_block": True,
        "full_32_block_dit": False,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }

    # Phase 12 staged DiT block targets (fresh build; report JSON to tmp paths)
    p12 = [
        ("ada_single_contract_tests", [str(P13_BUILD / "ada_single_contract_tests.exe")], {}),
        ("swiglu_contract_tests", [str(P13_BUILD / "swiglu_contract_tests.exe")], {}),
        ("dit_block_artifact_registry_tests", [str(P13_BUILD / "dit_block_artifact_registry_tests.exe"), DIT_ARTIFACTS], {}),
        ("synthetic_dit_block_validation", [str(P13_BUILD / "synthetic_dit_block_validation.exe"), DIT_ARTIFACTS,
                                            str(TMP / "synthetic_dit_block_ncnn_validation.json")], {}),
        ("synthetic_dit_block_last_layer_validation", [str(P13_BUILD / "synthetic_dit_block_last_layer_validation.exe"),
                                                       DIT_ARTIFACTS, str(TMP / "synthetic_dit_block_last_layer_validation.json")], {}),
    ]
    for name, argv, _ in p12:
        r = run(argv)
        report["phase12"][name] = {k: r[k] for k in ("exit", "pass")}
        print(f"phase12 {name}: exit={r['exit']} pass={r['pass']}")
        report["overall_pass"] = report["overall_pass"] and r["pass"]

    # core dispatcher regressions (Phase 10.1 build)
    core = [
        ("artifact_registry_tests", [str(P101_BUILD / "artifact_registry_tests.exe")]),
        ("dispatcher_tests", [str(P101_BUILD / "dispatcher_tests.exe")]),
        ("model_cache_tests", [str(P101_BUILD / "model_cache_tests.exe")]),
        ("runtime_validation", [str(P101_BUILD / "runtime_validation.exe"), CORE_BASELINE]),
        ("dynamic_padding_mask_tests", [str(P101_BUILD / "dynamic_padding_mask_tests.exe")]),
        ("dynamic_rope3d_tests", [str(P101_BUILD / "dynamic_rope3d_tests.exe")]),
        ("ncnn_mat_contract_tests", [str(P101_BUILD / "ncnn_mat_contract_tests.exe")]),
        ("hybrid_window_adapter_tests", [str(P101_BUILD / "hybrid_window_adapter_tests.exe"), CORE_FIDELITY]),
        ("hybrid_runtime_validation", [str(P101_BUILD / "hybrid_runtime_validation.exe"), CORE_BASELINE, CORE_FIDELITY]),
        ("dynamic_multi_shape_validation", [str(P101_BUILD / "dynamic_multi_shape_validation.exe"), CORE_BASELINE, CORE_FIDELITY]),
        ("synthetic_hd128_validation", [str(P101_BUILD / "synthetic_hd128_validation.exe"), CORE_FIDELITY,
                                        str(REPO / "reports" / "phase10_regression_inputs.txt")]),
        ("real_mmrope_contract_validation", [str(P101_BUILD / "real_mmrope_contract_validation.exe"), CORE_FIDELITY,
                                             str(REPO / "reports" / "phase10_regression_inputs.txt")]),
    ]
    for name, argv in core:
        r = run(argv)
        report["core"][name] = {k: r[k] for k in ("exit", "pass")}
        print(f"core {name}: exit={r['exit']} pass={r['pass']}")
        report["overall_pass"] = report["overall_pass"] and r["pass"]

    # Phase 11 Python suites (isolated probe env)
    py_suites = [
        ("test_dependency_compat.py", [PY, str(REPO / "tests" / "seedvr2_dit_block_contract" / "test_dependency_compat.py")]),
        ("test_real_block_forward.py", [PY, str(REPO / "tests" / "seedvr2_dit_block_contract" / "test_real_block_forward.py")]),
        ("test_runtime_contract.py", [PY, str(REPO / "tests" / "seedvr2_dit_block_contract" / "test_runtime_contract.py")]),
        ("run_phase11_1_probe.py", [PY, str(REPO / "run_phase11_1_probe.py")]),
    ]
    for name, argv in py_suites:
        r = run(argv, timeout_s=3600)
        report["phase11_python"][name] = {k: r[k] for k in ("exit", "pass")}
        print(f"py11 {name}: exit={r['exit']} pass={r['pass']}")
        report["overall_pass"] = report["overall_pass"] and r["pass"]

    report["overall_status"] = "PASS" if report["overall_pass"] else "FAIL"
    out = REPO / "reports" / "realwidth_phase13_regressions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    print("overall:", report["overall_status"])
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
