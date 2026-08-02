"""Phase 13: run the real-width C++ validation + inventory tests 3x and
aggregate a single machine-readable feasibility report.

Runs `realwidth_dit_block_validation` and `realwidth_artifact_inventory_tests`
three times each in separate processes against the local (git-ignored)
`.phase13_local_artifacts/` root, parses the per-run JSON / exit codes, and
writes `reports/realwidth_seedvr2_block_feasibility.json`.  Per-run JSONs are
kept only under `.phase13_export_tmp/` (never committed).
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
BUILD = REPO / "build" / "phase13-real-width-release"
VALIDATION_EXE = BUILD / "realwidth_dit_block_validation.exe"
INVENTORY_EXE = BUILD / "realwidth_artifact_inventory_tests.exe"
TMP = rw.EXPORT_TMP


def run_exe(exe: Path, args, timeout_s=1200):
    try:
        p = subprocess.run([str(exe), *args], capture_output=True, text=True,
                           cwd=REPO, timeout=timeout_s)
        return {"exit": p.returncode, "stdout_tail": p.stdout[-600:], "stderr_tail": p.stderr[-400:]}
    except subprocess.TimeoutExpired:
        return {"exit": -1, "stdout_tail": "TIMEOUT", "stderr_tail": ""}


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    artifact_root = str(rw.LOCAL_ROOT.parent)  # scan root: registry walks child variant dirs
    manifest = str(REPO / "reports" / "realwidth_artifact_inventory.json")

    runs = []
    for i in range(1, 4):
        out_json = TMP / f"realwidth_rw_a_run{i}.json"
        r = run_exe(VALIDATION_EXE, [artifact_root, str(out_json)])
        per = {"repetition": i, "exit_code": r["exit"]}
        if out_json.exists():
            d = json.load(io.open(out_json, "r", encoding="utf-8"))
            per["pass"] = bool(d.get("overall_pass"))
            per["determinism"] = bool(d.get("determinism"))
            per["init_ms_first_load"] = d.get("init_ms_first_load")
            per["latency_run1_ms"] = (d.get("run1") or {}).get("latency_ms")
            per["latency_run2_ms"] = (d.get("run2") or {}).get("latency_ms")
            per["working_set_mb_before"] = d.get("working_set_mb_before")
            per["working_set_mb_after_load"] = d.get("working_set_mb_after_load")
            per["peak_working_set_mb"] = d.get("peak_working_set_mb")
            per["working_set_mb_after_runs"] = d.get("working_set_mb_after_runs")
            per["cache_size_after_init"] = d.get("cache_size_after_init")
        else:
            per["pass"] = False
            per["error"] = r["stdout_tail"]
        runs.append(per)
        print(f"validation run {i}: exit={per.get('exit_code')} pass={per.get('pass')} "
              f"det={per.get('determinism')} init={per.get('init_ms_first_load')}")

    inv_runs = []
    for i in range(1, 4):
        r = run_exe(INVENTORY_EXE, [artifact_root, manifest], timeout_s=600)
        inv_runs.append({"repetition": i, "exit_code": r["exit"],
                         "pass": r["exit"] == 0,
                         "stdout_tail": r["stdout_tail"][-200:]})
        print(f"inventory run {i}: exit={r['exit']} pass={r['exit'] == 0}")

    stable_det = all(r["pass"] and r["determinism"] for r in runs)
    all_pass = stable_det and all(r["pass"] for r in inv_runs) and all(r["exit_code"] == 0 for r in runs)

    report = {
        "phase": "phase13",
        "target": "realwidth_seedvr2_block_feasibility",
        "variant": rw.VARIANT,
        "sequence": "RW-A,RW-A",
        "repetitions": 3,
        "runs": runs,
        "inventory_runs": inv_runs,
        "determinism_stable_across_runs": stable_det,
        "overall_pass": bool(all_pass),
        "overall_status": "PASS" if all_pass else "FAIL",
        "real_seedvr2_weights": False,
        "realwidth_block": True,
        "full_32_block_dit": False,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }
    out = REPO / "reports" / "realwidth_seedvr2_block_feasibility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    print("overall:", report["overall_status"])
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
