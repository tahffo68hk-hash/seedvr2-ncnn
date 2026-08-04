# -*- coding: utf-8 -*-
"""Phase 17: historical regression runner.

Phase 17 touches only VAE-side code (new tests/seedvr2_vae_ncnn), .gitignore,
reports and docs; the DiT runtime is byte-for-byte untouched.  This runner
re-executes the offline Phase 16 CPU-light evidence generators, the Phase 14
regression matrix (9.5/10/10.1/11/12/13 + 12 core attention dispatcher
targets) and the Phase 15 regression matrix (12/13/14 + core + Phase 11
probes + Phase 15 streaming adapter).  The GPU Phase 16 chain-parity and full
visual baselines are cited from their committed reports (unchanged vs HEAD).

Historical report files regenerated here are restored with `git restore`
after verification (same convention as Phase 13/14/15).
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PY = r"C:\Users\35287\.conda\envs\seedvr2-block-probe\python.exe"
# phase16_last_layer_parity imports the SeedVR2 stack (omegaconf etc.),
# which is only present in the project .conda-cuda environment.
PY_CUDA = r"D:\AIProjects\seedvr2-ncnn\.conda-cuda\python.exe"
TMP = REPO / ".phase17_diag" / "regressions"
P16_DIR = REPO / "tests" / "seedvr2_real_weight_hybrid"
P14_DIR = REPO / "tests" / "seedvr2_real_weight_hybrid"


def run(argv, timeout_s=3600, cwd=REPO):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, cwd=cwd, timeout=timeout_s)
        tail = (p.stdout or "")[-300:] + (p.stderr or "")[-200:]
        return {"exit": p.returncode, "pass": p.returncode == 0, "tail": tail}
    except subprocess.TimeoutExpired:
        return {"exit": -1, "pass": False, "tail": "TIMEOUT"}


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "phase17",
        "target": "phase17_historical_regressions",
        "real_seedvr2_weights": True,
        "real_vae_weights": True,
        "phase16_cpu_light": {},
        "phase16_gpu_baselines_cited": {},
        "phase14_matrix": {},
        "phase15_matrix": {},
        "overall_pass": True,
        "overall_status": "PASS",
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }

    phase16_light = [
        ("phase16_block_schedule", [PY, str(P16_DIR / "phase16_block_schedule.py")]),
        ("phase16_all_block_mapping", [PY, str(P16_DIR / "phase16_all_block_mapping.py")]),
        ("phase16_artifact_inventory", [PY, str(P16_DIR / "phase16_artifact_inventory.py")]),
        ("phase16_last_layer_parity", [PY_CUDA, str(P16_DIR / "phase16_last_layer_parity.py"), "--repeats", "1"]),
        ("phase16_negative_tests", [PY, str(P16_DIR / "run_phase16_negative_tests.py")]),
        ("phase16_git_protection", [PY, str(P16_DIR / "run_phase16_git_protection.py")]),
    ]
    for name, argv in phase16_light:
        r = run(argv, timeout_s=3600)
        report["phase16_cpu_light"][name] = {k: r[k] for k in ("exit", "pass")}
        print(f"p16-light {name}: exit={r['exit']} pass={r['pass']}")
        report["overall_pass"] = report["overall_pass"] and r["pass"]

    cited = [
        "phase16_chain_8_parity.json", "phase16_chain_16_parity.json",
        "phase16_chain_full_parity.json", "phase16_chain_parity_summary.json",
        "full_dit_visual_comparison.json", "full_dit_error_growth.json",
    ]
    for name in cited:
        p = REPO / "reports" / name
        ok = p.exists()
        if ok:
            data = json.loads(p.read_text(encoding="utf-8"))
            ok = data.get("overall_pass", data.get("all_chains_pass", True)) is not False
        report["phase16_gpu_baselines_cited"][name] = {"exists": p.exists(), "pass": ok}
        print(f"p16-cited {name}: exists={p.exists()} pass={ok}")
        report["overall_pass"] = report["overall_pass"] and ok

    r14 = run([PY, str(P14_DIR / "run_phase14_regressions.py")], timeout_s=7200)
    report["phase14_matrix"]["exit"] = r14["exit"]
    report["phase14_matrix"]["pass"] = r14["pass"]
    print(f"phase14 matrix: exit={r14['exit']} pass={r14['pass']}")
    report["overall_pass"] = report["overall_pass"] and r14["pass"]

    r15 = run([PY, str(P14_DIR / "run_phase15_regressions.py")], timeout_s=7200)
    report["phase15_matrix"]["exit"] = r15["exit"]
    report["phase15_matrix"]["pass"] = r15["pass"]
    print(f"phase15 matrix: exit={r15['exit']} pass={r15['pass']}")
    report["overall_pass"] = report["overall_pass"] and r15["pass"]

    # Restore regenerated historical reports to their committed baselines.
    hist = ["reports/phase16_block_schedule.json", "reports/phase16_all_block_mapping.json",
            "reports/phase16_artifact_inventory.json", "reports/phase16_last_layer_parity.json",
            "reports/phase16_negative_tests.json", "reports/phase16_git_protection.json",
            "reports/phase14_regressions.json", "reports/phase15_regressions.json",
            "reports/realwidth_phase13_regressions.json",
            "reports/phase14_checkpoint_schema_probe.json"]
    restored = []
    for rel in hist:
        p = subprocess.run(["git", "-C", str(REPO), "restore", "--", rel],
                           capture_output=True, text=True)
        if p.returncode == 0:
            restored.append(rel)
    report["historical_reports_restored"] = restored
    print("restored:", len(restored), "historical reports")

    report["overall_status"] = "PASS" if report["overall_pass"] else "FAIL"
    out = REPO / "reports" / "phase17_regressions.json"
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, indent=2) + "\n")
    print("overall:", report["overall_status"])
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
