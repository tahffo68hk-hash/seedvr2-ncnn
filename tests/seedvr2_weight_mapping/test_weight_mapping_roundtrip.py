"""Phase 13: mapping round-trip stability test (weight_mapping_tests x5).

Runs the full mapping validation 5 times in one process and asserts the
result is stable and passing (weight mapping must be deterministic).
Writes `reports/seedvr2_block_weight_mapping_validation.json`.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _realwidth_common as rw  # noqa: E402
from validate_weight_mapping_contract import run_validation  # noqa: E402


def main() -> int:
    report = {
        "phase": "phase13",
        "target": "realwidth_weight_mapping_tests",
        "repetitions": 5,
        "runs": [],
        "overall_pass": True,
        "overall_status": "PASS",
        "real_seedvr2_weights": False,
        "realwidth_block": True,
        "full_32_block_dit": False,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }
    first = None
    stable = True
    for i in range(5):
        res = run_validation()
        if first is None:
            first = json.dumps(res, sort_keys=True)
        else:
            stable = stable and json.dumps(res, sort_keys=True) == first
        report["runs"].append({"repetition": i + 1, "overall_pass": res["ok"],
                               "mapped_numel": res["coverage"]["mapped_numel"],
                               "source_numel": res["coverage"]["source_numel"]})
        report["overall_pass"] = report["overall_pass"] and res["ok"]
        print(f"run {i+1}: pass={res['ok']}")
    report["determinism_stable_across_runs"] = stable
    report["overall_pass"] = report["overall_pass"] and stable
    report["overall_status"] = "PASS" if report["overall_pass"] else "FAIL"
    out = rw.REPO / "reports" / "seedvr2_block_weight_mapping_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    print("overall_pass:", report["overall_pass"], "stable:", stable)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
