#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: chain parity orchestrator (8/16/full, offline, real context).

Runs the phase16_chain_worker for chain_8 (blocks 9..16), chain_16
(blocks 8..23) and chain_full (blocks 0..31 incl. last layer) in isolated
GPU processes (chain_8 x3, chain_16 x3, chain_full x1 with 3 internal
repeats) and writes reports/phase16_chain_8_parity.json,
reports/phase16_chain_16_parity.json and reports/phase16_chain_full_parity.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase16_common as p16  # noqa: E402

CHAINS = [
    ("chain_8", p16.RANGE_8, 9, 3),
    ("chain_16", p16.RANGE_16, 8, 3),
    ("chain_full", p16.RANGE_FULL, 0, 1),
]
REPORT_NAMES = {
    "chain_8": "phase16_chain_8_parity.json",
    "chain_16": "phase16_chain_16_parity.json",
    "chain_full": "phase16_chain_full_parity.json",
}


def main() -> int:
    all_ok = True
    results = {}
    for name, blocks, bi0, runs in CHAINS:
        per_run = []
        for r in range(runs):
            out = p16.PHASE16_ROOT / "chain_parity" / f"{name}_run{r}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(p16.GPU_PY), str(HERE / "phase16_chain_worker.py"),
                   "--chain", name, "--blocks", ",".join(str(b) for b in blocks),
                   "--input-block", str(bi0), "--repeats", "3", "--out", str(out)]
            if name == "chain_full":
                cmd.append("--no-resident")
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            dt = time.time() - t0
            tail = (proc.stdout or "")[-800:] + (proc.stderr or "")[-800:]
            print(f"{name} run{r}: rc={proc.returncode} {dt:.1f}s")
            if proc.returncode != 0:
                print("  ", tail.replace(chr(10), " ")[-600:])
                all_ok = False
            else:
                per_run.append(json.loads(out.read_text("utf-8")))
        if per_run:
            results[name] = per_run
    report = {
        "phase": 16,
        "real_seedvr2_weights": True,
        "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
        "chains": {
            name: {
                "range": CHAINS[i][1],
                "input_block": CHAINS[i][2],
                "runs": len(per_run),
                "rel_limit": p16.CHAIN_REL_LIMITS[len(CHAINS[i][1])],
                "final_relative_errors": [
                    r["final_error"][0]["relative_max_error"] if r.get("final_error") else None
                    for r in per_run],
                "pass": bool(per_run) and all(
                    r.get("pass", False) for r in per_run),
            }
            for i, (name, per_run) in enumerate(results.items())
        },
        "all_chains_pass": bool(results) and all(
            r.get("pass", False) for r in results["chain_full"]) and
            all(r.get("pass", False) for rr in results.values() for r in rr),
    }
    for name, fname in REPORT_NAMES.items():
        if name in results:
            (p16.REPO / "reports" / fname).write_text(
                json.dumps({"phase": 16, "chain": name, "runs": results[name]},
                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (p16.REPO / "reports" / "phase16_chain_parity_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["chains"], indent=2))
    return 0 if report["all_chains_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
