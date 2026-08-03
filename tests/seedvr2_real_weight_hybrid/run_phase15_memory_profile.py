#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: memory profile for the multiblock streaming adapter.

For block counts 1/2/4 and load modes streaming|resident, run the adapter in
its own process with --memory-profile and sample the process peak RSS with
psutil.  Evidence claim: the streaming-mode process peak stays roughly
constant (one block resident at a time) while the resident-mode peak grows
with the block count.

Writes reports/multiblock_memory_profile.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase15_common as p15  # noqa: E402

OUT = p15.REPO / "reports" / "multiblock_memory_profile.json"
TMP = p15.DIAG / "mem_profile_runs"
INPUT_BLOCK = 15  # chain start block (input identical for all chains)


def run_one(n: int, mode: str) -> dict:
    blocks = p15.RANGES["range_%d" % n]
    out_dir = TMP / ("n%d_%s" % (n, mode))
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = TMP / ("n%d_%s_report.json" % (n, mode))
    cmd = [
        str(p15.ADAPTER_EXE.resolve()),
        "--root", str(p15.PHASE15_ROOT.resolve()),
        "--blocks", ",".join("%d:%s" % (i, p15.variant_for(i)) for i in blocks),
        "--input-dir", str((p15.PHASE15_ROOT / "official_context_capture" / ("block_%d" % INPUT_BLOCK)).resolve()),
        "--output-dir", str(out_dir.resolve()),
        "--load-mode", mode,
        "--report", str(report_path.resolve()),
        "--memory-profile",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_rss = 0
    peak_vms = 0
    t0 = time.time()
    base_rss = None
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                m = p.memory_info()
                if base_rss is None:
                    base_rss = m.rss / (1024 * 1024)
                peak_rss = max(peak_rss, m.rss / (1024 * 1024))
                peak_vms = max(peak_vms, m.vms / (1024 * 1024))
            except Exception:
                pass
            time.sleep(0.01)
    finally:
        stdout, stderr = proc.communicate(timeout=60)
    elapsed = time.time() - t0
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text("utf-8"))
    return {
        "block_count": n, "mode": mode, "rc": proc.returncode,
        "elapsed_s": round(elapsed, 2),
        "process_peak_rss_mb": round(peak_rss, 1),
        "process_peak_vms_mb": round(peak_vms, 1),
        "base_rss_mb": round(base_rss or 0.0, 1),
        "adapter_report": report,
        "stdout_tail": (stdout or "")[-800:],
        "stderr_tail": (stderr or "")[-800:],
    }


def main() -> int:
    results = {}
    rows = []
    for n in (1, 2, 4):
        for mode in ("streaming", "resident"):
            r = run_one(n, mode)
            results["n%d_%s" % (n, mode)] = r
            rows.append((n, mode, r["rc"], r["process_peak_rss_mb"], r["process_peak_vms_mb"]))
            print("n=%d mode=%-9s rc=%d peak_rss=%.1fMB peak_vms=%.1fMB" % tuple(rows[-1]), flush=True)

    streaming_peak = {n: results["n%d_streaming" % n]["process_peak_rss_mb"] for n in (1, 2, 4)}
    resident_peak = {n: results["n%d_resident" % n]["process_peak_rss_mb"] for n in (1, 2, 4)}
    # Streaming growth factor from 1 to 4 blocks should be near 1 (not ~4).
    growth = {
        "streaming_peak_4_over_1": round(streaming_peak[4] / max(streaming_peak[1], 1.0), 3),
        "resident_peak_4_over_1": round(resident_peak[4] / max(resident_peak[1], 1.0), 3),
        "streaming_peak_4_over_2": round(streaming_peak[4] / max(streaming_peak[2], 1.0), 3),
    }
    streaming_flat = bool(growth["streaming_peak_4_over_1"] < 1.5 and growth["streaming_peak_4_over_2"] < 1.3)
    resident_grows = bool(growth["resident_peak_4_over_1"] > 1.8)
    ok = all(results["n%d_%s" % (n, m)]["rc"] == 0 for n in (1, 2, 4) for m in ("streaming", "resident"))
    ok = ok and streaming_flat and resident_grows

    report = {
        "phase": 15,
        "milestone": "multi-block streaming hybrid memory profile",
        "overall_pass": bool(ok),
        "overall_status": "PASS" if ok else "FAIL",
        "claim": "streaming process peak stays roughly constant (one block resident at a time); resident peak grows with block count",
        "streaming_peak_rss_mb": streaming_peak,
        "resident_peak_rss_mb": resident_peak,
        "growth": growth,
        "streaming_flat": streaming_flat,
        "resident_grows": resident_grows,
        "runs": results,
        "notes": [
            "process peak RSS sampled with psutil at 10 ms intervals; adapter --memory-profile snapshots (private/working set) are in adapter_report per run",
            "streaming loads/executes/unloads one block at a time (max_resident_weighted_blocks=1)",
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())