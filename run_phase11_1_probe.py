"""Phase 11.1 runner: run the three isolated probe suites and write the report.

Usage (from the repo root, inside the isolated conda env):

    conda run -n seedvr2-block-probe python run_phase11_1_probe.py

Runs tests/seedvr2_dit_block_contract/test_dependency_compat.py,
test_real_block_forward.py and test_runtime_contract.py as subprocesses,
collects exit codes + durations, and writes reports/seedvr2_real_block_forward.json
with the overall status (READY when every suite passes), the environment and
dependency fingerprints, and the required fidelity disclaimers.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(ROOT, "tests", "seedvr2_dit_block_contract")
REPORT_PATH = os.path.join(ROOT, "reports", "seedvr2_real_block_forward.json")
WHEEL_GLOB = os.path.join(ROOT, ".phase11_1_tmp", "wheels", "*.whl")

SUITES = [
    ("dependency_compat", "test_dependency_compat.py"),
    ("real_block_forward", "test_real_block_forward.py"),
    ("runtime_contract", "test_runtime_contract.py"),
]

DISCLAIMERS = {
    "flash_attention_cuda_fidelity": "not validated",
    "apex_fused_kernel_fidelity": "not validated",
    "real_seedvr2_weights": "not used (random init)",
}


def wheel_sha256s() -> dict:
    out = {}
    for whl in sorted(glob.glob(WHEEL_GLOB)):
        h = hashlib.sha256()
        with open(whl, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[os.path.basename(whl)] = h.hexdigest()
    return out


def env_info() -> dict:
    sys.path.insert(0, TESTS_DIR)
    from probe_lib import ensure_compat
    info = ensure_compat()
    import importlib.metadata as _md
    import numpy
    import einops
    import diffusers
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "numpy": numpy.__version__,
        "einops": getattr(einops, "__version__", "?"),
        "diffusers": getattr(diffusers, "__version__", "?"),
        "rotary_embedding_torch": _md.version("rotary_embedding_torch"),
        "seedvr2_root": info["seedvr2_root"],
        "injected": info["injected"],
        "real_modules": info["real_modules"],
        "wheel_sha256": wheel_sha256s(),
    }


def main() -> int:
    env = env_info()
    suites = []
    for label, fname in SUITES:
        script = os.path.join(TESTS_DIR, fname)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, script], cwd=TESTS_DIR,
                              capture_output=True, text=True, timeout=3600)
        dt = round(time.perf_counter() - t0, 3)
        suites.append({
            "suite": label, "script": fname, "exit_code": proc.returncode,
            "duration_sec": dt, "pass": proc.returncode == 0,
            "stdout_tail": proc.stdout.strip().splitlines()[-8:],
            "stderr_tail": proc.stderr.strip().splitlines()[-8:],
        })

    overall = all(s["pass"] for s in suites)
    report = {
        "probe": "seedvr2_real_block_forward",
        "status": "READY" if overall else "FAIL",
        "overall_pass": overall,
        "duration_sec": round(sum(s["duration_sec"] for s in suites), 3),
        "env": env,
        "suites": suites,
        "disclaimers": dict(DISCLAIMERS),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("run_phase11_1_probe status =", report["status"])
    for s in suites:
        print("  ", s["suite"], "->", "PASS" if s["pass"] else "FAIL",
              f"(exit {s['exit_code']}, {s['duration_sec']}s)")
    print("report:", REPORT_PATH)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
