"""Phase 11.1 orchestrator: execute the real SeedVR2 block forward contract.

Imports the REAL NaMMSRTransformerBlock from third_party/SeedVR2 (real class,
real forward source, real 3B config dims) and runs the three Phase 11.1 test
suites in one process:

  * test_dependency_compat   - CPU-compat math parity (RMSNorm / varlen attn)
  * test_real_block_forward  - real 2560-width forwards (A/B/C/D profiles)
  * test_runtime_contract    - runtime-captured contract points + negatives

Only the fused CUDA kernel libraries (flash_attn / apex) are replaced in the
test harness by math-equivalent CPU implementations (compat/); the block
forward itself is never copied or rewritten.  No real SeedVR2 weights are
used: real_seedvr2_weights=false, random init.

On full PASS writes probe_result.json {status: PASS, ...} and exits 0.
Any suite failure writes {status: FAIL, ...} and exits 1.
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import platform
import sys
import time

import torch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MARKER = os.path.join(OUT_DIR, "probe_result.json")

DISCLAIMERS = {
    "flash_attention_cuda_fidelity": "not validated",
    "apex_fused_kernel_fidelity": "not validated",
    "real_seedvr2_weights": "not used (random init)",
}


def env_info() -> dict:
    import importlib.metadata as _md
    import numpy
    import einops
    import diffusers
    from probe_lib import ensure_compat
    info = ensure_compat()
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
    }


def run_suite(import_name: str, label: str) -> dict:
    mod = importlib.import_module(import_name)
    t0 = time.perf_counter()
    report = mod.run_all()
    dt = time.perf_counter() - t0
    report.pop("cap", None)
    return {"suite": label, "module": import_name, "all_pass": bool(report["all_pass"]),
            "duration_sec": round(dt, 3), "report": report}


def main() -> int:
    t0 = time.perf_counter()
    env = env_info()
    suites = []
    try:
        suites.append(run_suite("test_dependency_compat", "dependency_compat"))
        gc.collect()
        suites.append(run_suite("test_real_block_forward", "real_block_forward"))
        gc.collect()
        suites.append(run_suite("test_runtime_contract", "runtime_contract"))
    except Exception as e:  # noqa: BLE001
        import traceback
        marker = {"status": "ERROR", "exception": str(e),
                  "traceback": traceback.format_exc().splitlines()[-12:],
                  "env": env, "suites": suites}
        with open(MARKER, "w", encoding="utf-8") as f:
            json.dump(marker, f, indent=2, default=str)
        print("probe_real_block=ERROR " + repr(e))
        return 4

    all_pass = all(s["all_pass"] for s in suites)
    total = round(time.perf_counter() - t0, 3)
    marker = {
        "status": "PASS" if all_pass else "FAIL",
        "overall_pass": all_pass,
        "duration_sec": total,
        "env": env,
        "suites": [{"suite": s["suite"], "all_pass": s["all_pass"],
                    "duration_sec": s["duration_sec"]} for s in suites],
        "disclaimers": dict(DISCLAIMERS),
    }
    with open(MARKER, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, default=str)
    print("probe_real_block=" + marker["status"] + " duration_sec=" + str(total))
    for s in suites:
        print("  suite " + s["suite"] + " -> " + ("PASS" if s["all_pass"] else "FAIL"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
