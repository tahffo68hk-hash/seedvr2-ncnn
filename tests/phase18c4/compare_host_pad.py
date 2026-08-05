#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--pad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host = np.fromfile(args.host, dtype="<f4")
    pad = np.fromfile(args.pad, dtype="<f4")
    expected = 512 * 5 * 64 * 64
    if host.size != expected or pad.size != expected:
        raise SystemExit(f"element mismatch host={host.size} pad={pad.size} expected={expected}")
    delta = host.astype(np.float64) - pad.astype(np.float64)
    denom = float(np.linalg.norm(host.astype(np.float64)) * np.linalg.norm(pad.astype(np.float64)))
    report = {
        "schema_version": 1,
        "shape": [512, 5, 64, 64],
        "host_sha256": digest(args.host), "pad_sha256": digest(args.pad),
        "finite": bool(np.isfinite(host).all() and np.isfinite(pad).all()),
        "nan_count": int(np.isnan(host).sum() + np.isnan(pad).sum()),
        "inf_count": int(np.isinf(host).sum() + np.isinf(pad).sum()),
        "max_abs_error": float(np.abs(delta).max()),
        "mean_abs_error": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": float(np.dot(host.astype(np.float64), pad.astype(np.float64)) / denom),
        "tolerance_max_abs": 1e-3,
    }
    report["overall_pass"] = report["finite"] and report["max_abs_error"] <= report["tolerance_max_abs"]
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
