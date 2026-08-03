#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14 checkpoint loader preflight (stdlib only, NO network, NO download).

Checks that a real SeedVR2 3B checkpoint exists at the expected local path and
matches the documented acquisition contract.  This tool never downloads, never
logs in, never accepts licenses, and never touches third-party sources.

Exit codes:
  0  READY  - checkpoint present, file, expected name, size and (when given) sha256 OK
  2  BLOCKED_CHECKPOINT_REQUIRED - checkpoint path missing (expected during Phase 14 prep)
  3  INVALID - path exists but fails the file/name/size contract
  4  SHA_MISMATCH - file present but sha256 differs from --expected-sha256

Usage:
  python checkpoint_loader_preflight.py [--path PATH] [--expected-sha256 HEX]
      [--min-bytes N] [--report OUT.json] [--probe-schema]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_FILENAMES = ("seedvr2_ema_3b.pth", "ema_vae.pth")
DEFAULT_MIN_BYTES = 1_000_000_000  # 3B DiT checkpoint is multi-GB


def default_checkpoint_path() -> Path:
    # Optional local override: SEEDVR2_CKPT (absolute path to the checkpoint file).
    env = os.environ.get("SEEDVR2_CKPT")
    if env:
        return Path(env)
    # Repository-local ignored directory is the default landing point.
    repo = Path(__file__).resolve().parent.parent.parent
    return repo / ".phase14_local_checkpoint" / "seedvr2_ema_3b.pth"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checks(path: Path, expected_sha256: str | None, min_bytes: int) -> dict:
    checks: list[dict] = []
    ok = lambda name, value, detail="": checks.append(
        {"check": name, "passed": bool(value), "detail": detail})

    ok("path_exists", path.exists(), str(path))
    if not path.exists():
        ok("is_regular_file", False, "missing path")
        ok("expected_filename", False, "missing path")
        ok("size_ok", False, "missing path")
        ok("sha256_match", False, "missing path")
        return {
            "overall_status": "BLOCKED_CHECKPOINT_REQUIRED",
            "path": str(path),
            "checks": checks,
        }

    ok("is_regular_file", path.is_file(), "file" if path.is_file() else "not a regular file")
    ok("expected_filename", path.name in EXPECTED_FILENAMES,
       f"filename={path.name} expected in {list(EXPECTED_FILENAMES)}")
    size = path.stat().st_size
    ok("size_ok", size >= min_bytes, f"bytes={size} min={min_bytes}")

    if expected_sha256:
        actual = sha256_of(path)
        ok("sha256_match", actual.lower() == expected_sha256.lower(),
           f"actual={actual} expected={expected_sha256.lower()}")
    else:
        ok("sha256_match", True, "no expected sha256 provided (record before use)")

    status = "READY"
    if not all(c["passed"] for c in checks):
        bad = [c["check"] for c in checks if not c["passed"]]
        status = "SHA_MISMATCH" if bad == ["sha256_match"] else "INVALID"
    return {
        "overall_status": status,
        "path": str(path),
        "size_bytes": size,
        "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=None, help="explicit checkpoint path (default: SEEDVR2_CKPT env or repo .phase14_local_checkpoint/seedvr2_ema_3b.pth)")
    ap.add_argument("--expected-sha256", default=None, help="recorded sha256 to verify")
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--report", default=None, help="optional JSON report output path")
    args = ap.parse_args()

    path = Path(args.path) if args.path else default_checkpoint_path()
    result = run_checks(path, args.expected_sha256, args.min_bytes)
    result["no_network"] = True
    result["download_attempted"] = False
    result["expected_filenames"] = list(EXPECTED_FILENAMES)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    code = {"READY": 0, "BLOCKED_CHECKPOINT_REQUIRED": 2, "INVALID": 3, "SHA_MISMATCH": 4}[result["overall_status"]]
    return code


if __name__ == "__main__":
    sys.exit(main())
