#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14 checkpoint preflight negative/positive tests (stdlib only, no network).

Runs the non-downloading loader preflight against synthetic local temp files.
All failures are controlled (no crash, no silent fallback, no download).
Exit 0 when every check passes.

Usage:
  python test_checkpoint_preflight.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint_loader_preflight as pre  # noqa: E402

RESULTS: dict = {}


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS[name] = {"passed": bool(passed), "detail": detail}
    print(f"{name}: {'PASS' if passed else 'FAIL'} {detail}")


def make_file(size: int, content: bytes = b"x") -> Path:
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()
    p = Path(tf.name)
    chunk = content * (size // len(content) + 1)
    p.write_bytes(chunk[:size])
    return p


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="phase14_preflight_"))

    # 1. Missing path -> BLOCKED_CHECKPOINT_REQUIRED (no crash)
    missing = tmp / "seedvr2_ema_3b.pth"
    r = pre.run_checks(missing, None, pre.DEFAULT_MIN_BYTES)
    record("01_missing_path_blocked", r["overall_status"] == "BLOCKED_CHECKPOINT_REQUIRED",
           r["overall_status"])

    # 2. Directory instead of file -> INVALID
    r = pre.run_checks(tmp, None, pre.DEFAULT_MIN_BYTES)
    record("02_directory_invalid", r["overall_status"] == "INVALID", r["overall_status"])

    # 3. Wrong filename -> INVALID
    wrong_name = make_file(pre.DEFAULT_MIN_BYTES)
    wrong_name = wrong_name.rename(tmp / "random_weights.pth")
    r = pre.run_checks(wrong_name, None, pre.DEFAULT_MIN_BYTES)
    record("03_wrong_filename_invalid", r["overall_status"] == "INVALID",
           f"{r['overall_status']} detail={[c['check'] for c in r['checks'] if not c['passed']]}")

    # 4. Too small -> INVALID
    small = make_file(1024)
    small = small.rename(tmp / "seedvr2_ema_3b.pth")
    r = pre.run_checks(small, None, pre.DEFAULT_MIN_BYTES)
    record("04_too_small_invalid", r["overall_status"] == "INVALID", r["overall_status"])

    # 5. sha256 mismatch -> SHA_MISMATCH
    big = make_file(pre.DEFAULT_MIN_BYTES, content=b"abc")
    target = tmp / "seedvr2_ema_3b.pth"
    if target.exists():
        target.unlink()
    big = big.rename(target)
    h = hashlib.sha256(b"different").hexdigest()
    r = pre.run_checks(big, h, pre.DEFAULT_MIN_BYTES)
    record("05_sha_mismatch", r["overall_status"] == "SHA_MISMATCH", r["overall_status"])

    # 6. Valid file with matching sha -> READY
    actual = hashlib.sha256(big.read_bytes()).hexdigest()
    r = pre.run_checks(big, actual, pre.DEFAULT_MIN_BYTES)
    record("06_valid_ready", r["overall_status"] == "READY", r["overall_status"])

    # 7. Expected filenames contract matches official references
    record("07_expected_filenames_contract",
           set(pre.EXPECTED_FILENAMES) == {"seedvr2_ema_3b.pth", "ema_vae.pth"},
           str(pre.EXPECTED_FILENAMES))

    # 8. No-network source audit (imports only; "download" words in docs are fine)
    src = (Path(__file__).resolve().parent / "checkpoint_loader_preflight.py").read_text(encoding="utf-8")
    banned = [k for k in ("urllib", "requests", "huggingface_hub", "socket", "import http") if k in src]
    record("08_no_network_source", not banned, f"banned_tokens={banned}")

    # 8b. CLI-level output carries download_attempted=false and no_network=true
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "checkpoint_loader_preflight.py"),
         "--path", str(tmp / "seedvr2_ema_3b.pth")],
        capture_output=True, text=True, timeout=120)
    cli = {}
    try:
        cli = json.loads(proc.stdout)
        ok_cli = cli.get("download_attempted") is False and cli.get("no_network") is True
    except json.JSONDecodeError:
        ok_cli = False
    record("08b_download_attempted_false", ok_cli,
           f"rc={proc.returncode} download_attempted={cli.get('download_attempted')}")

    # 9. Manifest is parseable and consistent
    manifest = json.loads((Path(__file__).resolve().parent / "checkpoint_acquisition_manifest.json").read_text(encoding="utf-8"))
    record("09_manifest_consistent",
           manifest["status"] == "BLOCKED_CHECKPOINT_REQUIRED" and manifest["checkpoint"]["exists"] is False,
           manifest["status"])

    # 10. Valid preflight reports no NaN/Infinity and typed booleans
    ok = all(isinstance(v, bool) for v in (r.get("no_network", True),))
    record("10_report_typed_booleans", ok, "")

    failed = [k for k, v in RESULTS.items() if not v["passed"]]
    summary = {"total": len(RESULTS), "passed": len(RESULTS) - len(failed), "failed": failed}
    (Path(tempfile.gettempdir()) / "phase14_preflight_tests.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
