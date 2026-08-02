"""Phase 13: Git large-file / artifact protection check.

Verifies before commit that:
- no tracked/staged file is a checkpoint/model weight (.pt/.pth/.ckpt/.safetensors)
- no real-width or large binary is tracked (threshold 10 MiB; Phase 12 max tracked
  binary is ~3.15 MiB so the threshold is well above legitimate small artifacts)
- `.phase13_local_artifacts/` and `.phase13_export_tmp/` are git-ignored and untracked
- the committed inventory manifest only stores small metadata (no file contents)

Exit code 0 on pass.  Writes `reports/realwidth_git_protection.json`.
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
LARGE_BYTES = 10 * 1024 * 1024
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".pt2")
IGNORED_DIRS = (".phase13_local_artifacts", ".phase13_export_tmp")


def git(args, cwd=REPO):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)


def main() -> int:
    report = {
        "phase": "phase13",
        "target": "git_large_artifact_protection",
        "large_binary_threshold_bytes": LARGE_BYTES,
        "checks": {},
        "overall_pass": True,
        "overall_status": "PASS",
    }
    checks = report["checks"]

    tracked = git(["ls-files"]).stdout.splitlines()
    staged = git(["diff", "--cached", "--name-only"]).stdout.splitlines()

    weight_tracked = [f for f in tracked if f.lower().endswith(WEIGHT_SUFFIXES)]
    weight_staged = [f for f in staged if f.lower().endswith(WEIGHT_SUFFIXES)]
    checks["no_checkpoint_or_weight_tracked"] = (weight_tracked == [])
    checks["no_checkpoint_or_weight_staged"] = (weight_staged == [])
    report["weight_files_tracked"] = weight_tracked
    report["weight_files_staged"] = weight_staged

    realwidth_bins = [f for f in tracked if "realwidth" in f.lower() and f.endswith(".bin")]
    checks["no_realwidth_bin_tracked"] = (realwidth_bins == [])
    report["realwidth_bins_tracked"] = realwidth_bins

    large = []
    for f in tracked:
        try:
            size = (REPO / f).stat().st_size
        except OSError:
            continue
        if size > LARGE_BYTES:
            large.append({"file": f, "bytes": int(size)})
    checks["no_large_binary_tracked"] = (large == [])
    report["large_files"] = large

    ignored_ok = True
    for d in IGNORED_DIRS:
        if d in tracked:
            ignored_ok = False
        ig = git(["check-ignore", "-q", d + "/probe"])
        checks[f"{d}_gitignored"] = (ig.returncode == 0)
        ignored_ok = ignored_ok and checks[f"{d}_gitignored"]
    checks["phase13_dirs_untracked_and_ignored"] = ignored_ok

    inventory = REPO / "reports" / "realwidth_artifact_inventory.json"
    report["inventory_manifest_path"] = str(inventory)
    if inventory.exists():
        inv = json.load(io.open(inventory, "r", encoding="utf-8"))
        entries = inv.get("entries", [])
        total_bytes = sum(int(e.get("bytes", 0)) for e in entries)
        checks["inventory_manifest_no_file_contents"] = all(
            set(e.keys()) <= {"logical_name", "local_relative_path", "bytes", "sha256",
                              "parameter_count", "stage", "pnnx_folded_identity_weight"} for e in entries)
        report["inventory_count"] = len(entries)
        report["inventory_total_bytes"] = total_bytes
    else:
        checks["inventory_manifest_no_file_contents"] = True

    report["overall_pass"] = all(checks.values())
    report["overall_status"] = "PASS" if report["overall_pass"] else "FAIL"
    out = REPO / "reports" / "realwidth_git_protection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in checks.items()}, indent=1))
    print("overall_pass:", report["overall_pass"])
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
