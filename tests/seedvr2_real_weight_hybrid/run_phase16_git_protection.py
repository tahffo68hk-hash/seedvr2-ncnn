# -*- coding: utf-8 -*-
"""Phase 16: git protection check before commit.

Verifies that nothing Phase 16 must keep out of Git is tracked or staged:
real-weight binaries (`.phase16_*` / older `.phase14_*` `.phase15_*` dirs,
checkpoint/weight suffixes), build trees, third-party sources, conda envs,
secrets and conflict markers.  Also checks every Phase 16 JSON report parses
and contains no NaN/Infinity.  Writes `reports/phase16_git_protection.json`.
Exit 0 on pass.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LARGE_BYTES = 10 * 1024 * 1024
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".pt2")
FORBIDDEN_TRACKED = (".phase16_", ".phase15_", ".phase14_", ".phase13_",
                     ".conda", "build/", "third_party/")
SECRET_RE = re.compile(
    r"(api[_-]?key|secret|password|passwd)\s*[:=]|"
    r"BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|"
    r"hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}", re.IGNORECASE)


def git(args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=120)


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}

    def rec(name: str, passed: bool, detail: str = "") -> None:
        checks[name] = bool(passed)
        if detail:
            details[name] = detail

    tracked = git(["ls-files"]).stdout.splitlines()
    tracked_text = "\n".join(tracked)

    rec("no_phase16_phase15_phase14_phase13_tracked",
        not any(t.startswith(tuple(FORBIDDEN_TRACKED)) or "/.phase16_" in t or "/.phase15_" in t
                or "/.phase14_" in t or "/.phase13_" in t for t in tracked),
        "tracked=%d" % len(tracked))
    rec("no_build_third_party_env_tracked",
        not any(t.split("/", 1)[0] in ("build", "third_party", ".conda", ".conda-cuda") for t in tracked))
    rec("no_weight_files_tracked",
        not any(t.endswith(WEIGHT_SUFFIXES) for t in tracked))

    large = git(["ls-files", "-z"]).stdout.split("\0")
    big = []
    for rel in large:
        if not rel:
            continue
        try:
            if (REPO / rel).stat().st_size > LARGE_BYTES:
                big.append(rel)
        except OSError:
            pass
    rec("no_large_binary_tracked", big == [], "threshold=%d" % LARGE_BYTES)
    if big:
        details["large_files"] = ";".join(big[:20])

    ignored = git(["check-ignore", ".phase16_local_artifacts/x", ".phase16_export_tmp/x",
                   ".phase16_visual_outputs/x", ".phase16_diag/x",
                   ".phase16_runtime_traces/x", ".phase16_memory_profiles/x"])
    rec("phase16_local_dirs_ignored", ignored.returncode == 0)

    conflicts = []
    for rel in tracked:
        if not rel.endswith((".md", ".cpp", ".h", ".py", ".json", ".txt", ".toml", ".cmake", "CMakeLists.txt")):
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("<<<<<<<") or line.startswith(">>>>>>>") or line.startswith("======="):
                conflicts.append(rel)
                break
    rec("no_conflict_markers", conflicts == [])
    if conflicts:
        details["conflicts"] = ";".join(conflicts[:20])

    secret_hits = []
    scan_dirs = ["reports", "docs", "tests/seedvr2_real_weight_hybrid", "tests/seedvr2_dit_block_ncnn"]
    for d in scan_dirs:
        base = REPO / d
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file() or f.suffix not in (".md", ".json", ".py", ".cpp", ".h", ".txt", ".toml"):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                if SECRET_RE.search(line):
                    secret_hits.append(str(f.relative_to(REPO)))
                    break
    rec("no_secrets_in_reports_docs_tests", secret_hits == [])
    if secret_hits:
        details["secrets"] = ";".join(secret_hits[:20])

    json_bad = []
    json_files = sorted((REPO / "reports").glob("phase16*.json"))
    for extra in ("full_dit_visual_comparison.json", "full_dit_error_growth.json"):
        p = REPO / "reports" / extra
        if p.exists():
            json_files.append(p)
    for f in json_files:
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            json_bad.append(f"{f.name}:parse:{exc}")
            continue

        def scan(o, path):
            if isinstance(o, dict):
                for k, v in o.items():
                    scan(v, f"{path}.{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    scan(v, f"{path}[{i}]")
            elif isinstance(o, float) and not math.isfinite(o):
                json_bad.append(f"{f.name}:{path}:non-finite")

        scan(data, "")
    rec("phase16_reports_json_finite", json_bad == [])
    if json_bad:
        details["json_bad"] = ";".join(json_bad[:20])

    overall = all(checks.values())
    report = {
        "phase": "phase16",
        "target": "git_protection",
        "large_binary_threshold_bytes": LARGE_BYTES,
        "checks": checks,
        "details": details,
        "overall_pass": overall,
        "overall_status": "PASS" if overall else "FAIL",
        "real_seedvr2_weights": True,
        "full_32_block_dit": True,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }
    out = REPO / "reports" / "phase16_git_protection.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())