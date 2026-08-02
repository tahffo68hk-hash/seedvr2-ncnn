"""Phase 13: generate the small committed artifact inventory manifest.

Walks the git-ignored real-width local artifacts and records only metadata
(logical name, local relative path, bytes, SHA256, parameter count) for every
stage artifact - never file contents.  The manifest is committed
(`reports/realwidth_artifact_inventory.json`) and verified by the C++ inventory
tests and by `validate_realwidth_artifact_inventory.py`.

`pnnx_folded_identity_weight` marks 0-byte ncnn bins whose state_dict params
were folded away by pnnx because every mapped weight is exactly 1.0 (identity
affine, e.g. the real SeedVR2 qk-norm RMSNorm weights).  Those empty bins are
legitimate; every other stage with state_dict params must produce a non-empty
ncnn bin.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _realwidth_common as rw  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parameter_count_for_stage(contract_path: Path, stage: str) -> int:
    c = json.load(io.open(contract_path, "r", encoding="utf-8"))
    total = 0
    for e in c["entries"]:
        if e["target_stage"] == stage and e["conversion"] in ("pnnx_managed", "identity"):
            total += int(e["numel"])
    return total


def main() -> int:
    contract_path = HERE / "seedvr2_block_weight_mapping.json"
    entries = []
    total_bytes = 0
    for stage in rw.MODEL_STAGES:
        d = rw.MODELS / stage
        if not (d / "model.ncnn.param").exists() or not (d / "model.ncnn.bin").exists():
            print(f"missing stage dir: {stage}")
            continue
        for suffix, kind in ((".param", "param"), (".bin", "bin")):
            f = d / ("model.ncnn" + suffix)
            b = f.stat().st_size
            total_bytes += b
            param_count = parameter_count_for_stage(contract_path, stage) if kind == "bin" else 0
            folded = kind == "bin" and int(b) == 0 and param_count > 0
            entries.append({
                "logical_name": f"{stage}.{kind}",
                "stage": stage,
                "local_relative_path": str(f.relative_to(rw.REPO)).replace("\\", "/"),
                "bytes": int(b),
                "sha256": sha256(f),
                "parameter_count": param_count,
                "pnnx_folded_identity_weight": folded,
            })
    manifest = {
        "phase": "phase13",
        "variant": rw.VARIANT,
        "entry_count": len(entries),
        "total_bytes": int(total_bytes),
        "fp16": 0,
        "entries": entries,
    }
    out = rw.REPO / "reports" / "realwidth_artifact_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(json.dumps(manifest, indent=2))
    print("entries:", len(entries), "total_bytes:", total_bytes)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
