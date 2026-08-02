"""Phase 13: artifact inventory validation (x3 stability runner).

Re-walks every real-width stage artifact, recomputes SHA256 / sizes, checks
non-empty files, pnnx fp16=0 metadata, param+bin pairing and manifest
consistency, and verifies the committed manifest matches the local artifacts.
Writes `reports/realwidth_artifact_inventory_validation.json`.

0-byte ncnn bins are legitimate in two cases: weight-free stages
(`parameter_count == 0`, e.g. head/rope_apply) and stages whose state_dict
params were folded away by pnnx because every mapped weight is exactly 1.0
(`pnnx_folded_identity_weight == true`, e.g. the real qk-norm RMSNorm
weights).  The runner independently verifies each folded marker against the
local state dict so a mis-exported stage cannot hide behind the marker.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _realwidth_common as rw  # noqa: E402

CONTRACT = json.load(io.open(HERE / "seedvr2_block_weight_mapping.json", "r", encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_weights_for_stage(stage: str) -> list:
    return [e["state_dict_key"] for e in CONTRACT["entries"]
            if e["target_stage"] == stage and e["kind"] == "weight"]


def stage_weights_all_ones(stage: str, sd: dict) -> bool:
    keys = source_weights_for_stage(stage)
    if not keys:
        return False
    return all(bool(torch.all(sd[k] == 1.0).item()) for k in keys)


def run_once(sd: dict) -> dict:
    manifest_path = rw.REPO / "reports" / "realwidth_artifact_inventory.json"
    manifest = json.load(io.open(manifest_path, "r", encoding="utf-8"))
    meta = json.load(io.open(rw.LOCAL_ROOT / "metadata.json", "r", encoding="utf-8"))
    problems = []
    for e in manifest["entries"]:
        f = rw.REPO / e["local_relative_path"]
        if not f.exists():
            problems.append(f"missing {e['logical_name']}")
            continue
        folded = bool(e.get("pnnx_folded_identity_weight", False))
        # empty bin is only legit for weight-free stages or pnnx-folded stages
        if f.stat().st_size == 0 and e.get("parameter_count", 0) > 0 and not folded:
            problems.append(f"empty {e['logical_name']}")
        if folded:
            if f.stat().st_size != 0:
                problems.append(f"folded marker on non-empty bin {e['logical_name']}")
            if not stage_weights_all_ones(e["stage"], sd):
                problems.append(f"folded marker without identity weights {e['logical_name']}")
        if f.stat().st_size != e["bytes"]:
            problems.append(f"size mismatch {e['logical_name']}")
        if sha256(f) != e["sha256"]:
            problems.append(f"sha256 mismatch {e['logical_name']}")
    # every base stage must have param + bin
    for stage in rw.MODEL_STAGES:
        if not (rw.MODELS / stage / "model.ncnn.param").exists():
            problems.append(f"missing param {stage}")
        if not (rw.MODELS / stage / "model.ncnn.bin").exists():
            problems.append(f"missing bin {stage}")
    ok = (not problems and meta["dtype"] == "fp32" and meta["pnnx"]["fp16"] == 0
          and meta["block"]["hidden_dim"] == 2560 and meta["block"]["heads"] == 20)
    return {"ok": ok, "problems": problems[:40], "entry_count": len(manifest["entries"]),
            "total_bytes": manifest["total_bytes"], "fp16": meta["pnnx"]["fp16"],
            "hidden_dim": meta["block"]["hidden_dim"], "heads": meta["block"]["heads"]}


def main() -> int:
    report = {
        "phase": "phase13",
        "target": "realwidth_artifact_inventory_tests",
        "repetitions": 3,
        "runs": [],
        "overall_pass": True,
        "overall_status": "PASS",
        "real_seedvr2_weights": False,
        "realwidth_block": True,
        "full_32_block_dit": False,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }
    sd = torch.load(rw.STATE_DICT, map_location="cpu", weights_only=True)
    first = None
    stable = True
    for i in range(3):
        r = run_once(sd)
        r["repetition"] = i + 1
        key = json.dumps({k: v for k, v in r.items() if k != "repetition"}, sort_keys=True)
        if first is None:
            first = key
        else:
            stable = stable and key == first
        report["runs"].append(r)
        report["overall_pass"] = report["overall_pass"] and r["ok"]
        print(f"run {i+1}: ok={r['ok']} problems={len(r['problems'])}")
    report["determinism_stable_across_runs"] = stable
    report["overall_pass"] = report["overall_pass"] and stable
    report["overall_status"] = "PASS" if report["overall_pass"] else "FAIL"
    out = rw.REPO / "reports" / "realwidth_artifact_inventory_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    print("overall_pass:", report["overall_pass"], "stable:", stable)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
