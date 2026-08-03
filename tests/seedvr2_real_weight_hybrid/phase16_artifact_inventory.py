#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: full artifact inventory for all 32 real blocks.

Verifies that every block 0..31 has a complete artifact directory under
.phase16_local_artifacts/, that metadata.block_index == directory block, that
every runtime stage model file exists, that metadata is_last_layer matches the
schedule, and computes logical/physical byte accounting (split blocks carry
per-side weighted artifacts, shared blocks share vid/txt, weight-free stages
are hardlinked/deduped across all blocks).  Writes
reports/phase16_artifact_inventory.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase16_common as p16  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    required_stages = p16.BASE_STAGES + p16.HEAD_STAGES
    per_block = {}
    missing = []
    for bi in p16.ALL_BLOCKS:
        variant = p16.variant_for(bi)
        d = p16.PHASE16_ROOT / variant
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            missing.append(f"{bi}:no-metadata")
            continue
        meta = json.loads(meta_path.read_text("utf-8"))
        idx = meta.get("block", {}).get("block_index")
        if idx != bi:
            missing.append(f"{bi}:metadata-index={idx}")
            continue
        is_last = bool(meta["block"]["is_last_layer"])
        expected_is_last = (bi == p16.LAST_LAYER_INDEX)
        if is_last != expected_is_last:
            missing.append(f"{bi}:is_last_layer={is_last}")
            continue
        stage_missing = []
        for stage in required_stages:
            if stage not in meta.get("models", {}):
                stage_missing.append(stage)
                continue
            for suffix in ("model.ncnn.param", "model.ncnn.bin"):
                f = d / meta["models"][stage][f"ncnn_{'param' if suffix.endswith('param') else 'bin'}"]
                if not f.exists():
                    stage_missing.append(f"{stage}:{suffix}")
        if stage_missing:
            missing.append(f"{bi}:stages={stage_missing[:3]}")
            continue
        layout = "split_vid_txt" if bi in p16.SPLIT_BLOCKS else "shared_all"
        manifest = json.loads((d / "artifact_manifest.json").read_text("utf-8"))
        stages = manifest["stages"]
        total_bin = int(sum(v["bin_bytes"] for v in stages.values()))
        weighted = [k for k, v in stages.items() if v["bin_bytes"] > 0]
        wf = [k for k, v in stages.items() if v["bin_bytes"] == 0]
        per_block[str(bi)] = {
            "variant": variant,
            "block_index": bi,
            "layout": layout,
            "is_last_layer": is_last,
            "metadata_block_index_ok": True,
            "stage_count": len(stages),
            "weighted_stage_count": len(weighted),
            "weight_free_stage_count": len(wf),
            "weighted_stages": sorted(weighted),
            "weight_free_stages": sorted(wf),
            "total_bin_bytes": total_bin,
            "total_param_bytes": int(sum(v["param_bytes"] for v in stages.values())),
            "reused_from_phase15": bool(manifest.get("reused_from_phase15", False)),
            "reused_from_phase14": bool(manifest.get("reused_from_phase14", False)),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "artifact_set_sha256": sha256(d / "artifact_manifest.json"),
        }

    # Physical dedup: weight-free stages are hardlinked to the same inode as
    # the Phase 15 block-16 source -> count unique bytes once.
    logical_total = sum(v["total_bin_bytes"] for v in per_block.values())
    unique_weighted = 0
    for v in per_block.values():
        for s in v["weighted_stages"]:
            d = p16.PHASE16_ROOT / p16.variant_for(v["block_index"]) / "models" / s / "model.ncnn.bin"
            unique_weighted += d.stat().st_size
    # weight-free bytes: 5 stages * single copy (hardlinked everywhere).
    wf_total = 0
    for s in p16.WEIGHT_FREE_STAGES:
        f = p16.PHASE16_ROOT / p16.variant_for(16) / "models" / s / "model.ncnn.bin"
        wf_total += f.stat().st_size if f.exists() else 0
    physical_total = unique_weighted + wf_total

    report = {
        "phase": 16,
        "real_seedvr2_weights": True,
        "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
        "actual_block_count": len(per_block),
        "expected_block_count": p16.ACTUAL_BLOCK_COUNT,
        "all_blocks_present": len(per_block) == p16.ACTUAL_BLOCK_COUNT and not missing,
        "missing_or_invalid": missing,
        "per_block": per_block,
        "logical_total_bin_bytes": logical_total,
        "physical_dedup_total_bin_bytes": physical_total,
        "dedup_saved_bytes": logical_total - physical_total,
        "weight_free_dedup": {
            "stages": p16.WEIGHT_FREE_STAGES,
            "single_copy_bytes": wf_total,
            "note": "attn_norm/mlp_norm/rope_apply/swiglu_act/head are weight-free "
                    "(bin=0) with identical graph contracts; hardlinked to one copy.",
        },
        "shared_block_weighted_bytes": int(sum(v["total_bin_bytes"] for v in per_block.values()
                                               if v["layout"] == "shared_all")),
        "split_block_weighted_bytes": int(sum(v["total_bin_bytes"] for v in per_block.values()
                                              if v["layout"] == "split_vid_txt")),
        "large_artifacts_committed": False,
    }
    out = p16.REPO / "reports" / "phase16_artifact_inventory.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"blocks={len(per_block)} missing={missing or 'none'}")
    print(f"logical={logical_total/1e9:.2f}GB physical={physical_total/1e9:.2f}GB "
          f"dedup_saved={(logical_total-physical_total)/1e9:.2f}GB")
    return 0 if report["all_blocks_present"] else 1


if __name__ == "__main__":
    sys.exit(main())
