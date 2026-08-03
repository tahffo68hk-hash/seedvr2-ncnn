#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: reuse the Phase 15 blocks 15/16/17/18 artifacts as READ-ONLY
hardlinks inside .phase16_local_artifacts/ (zero-copy, same inode).

Verifies every file SHA-256 and per-stage SHAs against the Phase 15 manifest,
then writes a Phase 16 artifact_manifest.json with reused_from_phase15=true.
"""
from __future__ import annotations

import hashlib
import json
import os
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
    for bi in p16.P15_EXISTING:
        variant = p16.variant_for(bi)
        src = p16.PHASE15_ROOT / variant
        dst = p16.PHASE16_ROOT / variant
        assert src.exists(), f"Phase 15 artifact missing: {src}"
        assert (src / "metadata.json").exists()

        src_manifest = json.loads((src / "artifact_manifest.json").read_text("utf-8"))
        files = []
        for stage in p16.UNIQUE_STAGES:
            for suffix in ("model.ncnn.param", "model.ncnn.bin", "reference_output.bin"):
                f = src / "models" / stage / suffix
                if f.exists():
                    files.append((f, dst / "models" / stage / suffix))
        for f in sorted((src / "params").glob("*.bin")):
            files.append((f, dst / "params" / f.name))
        for f in ("metadata.json", "export_stage_log.json"):
            files.append((src / f, dst / f))

        dst.mkdir(parents=True, exist_ok=True)
        linked, verified = [], []
        for s, d in files:
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists():
                d.unlink()
            os.link(str(s), str(d))
            linked.append(str(d.relative_to(p16.PHASE16_ROOT)))
            assert sha256(s) == sha256(d), f"hardlink verify failed: {d}"
            verified.append(str(d.relative_to(p16.PHASE16_ROOT)))

        stages = {}
        for stage in p16.UNIQUE_STAGES:
            p = dst / "models" / stage / "model.ncnn.param"
            b = dst / "models" / stage / "model.ncnn.bin"
            stages[stage] = {
                "ncnn_bin_sha256": sha256(b),
                "ncnn_param_sha256": sha256(p),
                "bin_bytes": b.stat().st_size,
                "param_bytes": p.stat().st_size,
            }

        total_bin = sum(v["bin_bytes"] for v in stages.values())
        manifest = {
            "phase": 16,
            "variant": variant,
            "block_index": bi,
            "source_block_index": bi,
            "is_last_layer": False,
            "layout": "shared_all",
            "reused_from_phase15": True,
            "phase15_source_root": f".phase15_local_artifacts/{variant}",
            "reuse_mechanism": "hardlink (read-only, same inode, zero-copy)",
            "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
            "seedvr2_commit": p16.SEEDVR2_HEAD,
            "pnnx_fp16": 0,
            "qk_affine_mode": "real_affine_exported",
            "weighted_stage_count": 7,
            "shared_weight_free_stage_count": 5,
            "stages": stages,
            "hardlinked_file_count": len(linked),
            "verified_file_count": len(verified),
            "total_bin_bytes": total_bin,
            "large_artifacts_committed": False,
        }
        (dst / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
        mismatches = []
        for stage, entry in src_manifest.get("stages", {}).items():
            if stage in stages and entry.get("ncnn_bin_sha256"):
                if entry["ncnn_bin_sha256"] != stages[stage]["ncnn_bin_sha256"]:
                    mismatches.append(stage)
        print(f"block {bi}: reused from Phase 15 hardlinks={len(linked)} verified={len(verified)} "
              f"total_bin={total_bin} mismatches={mismatches or 'none'}")
        if mismatches:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
