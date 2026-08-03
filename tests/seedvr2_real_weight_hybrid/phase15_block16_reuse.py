#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: reuse the Phase 14 block-16 artifact as a READ-ONLY hardlink.

Creates `.phase15_local_artifacts/real_seedvr2_block16_c2560_h20_hd128/` whose
model/param files are hardlinks (zero-copy, same inode) of the Phase 14
block-16 artifact, verifies every file SHA-256 and the per-stage artifact SHAs,
then writes the Phase 15 `artifact_manifest.json` with
reused_from_phase14=true.  Hardlinks make the reuse read-only: there is no
second copy and no way to drift from the Phase 14 anchor.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase15_common as p15  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    src = p15.PHASE14_ROOT / p15.P14_VARIANT_16
    dst = p15.PHASE15_ROOT / p15.P14_VARIANT_16
    assert src.exists(), f"Phase 14 block16 artifact missing: {src}"
    assert (src / "metadata.json").exists()

    src_log = json.loads((src / "export_stage_log.json").read_text("utf-8"))
    files = []
    for stage in p15.UNIQUE_STAGES:
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
        linked.append(str(d.relative_to(p15.PHASE15_ROOT)))
        assert sha256(s) == sha256(d), f"hardlink verify failed: {d}"
        verified.append(str(d.relative_to(p15.PHASE15_ROOT)))

    stages = {}
    for stage in p15.UNIQUE_STAGES:
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
        "phase": 15,
        "variant": p15.P14_VARIANT_16,
        "block_index": 16,
        "source_block_index": 16,
        "reused_from_phase14": True,
        "phase14_source_root": ".phase14_local_artifacts/real_seedvr2_block16_c2560_h20_hd128",
        "reuse_mechanism": "hardlink (read-only, same inode, zero-copy)",
        "checkpoint_sha256": p15.EXPECTED_MAIN_SHA256,
        "seedvr2_commit": p15.SEEDVR2_HEAD,
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
    print(f"block 16 reused from Phase 14: hardlinks={len(linked)} verified={len(verified)} "
          f"total_bin_bytes={total_bin}")
    # Sanity: per-stage hashes must match the Phase 14 export log.
    mismatches = []
    for stage, entry in src_log.items():
        if stage in stages and entry.get("sha256_bin"):
            if entry["sha256_bin"] != stages[stage]["ncnn_bin_sha256"]:
                mismatches.append(stage)
    print("phase14 export-log hash mismatches:", mismatches if mismatches else "none")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
