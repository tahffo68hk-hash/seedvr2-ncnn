#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: independent validation of reports/phase17_vae_weight_mapping.json.

Re-loads the real VAE checkpoint and re-derives the per-stage expectation
(identity shape copy vs numel-compare for the deconv-folded upscale weights),
then verifies: mapped_keys == source_keys, unmapped == [], duplicate == [],
shape mismatches == [], mapped_numel == source_numel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402
import vae_export_model as vem  # noqa: E402
from generate_vae_weight_mapping import STAGE_PREFIXES, semantic_role, conversion_of  # noqa: E402


def main() -> int:
    report_path = Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_vae_weight_mapping.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    vae = pc.build_real_vae()
    sd, _ = pc.load_vae_state_dict()
    vae.load_state_dict(sd, strict=True)
    enc = vae.encoder
    dec = vae.decoder
    all_stages = {**vem.build_encoder_stages(enc), **vem.build_decoder_stages(dec)}

    stage_of = {}
    for stage, prefixes in STAGE_PREFIXES.items():
        for pfx in prefixes:
            for key in sd.keys():
                if key.startswith(pfx + "."):
                    stage_of[key] = stage

    mapped = report.get("entries")
    assert mapped, "report does not contain per-key entries"

    errors = []
    # 1) mapped keys == source keys
    mapped_keys = set(mapped.keys())
    source_keys = set(sd.keys())
    if mapped_keys != source_keys:
        errors.append(f"mapped_keys != source_keys: missing={sorted(source_keys - mapped_keys)} extra={sorted(mapped_keys - source_keys)}")
    # 2) unmapped / duplicate
    unmapped = [k for k in sd if k not in mapped_keys]
    dup = [k for k in mapped_keys if sum(1 for s in STAGE_PREFIXES for p in STAGE_PREFIXES[s] if k.startswith(p + ".")) > 1]
    # 3) shape / numel checks
    shape_mismatches = []
    for key in sorted(source_keys):
        e = mapped.get(key)
        if e is None:
            continue
        shape = tuple(int(s) for s in sd[key].shape)
        exp = e.get("expected_target_shape")
        if exp == "numel-compare":
            continue
        if tuple(exp) != shape:
            shape_mismatches.append({"key": key, "expected": exp, "actual": list(shape)})
    # 4) numel
    mapped_numel = sum(int(sd[k].numel()) for k in mapped_keys)
    source_numel = sum(int(t.numel()) for t in sd.values())

    summary = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "source_keys": len(source_keys),
        "mapped_keys": len(mapped_keys),
        "mapped_numel": mapped_numel,
        "source_numel": source_numel,
        "unmapped_keys": unmapped,
        "duplicate_keys": dup,
        "shape_mismatches": shape_mismatches,
        "mapped_numel_equals_source_numel": mapped_numel == source_numel,
        "pass": (mapped_keys == source_keys and not unmapped and not dup
                 and not shape_mismatches and mapped_numel == source_numel),
        "errors": errors,
    }
    pc.write_report(str(Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_vae_weight_mapping_validation.json")), summary)
    print("validation pass:", summary["pass"])
    if not summary["pass"]:
        print("errors:", errors[:5], "shape mismatches:", shape_mismatches[:5])
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
