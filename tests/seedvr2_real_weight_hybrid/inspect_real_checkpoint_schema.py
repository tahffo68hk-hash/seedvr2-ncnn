#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14 checkpoint schema audit (read-only, safe load).

Loads the verified official seedvr2_ema_3b.pth and records the schema WITHOUT
executing arbitrary checkpoint code: prefers torch.load(weights_only=True);
if that fails, records the exact exception and reuses the OFFICIAL SeedVR2
loading method (torch.load(map_location="cpu", mmap=True)) only after hash and
source confirmation (caller-verified). Output is a typed JSON report with
sanitized paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def describe(t):
    return {"shape": list(t.shape), "dtype": str(t.dtype), "numel": int(t.numel())}


def scan(state, report):
    keys = list(state.keys())
    report["top_level_keys"] = keys[:50]
    report["top_level_type"] = type(state).__name__
    # flatten nested dicts (e.g. {"model": {...}})
    flat = {}
    def walk(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(v, key + ".")
            else:
                flat[key] = v
    walk(state)
    report["flat_key_count"] = len(flat)
    report["tensor_count"] = sum(1 for v in flat.values() if hasattr(v, "shape"))
    report["dtype_distribution"] = dict(Counter(str(v.dtype) for v in flat.values() if hasattr(v, "shape")))
    report["total_numel"] = int(sum(v.numel() for v in flat.values() if hasattr(v, "shape")))

    # block detection: find keys matching <prefix>blocks.<int>.
    import re
    blocks = set()
    block_prefixes = set()
    for k in flat:
        m = re.search(r"blocks\.(\d+)\.", k)
        if m:
            blocks.add(int(m.group(1)))
            block_prefixes.add(k[: k.index(f"blocks.{m.group(1)}.") + len(f"blocks.{m.group(1)}.")])
    report["block_indices"] = sorted(blocks)
    report["block_count"] = len(blocks)
    report["block_prefixes_sample"] = sorted(block_prefixes)[:10]

    # samples of one middle block key set
    if blocks:
        mid = sorted(blocks)[len(blocks) // 2]
        report["sample_block_index"] = mid
        keys_mid = sorted(k for k in flat if re.search(rf"blocks\.{mid}\.", k))
        report["sample_block_key_count"] = len(keys_mid)
        report["sample_block_keys"] = keys_mid
        # parameter count of the middle block
        numel_mid = sum(flat[k].numel() for k in keys_mid if hasattr(flat[k], "shape"))
        report["sample_block_numel"] = int(numel_mid)
        # representative shapes
        rep = {}
        for probe in [
            "attn.proj_qkv.vid.weight", "attn.proj_qkv.txt.weight",
            "attn.norm_q.vid.weight", "attn.norm_k.vid.weight",
            "attn.proj_out.vid.weight", "mlp.vid.proj_in.weight",
            "mlp.vid.proj_in_gate.weight", "mlp.vid.proj_out.weight",
        ]:
            for k in keys_mid:
                if k.endswith(probe):
                    rep[probe] = describe(flat[k])
                    break
        report["sample_block_shapes"] = rep

        # q/k RMSNorm real values
        qk = {}
        for probe in ["attn.norm_q.vid.weight", "attn.norm_q.txt.weight",
                      "attn.norm_k.vid.weight", "attn.norm_k.txt.weight"]:
            for k in keys_mid:
                if k.endswith(probe):
                    t = flat[k]
                    qk[probe] = {
                        "shape": list(t.shape), "dtype": str(t.dtype),
                        "all_ones": bool((t == 1).all().item()),
                        "min": float(t.min().item()), "max": float(t.max().item()),
                        "unique_count": int(t.unique().numel()),
                    }
                    break
        report["sample_block_qk_norm"] = qk
    return flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--report", default="reports/phase14_checkpoint_schema_probe.json")
    ap.add_argument("--expected-sha256", default="6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    ckpt = Path(args.checkpoint) if args.checkpoint else repo / ".phase14_local_checkpoint" / "seedvr2_ema_3b.pth"
    report_path = Path(args.report) if Path(args.report).is_absolute() else repo / args.report

    import hashlib
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    report = {
        "phase": 14,
        "sha256": sha,
        "sha256_matches_expected": sha.lower() == args.expected_sha256.lower(),
        "bytes": ckpt.stat().st_size,
        "loading_methods_tried": [],
        "weights_only_succeeded": None,
        "error": None,
    }
    if not report["sha256_matches_expected"]:
        report["status"] = "BLOCKED_CHECKPOINT_INTEGRITY"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 4

    import torch
    state = None
    # 1) weights_only=True
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        report["loading_methods_tried"].append("torch.load(weights_only=True)")
        report["weights_only_succeeded"] = True
    except Exception as e:
        report["loading_methods_tried"].append(f"torch.load(weights_only=True) failed: {type(e).__name__}: {e}")
        report["weights_only_succeeded"] = False
        report["weights_only_error"] = f"{type(e).__name__}: {e}"
        # 2) official SeedVR2 loading method (infer.py): map_location cpu + mmap
        try:
            state = torch.load(str(ckpt), map_location="cpu", mmap=True)
            report["loading_methods_tried"].append("torch.load(map_location=cpu, mmap=True) [official infer.py method]")
        except Exception as e2:
            report["error"] = f"{type(e2).__name__}: {e2}"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 3

    if isinstance(state, dict):
        report["status"] = "SCHEMA_OK"
        flat = scan(state, report)
    else:
        report["status"] = "SCHEMA_UNEXPECTED"
        report["top_level_type"] = type(state).__name__
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
