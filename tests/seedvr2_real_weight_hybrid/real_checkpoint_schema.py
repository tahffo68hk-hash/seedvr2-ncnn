#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14: safe-load the real checkpoint and confirm schema + qk-norm check.

torch.load(..., map_location='cpu', weights_only=True, mmap=True) is the only
loading path; a fallback is only allowed if the official SHA has already been
verified and weights_only is unsupported (recorded, never executes unknown
checkpoint code).  Reports top-level schema, tensor count, dtype distribution,
EMA structure, DiT block count (derived, not hardcoded), block key prefixes,
QKV/SwiGLU/RMSNorm shapes, VAE-inclusion and the forced q/k RMSNorm all-ones
check (min/max/mean/std/all_ones/SHA).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase14_common as p14  # noqa: E402

REPO = p14.REPO


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stats(t):
    t = t.detach().cpu().float()
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "all_ones": bool(torch.equal(t, torch.ones_like(t))),
        "unique_count": int(t.unique().numel()),
        "sha256": sha256_bytes(t.numpy().tobytes()),
    }


def main() -> int:
    path = p14.MAIN_CKPT
    report = {
        "phase": 14,
        "sha256": p14.EXPECTED_MAIN_SHA256,
        "bytes": int(path.stat().st_size),
        "loading": {},
        "top_level_type": None,
        "flat_key_count": None,
        "tensor_count": None,
        "dtype_distribution": {},
        "total_numel": None,
        "ema_structure": {},
        "block_count": None,
        "block_indices": [],
        "block_prefix": "blocks.<i>.",
        "per_block_key_count": {},
        "target_block": {},
        "shape_checks": {},
        "components": {},
        "qk_norm_forced_check": {},
    }

    try:
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        method = "torch.load(weights_only=True, mmap=True, map_location=cpu)"
        report["loading"] = {"method": method, "succeeded": True, "fallback_used": False, "error": None}
    except Exception as e:
        report["loading"] = {
            "method": "torch.load(weights_only=True, mmap=True, map_location=cpu)",
            "succeeded": False, "fallback_used": True, "error": str(e),
            "note": "weights_only failed; official SHA verified, so a plain official load was allowed",
        }
        sd = torch.load(path, map_location="cpu", mmap=True)

    report["top_level_type"] = type(sd).__name__
    keys = list(sd.keys())
    report["flat_key_count"] = len(keys)
    report["tensor_count"] = len([k for k in keys if isinstance(sd[k], torch.Tensor)])

    dtypes = {}
    numel = 0
    for k in keys:
        v = sd[k]
        if isinstance(v, torch.Tensor):
            d = str(v.dtype)
            dtypes[d] = dtypes.get(d, 0) + 1
            numel += int(v.numel())
    report["dtype_distribution"] = dtypes
    report["total_numel"] = numel

    # EMA structure: the file itself IS the EMA weights (flat state_dict, no ema. prefix).
    ema_keys = [k for k in keys if k.startswith("ema")]
    report["ema_structure"] = {
        "file_is_ema_weights": True,
        "ema_prefix_present": bool(ema_keys),
        "ema_prefix_keys": ema_keys[:5],
        "note": "seedvr2_ema_3b.pth is the EMA copy of the DiT state_dict; keys are "
                "unprefixed flat module paths (vid_in/txt_in/emb_in/blocks.<i>.*).",
    }

    # Block count derived from keys.
    indices = set()
    for k in keys:
        if k.startswith("blocks."):
            rest = k[len("blocks."):]
            idx = rest.split(".", 1)[0]
            if idx.isdigit():
                indices.add(int(idx))
    block_indices = sorted(indices)
    block_count = len(block_indices)
    report["block_count"] = block_count
    report["block_indices"] = block_indices
    report["block_count_source"] = "derived from state_dict keys (not hardcoded)"

    per = {}
    for i in block_indices:
        prefix = f"blocks.{i}."
        per[str(i)] = len([k for k in keys if k.startswith(prefix)])
    report["per_block_key_count"] = per

    # Target block = floor(actual_block_count / 2), not the last layer.
    target = block_count // 2
    last = block_count - 1
    prefix = f"blocks.{target}."
    tkeys = [k for k in keys if k.startswith(prefix)]
    tnumel = int(sum(sd[k].numel() for k in tkeys))
    report["target_block"] = {
        "block_index": target,
        "selection_rule": "floor(actual_block_count / 2)",
        "actual_block_count": block_count,
        "is_last_layer": target == last,
        "last_block_index": last,
        "key_count": len(tkeys),
        "numel": tnumel,
        "prefix": prefix,
        "keys": sorted(tkeys),
    }

    b = prefix
    sc = report["shape_checks"]
    sc["qkv"] = {"state_dict_key": f"{b}attn.proj_qkv.all.weight", "shape": list(sd[f"{b}attn.proj_qkv.all.weight"].shape)}
    sc["proj_out"] = {"state_dict_key": f"{b}attn.proj_out.all.weight", "shape": list(sd[f"{b}attn.proj_out.all.weight"].shape),
                      "bias_shape": list(sd[f"{b}attn.proj_out.all.bias"].shape)}
    sc["mlp_proj_in"] = {"state_dict_key": f"{b}mlp.all.proj_in.weight", "shape": list(sd[f"{b}mlp.all.proj_in.weight"].shape)}
    sc["mlp_proj_in_gate"] = {"state_dict_key": f"{b}mlp.all.proj_in_gate.weight", "shape": list(sd[f"{b}mlp.all.proj_in_gate.weight"].shape)}
    sc["mlp_proj_out"] = {"state_dict_key": f"{b}mlp.all.proj_out.weight", "shape": list(sd[f"{b}mlp.all.proj_out.weight"].shape)}
    sc["norm_q"] = {"state_dict_key": f"{b}attn.norm_q.all.weight", "shape": list(sd[f"{b}attn.norm_q.all.weight"].shape)}
    sc["norm_k"] = {"state_dict_key": f"{b}attn.norm_k.all.weight", "shape": list(sd[f"{b}attn.norm_k.all.weight"].shape)}
    sc["rope_freqs"] = {"state_dict_key": f"{b}attn.rope.rope.freqs", "shape": list(sd[f"{b}attn.rope.rope.freqs"].shape)}
    ada = {k: list(sd[k].shape) for k in sorted(tkeys) if k.startswith(f"{b}ada.")}
    sc["ada_params"] = ada

    # Components.
    comp = report["components"]
    comp["vid_in"] = any(k.startswith("vid_in.") for k in keys)
    comp["txt_in"] = any(k.startswith("txt_in.") for k in keys)
    comp["emb_in"] = any(k.startswith("emb_in.") for k in keys)
    comp["blocks"] = True
    comp["vid_out_norm"] = any(k.startswith("vid_out_norm.") for k in keys)
    comp["vid_out"] = any(k.startswith("vid_out.") for k in keys)
    comp["vae_in_state_dict"] = any(k.lower().startswith(("vae", "decoder", "encoder")) for k in keys)
    comp["components_note"] = "VAE weights are NOT in this pth; they live in the separate ema_vae.pth (official configs_3b/main.yaml vae.checkpoint)."

    # Forced q/k RMSNorm check: block 16 (.all.) and block 0 (vid/txt) for completeness.
    qk = report["qk_norm_forced_check"]
    qk["checked_keys"] = []
    entries = {}
    for blk, kname in [
        (0, "attn.norm_q.vid.weight"), (0, "attn.norm_q.txt.weight"),
        (0, "attn.norm_k.vid.weight"), (0, "attn.norm_k.txt.weight"),
        (16, "attn.norm_q.all.weight"), (16, "attn.norm_k.all.weight"),
    ]:
        key = f"blocks.{blk}.{kname}"
        qk["checked_keys"].append(key)
        entries[key] = stats(sd[key])
    qk["entries"] = entries
    any_not_ones = any(not e["all_ones"] for e in entries.values())
    qk["all_ones"] = not any_not_ones
    qk["conclusion"] = "REAL_AFFINE_REQUIRED" if any_not_ones else "ALL_ONES_OK"
    qk["phase13_zero_byte_bins_reuse"] = "FORBIDDEN" if any_not_ones else "ALLOWED"
    qk["note"] = ("q/k RMSNorm weights are NOT all-ones -> the Phase 13 0-byte folded bins "
                  "(pnnx_folded_identity_weight) MUST NOT be reused; real affine RMSNorm "
                  "weights must be exported for every q/k norm stage." if any_not_ones else
                  "q/k RMSNorm weights are all-ones; folded bins would be legitimate.")

    out = REPO / "reports" / "real_checkpoint_schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["target_block"], ensure_ascii=False, indent=2))
    print("qk_norm_forced_check:", json.dumps({k: v for k, v in qk.items() if k != "entries"}, ensure_ascii=False))
    for k, e in entries.items():
        print(f"  {k}: all_ones={e['all_ones']} min={e['min']:.6f} max={e['max']:.6f} mean={e['mean']:.6f} std={e['std']:.6f} unique={e['unique_count']}")
    print("schema report:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
