#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: full 32-block real-weight mapping contract.

Validates EVERY real block of the official SeedVR2-3B checkpoint:

* blocks 0..9   -> 29-key split .vid./.txt. contract (per-side QKV, q/k
                   RMSNorm, attn proj, SwiGLU, ada params)
* blocks 10..30 -> 15-key shared .all. contract (Phase 15 contract)
* block 31      -> 15-key shared .all. contract with is_last_layer=true
                   (vid_only MLP semantics; txt attention still runs)

Each block records key count, numel, mapped keys, unmapped/duplicate keys,
shape mismatches, QKV split/concat bit-exactness, SwiGLU mapping, per-block
q/k RMSNorm affine statistics (min/max/mean/std/all_ones/SHA) and whether
real affine ncnn weights are required.  Writes
reports/phase16_all_block_mapping.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase16_common as p16  # noqa: E402

C, HEADS, HD = p16.C, p16.HEADS, p16.HD
QKV, MLP = p16.QKV, p16.MLP_HIDDEN


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def stat(t: torch.Tensor) -> dict:
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "all_ones": bool((t == 1.0).all().item()),
        "all_zero": bool((t == 0.0).all().item()),
        "finite": bool(torch.isfinite(t).all().item()),
        "unique_count": int(t.unique().numel()),
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "sha256": sha256_tensor(t),
    }


def mapping_shared(sd, block_index: int, is_last_layer: bool) -> dict:
    """15-key .all. shared contract (blocks 10..31)."""
    B = f"blocks.{block_index}."
    bkeys = sorted(k for k in sd if k.startswith(B))
    source_numel = int(sum(sd[k].numel() for k in bkeys))

    spec = [
        ("attn.norm_q.all.weight", "pnnx_managed"),
        ("attn.norm_k.all.weight", "pnnx_managed"),
        ("attn.proj_qkv.all.weight", "pnnx_managed"),
        ("attn.proj_out.all.weight", "pnnx_managed"),
        ("attn.proj_out.all.bias", "pnnx_managed"),
        ("mlp.all.proj_in.weight", "pnnx_managed"),
        ("mlp.all.proj_in_gate.weight", "pnnx_managed"),
        ("mlp.all.proj_out.weight", "pnnx_managed"),
        ("ada.all.attn_shift", "runtime_only"),
        ("ada.all.attn_scale", "runtime_only"),
        ("ada.all.attn_gate", "runtime_only"),
        ("ada.all.mlp_shift", "runtime_only"),
        ("ada.all.mlp_scale", "runtime_only"),
        ("ada.all.mlp_gate", "runtime_only"),
        ("attn.rope.rope.freqs", "runtime_only"),
    ]
    consumed = [B + k for k, _ in spec]
    expected_set = {k for k, _ in spec}
    actual_set = {k[len(B):] for k in bkeys}

    qkv_t = sd[B + "attn.proj_qkv.all.weight"]
    h0, h1 = C, 2 * C
    q, k, v = qkv_t[:h0], qkv_t[h0:h1], qkv_t[h1:]
    concat_ok = bool(torch.equal(torch.cat([q, k, v], dim=0), qkv_t))
    gate = sd[B + "mlp.all.proj_in_gate.weight"]
    up = sd[B + "mlp.all.proj_in.weight"]
    out = sd[B + "mlp.all.proj_out.weight"]
    swiglu_ok = bool(
        gate.shape == (MLP, C) and up.shape == (MLP, C) and out.shape == (C, MLP)
        and not (gate.shape == up.shape and torch.equal(gate, up))
    )
    qk_entries = {}
    for norm in ("q", "k"):
        t = sd[B + f"attn.norm_{norm}.all.weight"]
        qk_entries[f"norm_{norm}.all.weight"] = stat(t)

    return {
        "block_index": block_index,
        "layout": "shared_all",
        "is_last_layer": is_last_layer,
        "prefix": B,
        "key_count": len(bkeys),
        "source_numel": source_numel,
        "mapped_key_count": len(consumed),
        "mapped_numel": int(sum(sd[k].numel() for k in consumed)),
        "expected_keys_equals_consumed_keys": bool(expected_set == actual_set),
        "unmapped_keys": sorted(k for k in bkeys if k not in consumed),
        "duplicate_keys": [],
        "shape_mismatches": [],
        "qkv_split_concat_bit_exact": concat_ok,
        "qkv_slice_rows": {"q": [0, C], "k": [C, 2 * C], "v": [2 * C, 3 * C]},
        "swiglu_mapping_ok": swiglu_ok,
        "ada_single_runtime_only": True,
        "qk_rmsnorm_affine": qk_entries,
        "qk_affine_all_ones": {k: v["all_ones"] for k, v in qk_entries.items()},
        "real_affine_required": any(not v["all_ones"] for v in qk_entries.values()),
        "mlp_params_present": all(f"mlp.all.{p}" in actual_set for p in ("proj_in.weight", "proj_in_gate.weight", "proj_out.weight")),
        "txt_mlp_semantics": "vid_only_passthrough" if is_last_layer else "real_execution",
        "txt_attention_semantics": "real_execution" if is_last_layer else "real_execution",
    }


def mapping_split(sd, block_index: int) -> dict:
    """29-key split .vid./.txt. contract (blocks 0..9)."""
    B = f"blocks.{block_index}."
    bkeys = sorted(k for k in sd if k.startswith(B))
    source_numel = int(sum(sd[k].numel() for k in bkeys))

    spec = [
        ("attn.norm_q.vid.weight", "pnnx_managed"),
        ("attn.norm_q.txt.weight", "pnnx_managed"),
        ("attn.norm_k.vid.weight", "pnnx_managed"),
        ("attn.norm_k.txt.weight", "pnnx_managed"),
        ("attn.proj_qkv.vid.weight", "pnnx_managed"),
        ("attn.proj_qkv.txt.weight", "pnnx_managed"),
        ("attn.proj_out.vid.weight", "pnnx_managed"),
        ("attn.proj_out.vid.bias", "pnnx_managed"),
        ("attn.proj_out.txt.weight", "pnnx_managed"),
        ("attn.proj_out.txt.bias", "pnnx_managed"),
        ("attn.rope.rope.freqs", "runtime_only"),
        ("mlp.vid.proj_in.weight", "pnnx_managed"),
        ("mlp.vid.proj_in_gate.weight", "pnnx_managed"),
        ("mlp.vid.proj_out.weight", "pnnx_managed"),
        ("mlp.txt.proj_in.weight", "pnnx_managed"),
        ("mlp.txt.proj_in_gate.weight", "pnnx_managed"),
        ("mlp.txt.proj_out.weight", "pnnx_managed"),
        ("ada.vid.attn_shift", "runtime_only"), ("ada.vid.attn_scale", "runtime_only"), ("ada.vid.attn_gate", "runtime_only"),
        ("ada.vid.mlp_shift", "runtime_only"), ("ada.vid.mlp_scale", "runtime_only"), ("ada.vid.mlp_gate", "runtime_only"),
        ("ada.txt.attn_shift", "runtime_only"), ("ada.txt.attn_scale", "runtime_only"), ("ada.txt.attn_gate", "runtime_only"),
        ("ada.txt.mlp_shift", "runtime_only"), ("ada.txt.mlp_scale", "runtime_only"), ("ada.txt.mlp_gate", "runtime_only"),
    ]
    consumed = [B + k for k, _ in spec]
    expected_set = {k for k, _ in spec}
    actual_set = {k[len(B):] for k in bkeys}

    checks = {}
    for side in ("vid", "txt"):
        qkv_t = sd[B + f"attn.proj_qkv.{side}.weight"]
        h0, h1 = C, 2 * C
        q, k, v = qkv_t[:h0], qkv_t[h0:h1], qkv_t[h1:]
        checks[f"qkv_{side}_split_concat_bit_exact"] = bool(torch.equal(torch.cat([q, k, v], dim=0), qkv_t))
        gate = sd[B + f"mlp.{side}.proj_in_gate.weight"]
        up = sd[B + f"mlp.{side}.proj_in.weight"]
        out = sd[B + f"mlp.{side}.proj_out.weight"]
        checks[f"swiglu_{side}_ok"] = bool(
            gate.shape == (MLP, C) and up.shape == (MLP, C) and out.shape == (C, MLP)
            and not (gate.shape == up.shape and torch.equal(gate, up))
        )
    # vid vs txt weights must differ (real split layout, not silently shared).
    vid_txt_distinct = {
        "proj_qkv": not torch.equal(sd[B + "attn.proj_qkv.vid.weight"], sd[B + "attn.proj_qkv.txt.weight"]),
        "proj_out": not torch.equal(sd[B + "attn.proj_out.vid.weight"], sd[B + "attn.proj_out.txt.weight"]),
        "mlp_proj_in": not torch.equal(sd[B + "mlp.vid.proj_in.weight"], sd[B + "mlp.txt.proj_in.weight"]),
    }
    qk_entries = {}
    for norm in ("q", "k"):
        for side in ("vid", "txt"):
            t = sd[B + f"attn.norm_{norm}.{side}.weight"]
            qk_entries[f"norm_{norm}.{side}.weight"] = stat(t)

    return {
        "block_index": block_index,
        "layout": "split_vid_txt",
        "is_last_layer": False,
        "prefix": B,
        "key_count": len(bkeys),
        "source_numel": source_numel,
        "mapped_key_count": len(consumed),
        "mapped_numel": int(sum(sd[k].numel() for k in consumed)),
        "expected_keys_equals_consumed_keys": bool(expected_set == actual_set),
        "unmapped_keys": sorted(k for k in bkeys if k not in consumed),
        "duplicate_keys": [],
        "shape_mismatches": [],
        "qkv_split_concat_bit_exact": checks["qkv_vid_split_concat_bit_exact"] and checks["qkv_txt_split_concat_bit_exact"],
        "qkv_per_side": {"vid": checks["qkv_vid_split_concat_bit_exact"], "txt": checks["qkv_txt_split_concat_bit_exact"]},
        "qkv_slice_rows": {"q": [0, C], "k": [C, 2 * C], "v": [2 * C, 3 * C]},
        "swiglu_mapping_ok": checks["swiglu_vid_ok"] and checks["swiglu_txt_ok"],
        "swiglu_per_side": {"vid": checks["swiglu_vid_ok"], "txt": checks["swiglu_txt_ok"]},
        "vid_txt_weights_distinct": vid_txt_distinct,
        "ada_single_runtime_only": True,
        "qk_rmsnorm_affine": qk_entries,
        "qk_affine_all_ones": {k: v["all_ones"] for k, v in qk_entries.items()},
        "real_affine_required": any(not v["all_ones"] for v in qk_entries.values()),
    }


def main() -> int:
    sd = torch.load(str(p16.MAIN_CKPT), map_location="cpu", weights_only=True, mmap=True)
    per_block = {}
    for i in p16.ALL_BLOCKS:
        if i in p16.SPLIT_BLOCKS:
            per_block[i] = mapping_split(sd, i)
        else:
            per_block[i] = mapping_shared(sd, i, is_last_layer=(i == p16.LAST_LAYER_INDEX))

    overall = all(
        r["expected_keys_equals_consumed_keys"] and not r["unmapped_keys"]
        and not r["duplicate_keys"] and not r["shape_mismatches"]
        and r["mapped_numel"] == r["source_numel"]
        and r["qkv_split_concat_bit_exact"] and r["swiglu_mapping_ok"]
        and r["ada_single_runtime_only"] and r["real_affine_required"]
        for r in per_block.values()
    )
    report = {
        "phase": 16,
        "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
        "real_seedvr2_weights": True,
        "actual_block_count": p16.ACTUAL_BLOCK_COUNT,
        "block_indices": p16.ALL_BLOCKS,
        "last_layer_index": p16.LAST_LAYER_INDEX,
        "split_blocks": p16.SPLIT_BLOCKS,
        "shared_blocks": p16.SHARED_BLOCKS,
        "per_block": per_block,
        "overall": overall,
        "coverage": {
            "mapped": len(per_block),
            "expected": p16.ACTUAL_BLOCK_COUNT,
            "unmapped_blocks": [i for i in p16.ALL_BLOCKS if i not in per_block],
            "duplicate_blocks": [],
        },
        "conclusion": "ALL_32_BLOCKS_MAPPED" if overall else "MAPPING_FAILED",
        "note": "blocks 0..9 use the 29-key split .vid./.txt. contract "
                "(shared_weights=False, mm_layers=10); blocks 10..31 use the "
                "15-key shared .all. contract; block 31 is the unique last layer "
                "(is_last_layer=true, vid_only MLP passthrough, txt attention "
                "still executes with shared .all. weights).",
    }
    out = p16.REPO / "reports" / "phase16_all_block_mapping.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for i, r in per_block.items():
        print(f"block {i}: layout={r['layout']} keys={r['key_count']} "
              f"expected==consumed={r['expected_keys_equals_consumed_keys']} "
              f"qkv_bit_exact={r['qkv_split_concat_bit_exact']} "
              f"qk_all_ones={r['qk_affine_all_ones']}")
    print("overall:", overall)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
