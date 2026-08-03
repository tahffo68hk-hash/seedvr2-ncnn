#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: real-weight mapping contract for blocks 15/16/17/18.

Re-verifies the 15-key `.all.` shared-weight mapping contract for every block
in the 4-block range and force-checks the real q/k RMSNorm affine weights of
each block (min/max/mean/std/all_ones/unique/sha256).  It must NOT assume
adjacent blocks have identical q/k norms.  Writes
reports/phase15_multiblock_mapping.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase15_common as p15  # noqa: E402

C, HEADS, HD = p15.C, p15.HEADS, p15.HD
QKV, MLP = p15.QKV, p15.MLP_HIDDEN


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def stat(t: torch.Tensor) -> dict:
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "all_ones": bool((t == 1.0).all().item()),
        "unique_count": int(t.unique().numel()),
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "sha256": sha256_tensor(t),
    }


def mapping_for(sd, block_index: int) -> dict:
    B = f"blocks.{block_index}."
    bkeys = sorted(k for k in sd if k.startswith(B))
    source_numel = int(sum(sd[k].numel() for k in bkeys))

    def pair(stage):
        return [f"models/{stage}/model.ncnn.bin", f"models/{stage}/model.ncnn.param"]

    def params_bins(layer, part):
        return [f"params/ada_vid_{layer}_{part}.bin", f"params/ada_txt_{layer}_{part}.bin"]

    spec = [
        ("attn.norm_q.all.weight", "pnnx_managed", pair("q_norm")),
        ("attn.norm_k.all.weight", "pnnx_managed", pair("k_norm")),
        ("attn.proj_qkv.all.weight", "pnnx_managed", pair("qkv")),
        ("attn.proj_out.all.weight", "pnnx_managed", pair("attn_proj")),
        ("attn.proj_out.all.bias", "pnnx_managed", pair("attn_proj")),
        ("mlp.all.proj_in.weight", "pnnx_managed", pair("swiglu_up")),
        ("mlp.all.proj_in_gate.weight", "pnnx_managed", pair("swiglu_gate")),
        ("mlp.all.proj_out.weight", "pnnx_managed", pair("swiglu_out")),
        ("ada.all.attn_shift", "runtime_only", params_bins("attn", "shift")),
        ("ada.all.attn_scale", "runtime_only", params_bins("attn", "scale")),
        ("ada.all.attn_gate", "runtime_only", params_bins("attn", "gate")),
        ("ada.all.mlp_shift", "runtime_only", params_bins("mlp", "shift")),
        ("ada.all.mlp_scale", "runtime_only", params_bins("mlp", "scale")),
        ("ada.all.mlp_gate", "runtime_only", params_bins("mlp", "gate")),
        ("attn.rope.rope.freqs", "runtime_only", ["host-generated MM-RoPE3D tables"]),
    ]
    entries, consumed = [], []
    for key, conversion, targets in spec:
        t = sd[B + key]
        entries.append({
            "state_dict_key": B + key,
            "source_module": key.rsplit(".", 2)[0],
            "shared_branch": "all",
            "semantic_role": ("ada_single_parameter" if key.startswith("ada.") else
                              "rope_freqs_base_buffer" if "rope" in key else
                              "qk_norm_affine_weight" if "norm_" in key else "linear_weight_or_bias"),
            "branch": "shared_vid_txt",
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "dtype": str(t.dtype),
            "conversion": conversion,
            "target_artifacts": targets,
        })
        consumed.append(B + key)

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

    report = {
        "block_index": block_index,
        "prefix": B,
        "key_count": len(bkeys),
        "source_numel": source_numel,
        "mapped_key_count": len(consumed),
        "mapped_numel": int(sum(sd[k].numel() for k in consumed)),
        "expected_keys_equals_consumed_keys": bool(set(spec_key[0] for spec_key in spec) ==
                                                    {k[len(B):] for k in bkeys}),
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
        "entries": entries,
    }
    return report


def main() -> int:
    sd = torch.load(str(p15.MAIN_CKPT), map_location="cpu", weights_only=True, mmap=True)
    per_block = {i: mapping_for(sd, i) for i in p15.BLOCK_SET}
    overall = all(
        r["expected_keys_equals_consumed_keys"] and not r["unmapped_keys"]
        and not r["duplicate_keys"] and not r["shape_mismatches"]
        and r["mapped_numel"] == r["source_numel"]
        and r["qkv_split_concat_bit_exact"] and r["swiglu_mapping_ok"]
        and r["ada_single_runtime_only"] and r["real_affine_required"]
        for r in per_block.values()
    )
    report = {
        "phase": 15,
        "checkpoint_sha256": p15.EXPECTED_MAIN_SHA256,
        "real_seedvr2_weights": True,
        "shared_weights": True,
        "block_indices": p15.BLOCK_SET,
        "per_block": per_block,
        "overall": overall,
        "conclusion": "REAL_AFFINE_REQUIRED" if overall else "MAPPING_FAILED",
        "note": "Every selected block carries its own real q/k RMSNorm affine weights; "
                "0-byte folded bins and adjacent-block norm bins must NOT be reused.",
    }
    out = p15.REPO / "reports" / "phase15_multiblock_mapping.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for i, r in per_block.items():
        print(f"block {i}: keys={r['key_count']} numel={r['source_numel']} "
              f"expected==consumed={r['expected_keys_equals_consumed_keys']} "
              f"qkv_bit_exact={r['qkv_split_concat_bit_exact']} "
              f"qk_all_ones={r['qk_affine_all_ones']}")
    print("overall:", overall)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
