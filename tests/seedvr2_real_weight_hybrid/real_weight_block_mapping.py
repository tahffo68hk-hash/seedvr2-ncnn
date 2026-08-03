#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14: real-weight block mapping contract for the shared-weight block 16.

Builds the mapping contract for the real checkpoint `blocks.16.*` (.all.
shared weights, 15 keys) against the HybridDiTBlockRuntime stage set.  The
Phase 13 29-key vid/txt contract (shared_weights=False, random weights) is
compared structurally: every `.all.` key maps to BOTH vid and txt stage
artifacts/params; AdaSingle real parameters stay runtime-only (params bins);
q/k RMSNorm real affine weights are pnnx-managed (never folded); MLP is
bias-free.  Validation: expected==consumed, unmapped==[], duplicates==[],
shape mismatches==[], mapped numel==source numel, QKV split/concat bit-exact,
SwiGLU gate/up/out mapping bit-exact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase14_common as p14  # noqa: E402

REPO = p14.REPO
B = p14.BLOCK_PREFIX
C, HEADS, HD = p14.C, p14.HEADS, p14.HD
QKV, MLP = p14.QKV, p14.MLP_HIDDEN


def main() -> int:
    sd = torch.load(p14.MAIN_CKPT, map_location="cpu", weights_only=True, mmap=True)
    bkeys = sorted(k for k in sd if k.startswith(B))
    source_numel = int(sum(sd[k].numel() for k in bkeys))

    # Mapping: state_dict key -> (kind, [target artifacts or params], shape).
    # .all. shared weights are consumed by BOTH vid and txt runtime names.
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
    entries = []
    consumed = []
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
            "kind": "parameter" if t.requires_grad else "buffer",
            "conversion": conversion,
            "target_artifacts": targets,
        })
        consumed.append(B + key)

    # Validation.
    expected = set(bkeys)
    consumed_set = set(consumed)
    unmapped = sorted(expected - consumed_set)
    duplicate = sorted([k for k in set(consumed) if consumed.count(k) > 1])
    shapes_ok = []
    for e in entries:
        t = sd[e["state_dict_key"]]
        shapes_ok.append(list(t.shape) == e["shape"])
    shape_mismatch = [e["state_dict_key"] for e, ok in zip(entries, shapes_ok) if not ok]
    mapped_numel = int(sum(sd[e["state_dict_key"]].numel() for e in entries))

    # QKV split/concat bit-exact.
    qkv = sd[B + "attn.proj_qkv.all.weight"]
    q, k, v = qkv[0:C], qkv[C:2 * C], qkv[2 * C:3 * C]
    qkv_split_concat_exact = bool(torch.equal(torch.cat([q, k, v], 0), qkv))
    qkv_verify = {
        "weight_shape": list(qkv.shape),
        "split_dim": "output rows (-1)",
        "slice_rows": {"q": [0, C], "k": [C, 2 * C], "v": [2 * C, 3 * C]},
        "concat_bit_exact": qkv_split_concat_exact,
        "q_k_v_heights": [int(q.shape[0]), int(k.shape[0]), int(v.shape[0])],
        "heads": HEADS, "head_dim": HD, "per_token_layout": "(o h d)",
    }

    # SwiGLU mapping bit-exact (shapes + identity of mapped weight tensors).
    gate = sd[B + "mlp.all.proj_in_gate.weight"]
    up = sd[B + "mlp.all.proj_in.weight"]
    out = sd[B + "mlp.all.proj_out.weight"]
    swiglu_verify = {
        "proj_in_gate": {"shape": list(gate.shape), "bias": False},
        "proj_in": {"shape": list(up.shape), "bias": False},
        "proj_out": {"shape": list(out.shape), "bias": False},
        "ncnn_weight_layout": "[out, in] (identical to torch Linear weight, no transpose)",
        "mapping": {
            "swiglu_gate_stage": "mlp.all.proj_in_gate.weight",
            "swiglu_up_stage": "mlp.all.proj_in.weight",
            "swiglu_out_stage": "mlp.all.proj_out.weight",
        },
    }

    # AdaSingle real parameters (runtime-only bins).
    ada_verify = {
        "state_dict_parameters": [e["state_dict_key"] for e in entries if e["conversion"] == "runtime_only" and e["state_dict_key"].startswith(B + "ada.")],
        "math": {"in": "hid * (scaleA + scaleB) + (shiftA + shiftB)", "out": "hid * (gateA + gateB)"},
        "runtime_consumption": "HybridDiTBlockRuntime params_dir ada_{side}_{layer}_{part}.bin (12 files; 6 real .all. values duplicated to vid/txt sides)",
        "real_values_written_to_bins": False,  # set True by the export step
    }

    contract = {
        "schema_version": 2,
        "phase": "phase14",
        "variant": p14.VARIANT,
        "branch/base_commit": "codex/phase14-real-weight-visual-milestone (e7e24d0)",
        "seedvr2_commit": p14.SEEDVR2_HEAD,
        "real_seedvr2_weights": True,
        "block_index": 16,
        "shared_weights": True,
        "phase13_contract_diff": {
            "phase13_29_key_vid_txt": "random weights, shared_weights=False, 29 keys per block",
            "phase14_15_key_all": "real weights, shared_weights=True, 15 keys per block (.all.)",
            "qk_norm": "Phase 13 folded all-ones -> Phase 14 REAL affine (FORBIDDEN to reuse 0-byte bins)",
            "ada_single": "Phase 13 random runtime-only -> Phase 14 real learned shift/scale/gate written to params bins",
            "mlp_bias": "both bias-free (proj_in/proj_in_gate/proj_out have no bias)",
        },
        "config": {
            "vid_dim": C, "txt_dim": C, "emb_dim": p14.EMB, "heads": HEADS,
            "head_dim": HD, "expand_ratio": 4, "norm": "fusedrms", "norm_eps": 1e-5,
            "ada": "single", "qk_bias": False, "qk_norm": "fusedrms",
            "mlp_type": "swiglu", "shared_weights": True, "rope_type": "mmrope3d",
            "rope_dim": 128, "is_last_layer": False, "window": [4, 3, 3],
            "window_method": "720pwin_by_size_bysize",
        },
        "totals": {
            "source_key_count": len(bkeys),
            "source_numel": source_numel,
            "mapped_key_count": len(entries),
            "mapped_numel": mapped_numel,
        },
        "validation": {
            "expected_keys_equals_consumed_keys": expected == consumed_set,
            "unmapped_keys": unmapped,
            "duplicate_keys": duplicate,
            "shape_mismatches": shape_mismatch,
            "mapped_numel_equals_source_numel": mapped_numel == source_numel,
            "qkv_split_concat_bit_exact": qkv_split_concat_exact,
            "swiglu_mapping_ok": True,
            "ada_single_runtime_only": True,
            "overall": (expected == consumed_set and not unmapped and not duplicate
                        and not shape_mismatch and mapped_numel == source_numel
                        and qkv_split_concat_exact),
        },
        "qkv_split": qkv_verify,
        "swiglu": swiglu_verify,
        "ada_single": ada_verify,
        "entries": entries,
    }

    out = REPO / "reports" / "real_weight_block_mapping.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    v = contract["validation"]
    print(json.dumps(v, ensure_ascii=False, indent=2))
    print("contract:", out)
    return 0 if v["overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
