"""Phase 13: derive the committed machine-readable weight mapping contract.

Reads the real-object inventory produced by `extract_real_block_state_dict.py`
and annotates every parameter with target stage / target artifact / target
blob / conversion (restricted enum) / transpose / reshape / split / concat /
expected target shape / shared-or-independent / last-layer applicability.
The contract is stdlib-JSON-parseable and is validated bit-exactly by
`validate_weight_mapping_contract.py`.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _realwidth_common as rw  # noqa: E402

# conversion enum (task section 9)
IDENTITY = "identity"
TRANSPOSE_2D = "transpose_2d"
SPLIT_QKV = "split_qkv"
RESHAPE_PER_HEAD_NORM = "reshape_per_head_norm"
PNNX_MANAGED = "pnnx_managed"
RUNTIME_ONLY = "runtime_only"
NOT_APPLICABLE = "not_applicable"

# stage target for each semantic role / branch
STAGE_TARGET = {
    ("qkv_projection", "vid"): "qkv_vid",
    ("qkv_projection", "txt"): "qkv_txt",
    ("attention_output_projection", "vid"): "attn_proj_vid",
    ("attention_output_projection", "txt"): "attn_proj_txt",
    ("q_rmsnorm", "vid"): "q_norm_vid",
    ("q_rmsnorm", "txt"): "q_norm_txt",
    ("k_rmsnorm", "vid"): "k_norm_vid",
    ("k_rmsnorm", "txt"): "k_norm_txt",
    ("swiglu_gate_projection", "vid"): "swiglu_gate_vid",
    ("swiglu_gate_projection", "txt"): "swiglu_gate_txt",
    ("swiglu_up_projection", "vid"): "swiglu_up_vid",
    ("swiglu_up_projection", "txt"): "swiglu_up_txt",
    ("swiglu_output_projection", "vid"): "swiglu_out_vid",
    ("swiglu_output_projection", "txt"): "swiglu_out_txt",
}


def conversion_for(role: str) -> str:
    if role in ("qkv_projection", "attention_output_projection",
                "swiglu_gate_projection", "swiglu_up_projection", "swiglu_output_projection"):
        return PNNX_MANAGED
    if role in ("q_rmsnorm", "k_rmsnorm"):
        return IDENTITY
    if role == "ada_single_parameter":
        return RUNTIME_ONLY
    return NOT_APPLICABLE


def is_last_layer_applicable(role: str, branch: str) -> bool:
    # last-layer keeps attention (both branches), vid-only MLP, vid-only ada.
    if role in ("qkv_projection", "attention_output_projection", "q_rmsnorm", "k_rmsnorm"):
        return True
    if role in ("swiglu_gate_projection", "swiglu_up_projection", "swiglu_output_projection",
                "ada_single_parameter"):
        return branch == "vid"
    return False


def main() -> int:
    inventory = json.load(io.open(rw.INVENTORY, "r", encoding="utf-8"))
    entries = []
    consumed = {}
    for e in inventory["entries"]:
        role = e["semantic_role"]
        branch = e["branch"]
        conv = conversion_for(role)
        if e["kind"] == "buffer" and role == "rope_freqs_base_buffer":
            entries.append({
                "state_dict_key": e["state_dict_key"],
                "source_module": e["source_module"],
                "source_module_class": e["source_module_class"],
                "semantic_role": role,
                "branch": branch,
                "shape": e["shape"],
                "numel": e["numel"],
                "dtype": e["dtype"],
                "requires_grad": False,
                "kind": "buffer",
                "target_stage": "runtime_only",
                "target_artifact": "host-generated RoPE tables (theta^(-k/21))",
                "target_blob": "none",
                "conversion": "runtime_only",
                "transpose": False,
                "reshape": False,
                "split": False,
                "concat": False,
                "expected_target_shape": list(e["shape"]),
                "shared_or_independent": "independent_per_branch",
                "is_last_layer_applicable": True,
            })
            consumed[e["state_dict_key"]] = True
            continue
        if role == "ada_single_parameter":
            layer = "attn" if "attn_" in e["state_dict_key"] else "mlp"
            part = "shift" if e["state_dict_key"].endswith("_shift") else \
                   "scale" if e["state_dict_key"].endswith("_scale") else "gate"
            target_stage = "runtime_only"
            target_artifact = f"params/ada_{branch}_{layer}_{part}.bin"
            target_blob = "param_file"
            expected = list(e["shape"])
        else:
            target_stage = STAGE_TARGET[(role, branch)]
            target_artifact = f"models/{target_stage}/model.ncnn.param/.bin"
            target_blob = "ncnn_bin_weight"
            expected = list(e["shape"])  # ncnn InnerProduct keeps [out,in]; norm keeps [dim]
        transpose = False
        reshape = False
        split = False
        concat = False
        if role == "qkv_projection":
            split = True  # Q/K/V are slices of the 7680-wide output rows
        entry = {
            "state_dict_key": e["state_dict_key"],
            "source_module": e["source_module"],
            "source_module_class": e["source_module_class"],
            "semantic_role": role,
            "branch": branch,
            "shape": e["shape"],
            "numel": e["numel"],
            "dtype": e["dtype"],
            "requires_grad": e["requires_grad"],
            "kind": e["kind"],
            "target_stage": target_stage,
            "target_artifact": target_artifact,
            "target_blob": target_blob,
            "conversion": conv,
            "transpose": transpose,
            "reshape": reshape,
            "split": split,
            "concat": concat,
            "expected_target_shape": expected,
            "shared_or_independent": "independent_per_branch",
            "is_last_layer_applicable": is_last_layer_applicable(role, branch),
        }
        if e["state_dict_key"] in consumed:
            raise SystemExit(f"duplicate consumption: {e['state_dict_key']}")
        consumed[e["state_dict_key"]] = entry
        entries.append(entry)

    sd_keys = set(inventory["entries"][i]["state_dict_key"] for i in range(len(inventory["entries"])))
    mapped_keys = set(consumed.keys())
    unmapped = sorted(sd_keys - mapped_keys)
    duplicates = []

    contract = {
        "schema_version": 1,
        "phase": "phase13",
        "branch/base_commit": "codex/phase13-real-width-weight-map (c05a5a4)",
        "seedvr2_commit": "e4de8c24441a67e1b7df56abea10645059bb1185",
        "variant": rw.VARIANT,
        "seed": rw.SEED,
        "real_seedvr2_weights": False,
        "realwidth_block": True,
        "config": dict(rw.REAL_CFG),
        "totals": {
            "parameter_count": len(entries),
            "numel": int(sum(e["numel"] for e in entries)),
            "theoretical_fp32_bytes": int(sum(e["numel"] for e in entries) * 4),
            "mapped_parameter_count": len(entries),
            "mapped_numel": int(sum(e["numel"] for e in entries)),
        },
        "coverage": {
            "mapped_parameter_count": len(entries),
            "source_parameter_count": len(inventory["entries"]),
            "mapped_numel": int(sum(e["numel"] for e in entries)),
            "source_numel": int(sum(e["numel"] for e in inventory["entries"])),
            "unmapped_keys": unmapped,
            "duplicate_keys": duplicates,
        },
        "conversions_used": sorted({e["conversion"] for e in entries}),
        "qkv_split": {
            "order": ["q", "k", "v"],
            "split_dim": "output rows (-1)",
            "output_dim": rw.QKV,
            "inner_dim": rw.C,
            "heads": rw.HEADS,
            "head_dim": rw.HD,
            "per_token_layout": "(o h d)",
            "slice_rows": {"q": [0, rw.C], "k": [rw.C, 2 * rw.C], "v": [2 * rw.C, 3 * rw.C]},
            "bias": False,
        },
        "rmsnorm": {
            "attn_norm/mlp_norm": {"elementwise_affine": False, "state_dict_parameters": []},
            "norm_q/norm_k": {"elementwise_affine": True, "weight_shape": [rw.HD],
                              "per_head_norm_dim": rw.HD, "shared_per_head": True,
                              "epsilon": 1e-5, "epsilon_source": "constructor norm_eps (not in state_dict)"},
        },
        "swiglu": {
            "proj_in_gate": {"shape": [rw.MLP_HIDDEN, rw.C], "bias": False},
            "proj_in": {"shape": [rw.MLP_HIDDEN, rw.C], "bias": False},
            "proj_out": {"shape": [rw.C, rw.MLP_HIDDEN], "bias": False},
            "ncnn_weight_layout": "[out, in] (identical to torch Linear weight, no transpose)",
        },
        "ada_single": {
            "state_dict_parameters": sorted(k for k in sd_keys if k.startswith("ada.")),
            "runtime_dynamic_input": "emb",
            "emb_dim": rw.EMB,
            "emb_layout": "b (d l g) -> b d l g",
            "per_layer_group": ["attn_shift", "attn_scale", "attn_gate", "mlp_shift", "mlp_scale", "mlp_gate"],
            "consumed_by": "HybridDiTBlockRuntime params_dir (ada_{side}_{layer}_{part}.bin)",
            "ncnn_artifact": False,
        },
        "last_layer": {
            "is_last_layer": False,
            "parameter_sets_identical_with_base": "verified_by_extract (same constructor seed for shared modules)",
            "control_flow_only": True,
            "txt_path_semantics": "vid_only: mlp_norm/ada/mlp identity on txt; attention runs on txt",
            "artifact_reuse": "all 19 base stage models reused; no real-width last-layer re-export",
            "is_last_layer_applicable_count": sum(1 for e in entries if e["is_last_layer_applicable"]),
        },
        "stage_exports": {s: {"model_param": f"models/{s}/model.ncnn.param", "model_bin": f"models/{s}/model.ncnn.bin"}
                          for s in rw.BASE_STAGES},
        "entries": entries,
    }
    out = HERE / "seedvr2_block_weight_mapping.json"
    io.open(out, "w", encoding="utf-8").write(json.dumps(contract, indent=2))
    print("mapped:", len(entries), "numel:", contract["totals"]["numel"])
    print("unmapped:", unmapped)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
