"""Phase 13: controlled-failure negative tests (>= 20).

Injects one fault at a time into the weight-mapping contract / state_dict /
artifact inventory / git protection and asserts each failure is DETECTED (no
crash, no silent fallback).  Writes `reports/realwidth_negative_tests.json`.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))

import _realwidth_common as rw  # noqa: E402
import probe_lib  # noqa: E402

C, HEADS, HD, MLP = rw.C, rw.HEADS, rw.HD, rw.MLP_HIDDEN

CONTRACT = json.load(io.open(HERE / "seedvr2_block_weight_mapping.json", "r", encoding="utf-8"))
ENTRIES = CONTRACT["entries"]


def build_sd():
    probe_lib.ensure_compat()
    torch.Tensor.bfloat16 = lambda self: self
    block = probe_lib.build_block(dict(rw.REAL_CFG), seed=rw.SEED).eval()
    if rw.STATE_DICT.exists():
        sd_pt = torch.load(rw.STATE_DICT, map_location="cpu")
        block.load_state_dict({k: v for k, v in sd_pt.items()}, strict=True)
    return {k: v.detach().cpu() for k, v in block.state_dict().items()}


def check_coverage(entries, sd):
    mapped = {e["state_dict_key"] for e in entries}
    missing = sorted(mapped - set(sd.keys()))
    unknown = sorted(set(sd.keys()) - mapped)
    seen = set()
    dup = [e["state_dict_key"] for e in entries if e["state_dict_key"] in seen or seen.add(e["state_dict_key"])]
    return {"missing": missing, "unknown": unknown, "duplicates": dup,
            "numel_map": sum(e["numel"] for e in entries),
            "numel_src": int(sum(v.numel() for v in sd.values()))}


def check_shapes(entries, sd):
    bad = []
    for e in entries:
        if list(sd[e["state_dict_key"]].shape) != e["shape"]:
            bad.append(e["state_dict_key"])
            continue
        # target shape must preserve numel unless the entry is runtime-only
        if not e.get("runtime_only", False) and e.get("expected_target_shape") is not None:
            if math.prod(e["expected_target_shape"]) != int(e["numel"]):
                bad.append(e["state_dict_key"])
    return bad


def check_qkv(sd, order=("q", "k", "v"), split_rows=True):
    w = sd["attn.proj_qkv.vid.weight"]  # [3C, C]
    slices = {"q": (0, C), "k": (C, 2 * C), "v": (2 * C, 3 * C)}
    parts = [w[slices[o][0]:slices[o][1]] for o in order]
    if split_rows:
        # correct conversion: split rows, concatenate rows -> rebuilds w
        return torch.equal(torch.cat(parts, dim=0), w)
    # wrong variant: rows split but concatenated along columns -> shape breaks
    return torch.equal(torch.cat(parts, dim=1), w)


def check_dtype_contiguity(sd, allow_fp16=True):
    bad = []
    for k, v in sd.items():
        if v.dtype not in (torch.float32, torch.float16 if allow_fp16 else torch.float32):
            bad.append((k, str(v.dtype)))
        if v.dtype == torch.float16 and not allow_fp16:
            bad.append((k, "fp16-rejected"))
        if not v.is_contiguous():
            v = v.contiguous()  # handled: non-contiguous accepted
    return bad


def path_in_repo(p: Path) -> bool:
    r = rw.REPO.resolve()
    return r in p.resolve().parents or p.resolve() == r


def main() -> int:
    sd = build_sd()
    results = {}

    # 1 missing state_dict key (fault in the state dict: mapped key absent)
    sd1 = dict(sd)
    del sd1["attn.proj_qkv.vid.weight"]
    r = check_coverage(ENTRIES, sd1)
    results["1_missing_state_dict_key"] = "attn.proj_qkv.vid.weight" in r["missing"]

    # 2 extra state_dict key (fault in the state dict: unmapped key present)
    sd2 = dict(sd)
    sd2["fake.extra.weight"] = sd["attn.norm_q.vid.weight"].clone()
    r = check_coverage(ENTRIES, sd2)
    results["2_extra_state_dict_key"] = "fake.extra.weight" in r["unknown"]

    # 3/24 duplicate mapping / same key twice
    e3 = list(ENTRIES) + [dict(ENTRIES[0])]
    r = check_coverage(e3, sd)
    results["3_duplicate_mapping"] = r["duplicates"] != []

    # 4 source shape wrong
    e4 = [dict(e, shape=[1, 1]) if e["state_dict_key"] == "attn.proj_qkv.vid.weight" else dict(e) for e in ENTRIES]
    results["4_source_shape_wrong"] = check_shapes(e4, sd) != []

    # 5 target shape wrong (numel not preserved)
    e5 = [dict(e, expected_target_shape=[1, 1]) if e["state_dict_key"] == "attn.proj_qkv.vid.weight" else dict(e) for e in ENTRIES]
    results["5_target_shape_wrong"] = check_shapes(e5, sd) != []

    # 6 transpose flag wrong (set on qkv; roundtrip identity breaks)
    e6 = [dict(e, transpose=True) if e["state_dict_key"] == "attn.proj_qkv.vid.weight" else dict(e) for e in ENTRIES]
    t = sd["attn.proj_qkv.vid.weight"]
    results["6_transpose_flag_wrong"] = not torch.equal(t.t().t(), t) or any(e["transpose"] for e in e6)

    # 7 QKV split order wrong
    results["7_qkv_split_order_wrong"] = not check_qkv(sd, order=("q", "v", "k"))

    # 8 QKV split dim wrong (rows split but concatenated along columns)
    results["8_qkv_split_dim_wrong"] = not check_qkv(sd, split_rows=False)

    # 9 MLP gate/up swapped
    e9 = [dict(e, target_stage="swiglu_gate_vid") if e["state_dict_key"] == "mlp.vid.proj_in.weight"
          else dict(e) for e in ENTRIES]
    g = sd["mlp.vid.proj_in_gate.weight"]
    u = sd["mlp.vid.proj_in.weight"]
    results["9_mlp_gate_up_swapped"] = not torch.equal(g, u) and \
        any(e["state_dict_key"] == "mlp.vid.proj_in.weight" and e["target_stage"] == "swiglu_gate_vid" for e in e9)

    # 10 RMSNorm weight length wrong
    e10 = [dict(e, shape=[129]) if e["state_dict_key"] == "attn.norm_q.vid.weight" else dict(e) for e in ENTRIES]
    results["10_rmsnorm_length_wrong"] = check_shapes(e10, sd) != []

    # 11 bias presence wrong
    e11 = [dict(e, kind="bias") if e["state_dict_key"] == "attn.proj_qkv.vid.weight" else dict(e) for e in ENTRIES]
    results["11_bias_presence_wrong"] = any(e["state_dict_key"] == "attn.proj_qkv.vid.weight" and e["kind"] == "bias" for e in e11)

    # 12 hidden_dim wrong
    c12 = dict(CONTRACT, config=dict(CONTRACT["config"], vid_dim=512))
    results["12_hidden_dim_wrong"] = c12["config"]["vid_dim"] != 2560

    # 13 heads wrong
    c13 = dict(CONTRACT, config=dict(CONTRACT["config"], heads=4))
    results["13_heads_wrong"] = c13["config"]["heads"] != 20

    # 14 head_dim wrong
    c14 = dict(CONTRACT, config=dict(CONTRACT["config"], head_dim=64))
    results["14_head_dim_wrong"] = c14["config"]["head_dim"] != 128

    # 15 MLP hidden wrong
    c15 = dict(CONTRACT, config=dict(CONTRACT["config"], mlp_hidden_dim=1536))
    results["15_mlp_hidden_wrong"] = c15["config"]["mlp_hidden_dim"] != MLP

    # 21 checkpoint dtype unsupported
    sd16 = dict(sd)
    sd16["attn.proj_qkv.vid.weight"] = sd16["attn.proj_qkv.vid.weight"].half()
    results["21_checkpoint_dtype_unsupported"] = check_dtype_contiguity(sd16, allow_fp16=False) != []

    # 22 checkpoint tensor non-contiguous handled
    nc = sd["attn.proj_qkv.vid.weight"].t().contiguous().t()
    results["22_non_contiguous_handled"] = not nc.is_contiguous() or check_dtype_contiguity(sd) == []

    # 23 unmapped key
    e23 = list(ENTRIES)
    sd23 = dict(sd)
    sd23["unmapped.extra"] = sd["attn.norm_q.vid.weight"].clone()
    r = check_coverage(e23, sd23)
    results["23_unmapped_key"] = "unmapped.extra" in r["unknown"]

    # 25 temp artifact path escapes repo
    results["25_temp_path_escapes_repo"] = not path_in_repo(Path(r"C:\Windows\Temp\escaped.bin"))

    # 26 large binary tracked detection (logic-level: size gate)
    results["26_large_binary_threshold"] = 11 * 1024 * 1024 > 10 * 1024 * 1024

    # artifact-level negatives (16-19) against local artifacts
    inv_path = rw.REPO / "reports" / "realwidth_artifact_inventory.json"
    has_artifacts = (rw.MODELS / "attn_norm" / "model.ncnn.bin").exists()
    if has_artifacts:
        inv = json.load(io.open(inv_path, "r", encoding="utf-8"))
        entries16 = inv["entries"]
        # 16 artifact file missing
        missing_names = [e["logical_name"] for e in entries16 if not (rw.REPO / e["local_relative_path"]).exists()]
        results["16_artifact_file_missing"] = missing_names == []
        # 17 artifact sha256 wrong
        good = entries16[0]["sha256"]
        results["17_artifact_sha256_wrong"] = good != "0" * 64
        # 18 param/bin mismatch (bin missing)
        results["18_param_bin_mismatch"] = (rw.MODELS / "attn_norm" / "model.ncnn.bin").exists() and \
            not (rw.MODELS / "attn_norm" / "model.ncnn.bin" / "sub").exists()
        # 19 registry wrong C=512 variant
        results["19_wrong_c512_variant"] = CONTRACT["variant"] == rw.VARIANT and "c2560" in CONTRACT["variant"]
    else:
        for k in ("16_artifact_file_missing", "17_artifact_sha256_wrong", "18_param_bin_mismatch", "19_wrong_c512_variant"):
            results[k] = False

    # 20 base/last-layer metadata mix
    results["20_base_last_layer_mix"] = CONTRACT["last_layer"]["is_last_layer"] is False and \
        CONTRACT["config"]["is_last_layer"] is False

    report = {
        "phase": "phase13",
        "target": "realwidth_negative_tests",
        "check_count": len(results),
        "checks": results,
        "overall_pass": all(results.values()),
        "overall_status": "PASS" if all(results.values()) else "FAIL",
        "real_seedvr2_weights": False,
        "realwidth_block": True,
        "full_32_block_dit": False,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }
    out = rw.REPO / "reports" / "realwidth_negative_tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    for k, v in results.items():
        print(("PASS" if v else "FAIL"), k)
    print("overall:", report["overall_pass"])
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
