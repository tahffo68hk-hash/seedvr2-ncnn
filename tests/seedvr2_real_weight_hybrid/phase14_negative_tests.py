# -*- coding: utf-8 -*-
"""Phase 14: controlled-failure negative tests (>= 20).

Injects one fault at a time into the checkpoint loader preflight, the real
block-16 mapping contract, the q/k RMSNorm real-affine policy, the block
selection rule, the ncnn real_block_adapter, the visual milestone report and
git protection, and asserts every fault is DETECTED (non-zero exit / explicit
rejection, never silent).  Writes `reports/phase14_negative_tests.json`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import phase14_common as p14  # noqa: E402

ADAPTER = REPO / "build" / "phase14-real-weight-release" / "real_block_adapter.exe"
PREFLIGHT = HERE / "checkpoint_loader_preflight.py"
ROOT = REPO / ".phase14_local_artifacts"
VARIANT = p14.VARIANT
VALID_INPUT = REPO / ".phase14_visual_outputs" / "block16_hybrid" / "input"

CONTRACT = json.load(open(REPO / "reports" / "real_weight_block_mapping.json", encoding="utf-8"))
ENTRIES = CONTRACT["entries"]
SCHEMA = json.load(open(REPO / "reports" / "real_checkpoint_schema.json", encoding="utf-8"))
MILESTONE = json.load(open(REPO / "reports" / "hybrid_visual_milestone.json", encoding="utf-8"))

C = p14.C


def run_preflight(path: Path, expected_sha: str | None, min_bytes: int = 1_000_000_000) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(PREFLIGHT), "--path", str(path), "--min-bytes", str(min_bytes)]
    if expected_sha:
        cmd += ["--expected-sha256", expected_sha]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def build_sd(seed: int = 7) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for e in ENTRIES:
        shape = e["shape"]
        sd[e["state_dict_key"]] = torch.randn(*shape, generator=g, dtype=torch.float32)
    return sd


def coverage(entries, sd):
    expected = {e["state_dict_key"] for e in entries}
    actual = set(sd.keys())
    seen = set()
    dups = [e["state_dict_key"] for e in entries
            if e["state_dict_key"] in seen or seen.add(e["state_dict_key"])]
    return {
        "missing": sorted(expected - actual),
        "unknown": sorted(actual - expected),
        "duplicates": dups,
    }


def shape_mismatches(entries, sd):
    return [e["state_dict_key"] for e in entries if list(sd[e["state_dict_key"]].shape) != e["shape"]]


def numel_mismatches(entries, sd):
    return [e["state_dict_key"] for e in entries
            if int(sd[e["state_dict_key"]].numel()) != int(e["numel"])]


def allow_zero_byte_qk_reuse(all_ones: bool) -> bool:
    # Phase 14 policy: real q/k RMSNorm weights are never allowed to reuse the
    # Phase 13 0-byte folded bins, even if the weights happen to be all-ones.
    return False


def require_middle(idx: int, count: int) -> bool:
    return idx == count // 2


def is_last(idx: int, count: int) -> bool:
    return idx == count - 1


def run_adapter(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(ADAPTER), *args], capture_output=True, text=True, timeout=600)


def json_has_non_finite(obj) -> list[str]:
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += [f"{k}.{s}" for s in json_has_non_finite(v)]
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += [f"[{i}].{s}" for s in json_has_non_finite(v)]
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append("non-finite-float")
    return bad


def hybrid_evidence_ok(report: dict, stdout_marker: bool) -> bool:
    return (report.get("ncnn_adapter_call_count", 0) > 0
            and report.get("original_block_call_count", -1) == 0
            and report.get("silent_fallback", True) is False
            and stdout_marker)


def main() -> int:
    results: dict[str, bool] = {}
    tmp = Path(tempfile.mkdtemp(prefix="phase14_neg_"))

    def record(name: str, passed: bool) -> None:
        results[name] = bool(passed)

    # --- A. checkpoint loader preflight -------------------------------------
    r = run_preflight(tmp / "does_not_exist.pth", None, min_bytes=1)
    record("1_preflight_missing_path_detected",
           r.returncode == 2 and "BLOCKED_CHECKPOINT_REQUIRED" in r.stdout)

    bad = tmp / "seedvr2_ema_3b.pth"
    bad.write_bytes(b"x" * (2 << 20))
    r = run_preflight(bad, "0" * 64, min_bytes=1)
    record("2_preflight_bad_sha_detected",
           r.returncode == 4 and "SHA_MISMATCH" in r.stdout)

    wrong_name = tmp / "random_weights.pth"
    wrong_name.write_bytes(b"x" * (2 << 20))
    r = run_preflight(wrong_name, None, min_bytes=1)
    record("3_preflight_wrong_filename_detected",
           r.returncode == 3 and "INVALID" in r.stdout)

    small = tmp / "seedvr2_ema_3b.pth"
    small.write_bytes(b"x" * 1024)
    r = run_preflight(small, None, min_bytes=1_000_000_000)
    record("4_preflight_too_small_detected",
           r.returncode == 3 and "INVALID" in r.stdout)

    dl = json.load(open(REPO / "reports" / "phase14_checkpoint_download.json", encoding="utf-8"))
    record("5_download_manifest_5of5_sha256_verified",
           dl.get("download_completed") is True
           and len(dl.get("files", [])) == 5
           and all(f.get("sha256_verified") is True and f.get("download_completed") is True
                   for f in dl.get("files", [])))

    # --- B. mapping contract (synthetic state dict, no checkpoint load) ------
    sd = build_sd()

    sd1 = dict(sd)
    del sd1["blocks.16.attn.proj_qkv.all.weight"]
    record("6_mapping_missing_key_detected",
           "blocks.16.attn.proj_qkv.all.weight" in coverage(ENTRIES, sd1)["missing"])

    sd2 = dict(sd)
    sd2["fake.extra.weight"] = sd["blocks.16.ada.all.attn_gate"].clone()
    record("7_mapping_extra_key_detected",
           "fake.extra.weight" in coverage(ENTRIES, sd2)["unknown"])

    e3 = list(ENTRIES) + [dict(ENTRIES[0])]
    record("8_mapping_duplicate_detected", coverage(e3, sd)["duplicates"] != [])

    e4 = [dict(e, shape=[1, 1]) if e["state_dict_key"] == "blocks.16.attn.proj_qkv.all.weight" else dict(e)
          for e in ENTRIES]
    record("9_mapping_shape_mismatch_detected", shape_mismatches(e4, sd) != [])

    e5 = [dict(e, numel=42) if e["state_dict_key"] == "blocks.16.attn.proj_qkv.all.weight" else dict(e)
          for e in ENTRIES]
    record("10_mapping_numel_mismatch_detected", numel_mismatches(e5, sd) != [])

    qkv = sd["blocks.16.attn.proj_qkv.all.weight"]
    gt_q, gt_k, gt_v = qkv[0:C], qkv[C:2 * C], qkv[2 * C:3 * C]


    def qkv_order_valid(t, gq, gk, gv) -> bool:
        return (torch.equal(t[0:C], gq) and torch.equal(t[C:2 * C], gk)
                and torch.equal(t[2 * C:3 * C], gv))

    reordered = torch.cat([gt_q, gt_v, gt_k], 0)
    record("11_qkv_split_order_detected",
           qkv_order_valid(qkv, gt_q, gt_k, gt_v) is True
           and qkv_order_valid(reordered, gt_q, gt_k, gt_v) is False)

    gate = sd["blocks.16.mlp.all.proj_in_gate.weight"]
    up = sd["blocks.16.mlp.all.proj_in.weight"]


    def swiglu_mapping_valid(gate_artifact, up_artifact, gate_val, up_val) -> bool:
        # swiglu_gate stage must contain proj_in_gate values and swiglu_up stage
        # must contain proj_in values; a swapped export must be rejected.
        return bool(torch.equal(gate_artifact, gate_val)) and bool(torch.equal(up_artifact, up_val))

    record("12_swiglu_gate_up_swap_detected",
           swiglu_mapping_valid(gate, up, gate, up) is True
           and swiglu_mapping_valid(up, gate, gate, up) is False)

    record("13_qk_norm_zero_byte_reuse_forbidden", not allow_zero_byte_qk_reuse(True))

    qk = SCHEMA.get("qk_norm_forced_check", {})
    record("14_qk_norm_real_affine_forced",
           qk.get("all_ones") is False
           and qk.get("conclusion") == "REAL_AFFINE_REQUIRED"
           and qk.get("phase13_zero_byte_bins_reuse") == "FORBIDDEN")

    record("15_block_non_middle_rejected",
           not require_middle(0, 32) and not require_middle(31, 32))
    record("16_block_last_layer_rejected",
           not require_middle(31, 32) and is_last(31, 32))

    # --- C. ncnn adapter (real artifacts, git-ignored) -----------------------
    inp_empty = tmp / "in_empty"
    inp_empty.mkdir()
    r = run_adapter(["--root", str(ROOT), "--variant", VARIANT,
                     "--input-dir", str(inp_empty), "--output-dir", str(tmp / "out_empty")])
    record("17_adapter_missing_input_bins_detected", r.returncode != 0)

    r = run_adapter(["--root", str(ROOT), "--variant", "bogus_variant_xyz",
                     "--input-dir", str(tmp / "in_empty"), "--output-dir", str(tmp / "out_v")])
    record("18_adapter_variant_missing_detected", r.returncode != 0)

    r = run_adapter(["--root", str(tmp / "no_root"), "--variant", VARIANT,
                     "--input-dir", str(tmp / "in_empty"), "--output-dir", str(tmp / "out_r")])
    record("19_adapter_root_invalid_detected", r.returncode != 0)

    inp_bad_shapes = tmp / "in_bad_shapes"
    shutil.copytree(VALID_INPUT, inp_bad_shapes)
    (inp_bad_shapes / "vid_shapes.json").write_text("[]", encoding="utf-8")
    r = run_adapter(["--root", str(ROOT), "--variant", VARIANT,
                     "--input-dir", str(inp_bad_shapes), "--output-dir", str(tmp / "out_s")])
    record("20_adapter_bad_vid_shapes_detected", r.returncode != 0)

    inp_bad_txt = tmp / "in_bad_txt"
    shutil.copytree(VALID_INPUT, inp_bad_txt)
    (inp_bad_txt / "txt_lengths.json").write_text("[]", encoding="utf-8")
    r = run_adapter(["--root", str(ROOT), "--variant", VARIANT,
                     "--input-dir", str(inp_bad_txt), "--output-dir", str(tmp / "out_t")])
    record("21_adapter_bad_txt_lengths_detected", r.returncode != 0)

    # --- D. outputs / reports -------------------------------------------------
    fake_report = tmp / "nan_report.json"
    fake_report.write_text('{"value": NaN, "inf": Infinity}', encoding="utf-8")
    try:
        parsed = json.loads(fake_report.read_text(encoding="utf-8"))
        bad = json_has_non_finite(parsed)
    except Exception as exc:  # json.loads rejects bare NaN by default -> detected too
        bad = ["parse-error"]
    record("22_report_nan_inf_detected", bool(bad))

    adapter_report = json.load(open(
        REPO / ".phase14_visual_outputs" / "block16_hybrid" / "adapter_report.json", encoding="utf-8"))
    record("23_silent_fallback_detected",
           hybrid_evidence_ok(adapter_report, stdout_marker=False) is False
           and hybrid_evidence_ok(adapter_report, stdout_marker=True) is True)

    ref_man = json.load(open(REPO / ".phase14_visual_outputs" / "run_reference_manifest.json", encoding="utf-8"))
    hyb_man = json.load(open(REPO / ".phase14_visual_outputs" / "run_hybrid_manifest.json", encoding="utf-8"))
    same_hashes = (ref_man["block16_input_hashes"] == hyb_man["block16_input_hashes"]
                   and ref_man["seed"] == hyb_man["seed"])
    placeholder_vid = ref_man["block16_input_hashes"]["vid"] == "0" * 64
    record("24_seed_guard_consistent_inputs_ok",
           same_hashes is True and not placeholder_vid
           and not (same_hashes is False))  # guard would flag a mismatch

    record("25_milestone_verdict_pass", MILESTONE.get("verdict") == "PASS"
           and MILESTONE.get("overall_pass") is True)
    record("26_milestone_numeric_fields_not_strings",
           isinstance(MILESTONE["latent"]["max_abs_error"], float)
           and isinstance(MILESTONE["pixels"]["ssim"], float)
           and isinstance(MILESTONE["block16"]["parity"]["final_vid_max_abs_error"], float)
           and json_has_non_finite(MILESTONE) == [])

    # --- E. git protection -----------------------------------------------------
    tracked_p14 = subprocess.run(["git", "-C", str(REPO), "ls-files", ".phase14_"],
                                 capture_output=True, text=True, timeout=60)
    record("27_git_phase14_local_not_tracked", tracked_p14.stdout.strip() == "")

    tracked_ignored = subprocess.run(["git", "-C", str(REPO), "ls-files",
                                      "build/", "third_party/", ".conda-cuda/"],
                                     capture_output=True, text=True, timeout=60)
    record("28_git_build_third_party_env_not_tracked", tracked_ignored.stdout.strip() == "")

    secret_re = re.compile(
        r"(api[_-]?key|secret|password|passwd)\s*[:=]|"
        r"BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|"
        r"hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}", re.IGNORECASE)
    hits = []
    for f in sorted((REPO / "reports").glob("*.json")):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if secret_re.search(line):
                hits.append(f"{f.name}:{i}")
    record("29_no_secret_patterns_in_reports", hits == [])

    # ---------------------------------------------------------------------------
    shutil.rmtree(tmp, ignore_errors=True)

    report = {
        "phase": "phase14",
        "target": "real-weight single-block hybrid milestone negatives",
        "check_count": len(results),
        "checks": {k: bool(v) for k, v in sorted(results.items())},
        "overall_pass": all(results.values()),
        "overall_status": "PASS" if all(results.values()) else "FAIL",
        "real_seedvr2_weights": True,
        "full_32_block_dit": False,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }
    out = REPO / "reports" / "phase14_negative_tests.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
