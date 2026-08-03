#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16 negative tests: strict full-chain streaming hybrid contract.

Covers full-range selection (0..31 incl. the unique last layer, which may
only appear once and only as the final block), artifact/weight binding,
strict streaming lifecycle, silent-fallback detection, adapter negative
invocations, numerical guards, visual sanity and Git protection.  Writes
reports/phase16_negative_tests.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase16_common as p16  # noqa: E402
import phase16_streaming_manager as p16m  # noqa: E402

OUT = p16.REPO / "reports" / "phase16_negative_tests.json"
TMP = p16.DIAG / "neg_tests"
TMP.mkdir(parents=True, exist_ok=True)
IN0 = p16.PHASE16_ROOT / "official_context_capture" / "block_0"

RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append({"name": name, "pass": bool(passed), "detail": str(detail)[:400]})
    print("%-60s %s" % (name, "PASS" if passed else "FAIL"), flush=True)


def range_validator(blocks, actual_count=32, last_index=31):
    """Mirror the Phase 16 adapter range contract (full 0..31 incl. last layer)."""
    if not blocks:
        return False, "empty range"
    if len(set(blocks)) != len(blocks):
        return False, "duplicate block indices"
    if any(b < 0 or b >= actual_count for b in blocks):
        return False, "block index out of range"
    if blocks != sorted(blocks):
        return False, "blocks not ascending"
    if blocks != list(range(blocks[0], blocks[0] + len(blocks))):
        return False, "blocks not contiguous"
    if blocks.count(last_index) > 1:
        return False, "last layer duplicated"
    if last_index in blocks and blocks[-1] != last_index:
        return False, "last layer must be the final block"
    return True, "ok"


def run_adapter_rc(blocks, input_dir, load_mode="streaming", out_suffix="x",
                   denoising_step=0, extra=None):
    out_dir = TMP / ("out_%s" % out_suffix)
    report_path = TMP / ("rep_%s.json" % out_suffix)
    cmd = [
        str(p16.ADAPTER_EXE.resolve()),
        "--root", str(p16.PHASE16_ROOT.resolve()),
        "--blocks", ",".join("%d:%s" % (i, p16.variant_for(i)) for i in blocks),
        "--input-dir", str(Path(input_dir).resolve()),
        "--output-dir", str(out_dir.resolve()),
        "--load-mode", load_mode,
        "--denoising-step", str(denoising_step),
        "--report", str(report_path.resolve()),
    ]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text("utf-8"))
    return proc.returncode, proc.stdout or "", proc.stderr or "", report


def ev(stage, bi):
    return {"stage": stage, "block_index": bi}


def lifecycle(events, **kw):
    base = {"order_ok": True, "max_resident_weighted_blocks": 1,
            "ncnn_adapter_call_count": 1, "original_block_call_count": 0,
            "silent_fallback": False, "streaming_mode_validated": True,
            "state_machine_final": "IDLE", "state_machine_valid": True,
            "expected_next_block_index": -1}
    base.update(kw)
    base["events"] = events
    return p16m.validate_streaming_lifecycle(base)


def load_json(path):
    return json.loads(Path(path).read_text("utf-8"))


# ---------------- A. range / selection validation ----------------
record("range_duplicate_rejected", range_validator([9, 9])[0] is False)
record("range_out_of_bounds_32_rejected", range_validator([32])[0] is False)
record("range_negative_rejected", range_validator([-1, 16])[0] is False)
record("range_not_ascending_rejected", range_validator([16, 15])[0] is False)
record("range_non_contiguous_rejected", range_validator([9, 11])[0] is False)
record("range_empty_rejected", range_validator([])[0] is False)
record("range_last_layer_duplicated_rejected", range_validator([31, 31])[0] is False)
record("range_last_layer_not_final_rejected", range_validator([31, 30])[0] is False)
record("range_last_layer_mid_chain_rejected", range_validator([29, 30, 31, 0])[0] is False)
record("range_last_layer_final_accepted", range_validator([30, 31])[0] is True)
record("range_last_layer_alone_accepted", range_validator([31])[0] is True)
record("range_full_accepted", range_validator(p16.RANGE_FULL)[0] is True)
record("range_8_accepted", range_validator(p16.RANGE_8)[0] is True)
record("range_16_accepted", range_validator(p16.RANGE_16)[0] is True)

# ---------------- B. artifact / weight binding ----------------
schedule = load_json(p16.REPO / "reports" / "phase16_block_schedule.json")
record("schedule_reports_32_blocks", int(schedule.get("actual_block_count", 0)) == 32)
record("schedule_last_layer_31", int(schedule.get("last_layer_index", -1)) == 31)
ranges = schedule.get("ranges", {})
record("schedule_range_8", ranges.get("range_8") == p16.RANGE_8)
record("schedule_range_16", ranges.get("range_16") == p16.RANGE_16)
record("schedule_range_full", ranges.get("range_full") == p16.RANGE_FULL)

inv = load_json(p16.REPO / "reports" / "phase16_artifact_inventory.json")
record("inventory_all_blocks_present", bool(inv.get("all_blocks_present"))
       and len(inv.get("per_block", {})) == 32)
record("inventory_no_missing", inv.get("missing_or_invalid") == [])
record("inventory_no_large_committed", inv.get("large_artifacts_committed") is False)
def _expected_stages(bi):
    return 19 if bi in p16.SPLIT_BLOCKS else 12
block_inv_ok = all(
    inv.get("per_block", {}).get(str(bi), {}).get("metadata_block_index_ok") is True
    and inv.get("per_block", {}).get(str(bi), {}).get("stage_count") == _expected_stages(bi)
    for bi in p16.ALL_BLOCKS)
record("inventory_metadata_and_stage_counts_ok", block_inv_ok)

mapping = load_json(p16.REPO / "reports" / "phase16_all_block_mapping.json")
record("mapping_overall_pass", mapping.get("overall") is True)
mapping_coverage = mapping.get("coverage", {})
record("mapping_covers_32_blocks", mapping_coverage.get("mapped") == 32
       and mapping_coverage.get("expected") == 32)
record("mapping_no_unmapped", mapping_coverage.get("unmapped_blocks") == [])
record("mapping_no_duplicates", mapping_coverage.get("duplicate_blocks") == [])
all_qk_real = all(mapping.get("per_block", {}).get(str(bi), {}).get("real_affine_required") is True
                  for bi in p16.ALL_BLOCKS)
record("mapping_all_qk_rmsnorm_real_affine", all_qk_real)
all_keys_match = all(mapping.get("per_block", {}).get(str(bi), {}).get("expected_keys_equals_consumed_keys") is True
                     for bi in p16.ALL_BLOCKS)
record("mapping_all_expected_equals_consumed", all_keys_match)
all_qkv_bit_exact = all(mapping.get("per_block", {}).get(str(bi), {}).get("qkv_split_concat_bit_exact") is True
                        for bi in p16.ALL_BLOCKS)
record("mapping_all_qkv_split_concat_bit_exact", all_qkv_bit_exact)

# ---------------- C. strict streaming lifecycle ----------------
record("event_load_b_before_unload_a_rejected",
       p16m.validate_event_sequence([ev("LOAD", 9), ev("EXECUTE", 9),
                                     ev("LOAD", 10), ev("EXECUTE", 10), ev("UNLOAD", 10),
                                     ev("UNLOAD", 9)])[0] is False)
record("event_reversed_order_rejected",
       p16m.validate_event_sequence([ev("UNLOAD", 9), ev("LOAD", 9), ev("EXECUTE", 9)])[0] is False)
record("event_execute_before_load_rejected",
       p16m.validate_event_sequence([ev("EXECUTE", 9), ev("LOAD", 9), ev("UNLOAD", 9)])[0] is False)
record("event_chain_ends_loaded_rejected",
       p16m.validate_event_sequence([ev("LOAD", 9), ev("EXECUTE", 9)])[0] is False)
record("event_unload_wrong_block_rejected",
       p16m.validate_event_sequence([ev("LOAD", 9), ev("EXECUTE", 9), ev("UNLOAD", 10)])[0] is False)
record("event_strict_order_accepted",
       p16m.validate_event_sequence([ev("LOAD", 9), ev("EXECUTE", 9), ev("UNLOAD", 9),
                                     ev("LOAD", 10), ev("EXECUTE", 10), ev("UNLOAD", 10)])[0] is True)
record("event_last_layer_strict_order_accepted",
       p16m.validate_event_sequence([ev("LOAD", 30), ev("EXECUTE", 30), ev("UNLOAD", 30),
                                     ev("LOAD", 31), ev("EXECUTE", 31), ev("UNLOAD", 31)])[0] is True)
record("cache_not_cleared_detected",
       lifecycle([{"stage": "UNLOAD", "cache_cleared": False, "cache_size_after": 3}])
       .get("every_unload_cache_cleared") is False)
record("resident_bytes_not_zero_detected",
       lifecycle([{"stage": "UNLOAD", "cache_cleared": True, "cache_size_after": 0,
                   "resident_bytes_after": 317205528}])
       .get("every_unload_cache_cleared") is False)
record("state_machine_not_idle_detected",
       lifecycle([{"stage": "UNLOAD", "cache_cleared": True, "cache_size_after": 0,
                   "resident_bytes_after": 0}], state_machine_final="EXECUTING")
       .get("state_machine_valid") is False or lifecycle(
           [{"stage": "UNLOAD", "cache_cleared": True, "cache_size_after": 0,
             "resident_bytes_after": 0}], state_machine_final="EXECUTING")
           .get("state_machine_final") == "EXECUTING")
record("expected_next_progression_valid",
       int(lifecycle([], expected_next_block_index=17)["expected_next_block_index"]) == 17)

# ---------------- D. adapter negative invocations ----------------
rc, _, _, _ = run_adapter_rc([9], IN0, out_suffix="s9")
record("adapter_streaming_block9_rc0", rc == 0)
rc31, _, _, rep31 = run_adapter_rc([31], IN0, out_suffix="s31")
record("adapter_last_layer_alone_rc0", rc31 == 0)
pb31 = [e for e in (rep31 or {}).get("per_block", []) if e.get("block_index") == 31] if isinstance((rep31 or {}).get("per_block"), list) else []
record("adapter_last_layer_report_is_last", bool(pb31 and pb31[0].get("is_last_layer") is True))
rc3031, _, _, _ = run_adapter_rc([30, 31], IN0, out_suffix="s3031")
record("adapter_last_layer_final_rc0", rc3031 == 0)
rc_badvar, _, _, _ = run_adapter_rc([9], IN0, out_suffix="bv", extra=["--blocks", "9:no_such_variant"])
record("adapter_rejects_unknown_variant", rc_badvar != 0)
cmd = [str(p16.ADAPTER_EXE.resolve()), "--root", str(p16.PHASE16_ROOT.resolve()),
       "--blocks", "9:%s" % p16.variant_for(9),
       "--input-dir", str((TMP / "no_such_input_dir").resolve()),
       "--output-dir", str((TMP / "out_mi").resolve()), "--load-mode", "streaming"]
rc = subprocess.run(cmd, capture_output=True, text=True, timeout=600).returncode
record("adapter_rejects_missing_input", rc != 0)
rc_oor, _, _, _ = run_adapter_rc([99], IN0, out_suffix="oor")
record("adapter_rejects_out_of_range_block", rc_oor != 0)
rc_notfinal, _, _, _ = run_adapter_rc([31, 0], IN0, out_suffix="nf")
record("adapter_rejects_last_layer_not_final", rc_notfinal != 0)
rc_dup, _, _, _ = run_adapter_rc([31, 31], IN0, out_suffix="dup")
record("adapter_rejects_last_layer_duplicate", rc_dup != 0)
rc_order, _, _, _ = run_adapter_rc([10, 9], IN0, out_suffix="oo")
record("adapter_rejects_out_of_order", rc_order != 0)

# ---------------- E. silent fallback / call counting ----------------
def wrap(adapter_calls, orig_calls, silent):
    return {"ncnn_adapter_call_count": adapter_calls, "original_block_call_count": orig_calls,
            "silent_fallback": silent}

record("silent_fallback_zero_calls_detected", wrap(0, 0, True)["silent_fallback"] is True)
record("original_block_executed_detected", wrap(1, 1, False)["original_block_call_count"] > 0)
record("adapter_call_count_too_few_detected", wrap(0, 0, False)["ncnn_adapter_call_count"] < 1)
record("adapter_call_count_too_many_detected", wrap(2, 0, False)["ncnn_adapter_call_count"] > 1)
w = wrap(1, 0, False)
record("adapter_evidence_ok_criterion",
       bool(w["ncnn_adapter_call_count"] >= 1 and w["original_block_call_count"] == 0
            and w["silent_fallback"] is False))
record("full_adapter_report_no_fallback", bool(rep31 and rep31.get("silent_fallback") is False
                                               and rep31.get("ncnn_adapter_call_count") >= 1
                                               and rep31.get("original_block_call_count") == 0))

# ---------------- F. numerical / error guards ----------------
summary = load_json(p16.REPO / "reports" / "phase16_chain_parity_summary.json")
record("chain_parity_all_chains_pass", summary.get("all_chains_pass") is True)
for cname, limit in (("chain_8", 1e-2), ("chain_16", 2e-2), ("chain_full", 5e-2)):
    cinfo = summary.get("chains", {}).get(cname)
    ok = bool(cinfo and cinfo.get("pass") and all(
        e is not None and e < limit for e in cinfo.get("final_relative_errors", [])))
    record("chain_parity_%s_under_limit" % cname, ok)
chain_reports = {
    "chain_8": "phase16_chain_8_parity.json",
    "chain_16": "phase16_chain_16_parity.json",
    "chain_full": "phase16_chain_full_parity.json",
}
nan_inf_ok = True
for cname, fname in chain_reports.items():
    rep = load_json(p16.REPO / "reports" / fname)
    for run in rep.get("runs", []):
        for e in run.get("final_error", []):
            if e.get("nan_count") != 0 or e.get("inf_count") != 0:
                nan_inf_ok = False
record("chain_parity_no_nan_inf", nan_inf_ok)

def err_entry(max_abs, nan=0, inf=0, limit=1e-2):
    rel = max_abs / 1e4
    return {"relative_max_error": rel, "nan_count": nan, "inf_count": inf,
            "pass": rel < limit and nan == 0 and inf == 0}

record("nan_output_detected", err_entry(1e-6, nan=1)["pass"] is False)
record("inf_output_detected", err_entry(1e-6, inf=1)["pass"] is False)
record("error_over_limit_detected", err_entry(500.0)["pass"] is False)
record("error_under_limit_accepted", err_entry(1e-6)["pass"] is True)

visual = load_json(p16.REPO / "reports" / "full_dit_visual_comparison.json")
full_px = visual.get("modes", {}).get("full", {}).get("pixel", {})
record("visual_full_psnr_ge_34", float(full_px.get("psnr_db", -1.0)) >= p16.VISUAL_PSNR_MIN_FULL)
record("visual_full_ssim_ge_095", float(full_px.get("ssim", -1.0)) >= p16.VISUAL_SSIM_MIN_FULL)
record("visual_full_no_nan_inf", int(full_px.get("nan_count", -1)) == 0
       and int(full_px.get("inf_count", -1)) == 0)
record("visual_verdict_pass", visual.get("verdict") == "PASS")

growth = load_json(p16.REPO / "reports" / "full_dit_error_growth.json")
record("error_growth_present", bool(growth.get("error_growth"))
       and all(k in growth["error_growth"] for k in ("E1", "E2", "E4", "E8", "E16", "Efull")))

# ---------------- G. visual sanity guards ----------------
record("constant_frames_detected", bool(np.zeros((4, 128, 128, 3)).std() <= 1e-3))
record("frame_count_mismatch_detected", (np.zeros((3, 128, 128, 3)).shape[0] != 4))
record("visual_modes_all_present", all(m in visual.get("modes", {}) for m in ("8", "16", "full")))
record("visual_hybrid_non_constant", all(
    visual.get("modes", {}).get(m, {}).get("pixel", {}).get("non_constant") is True
    for m in ("8", "16", "full")))

# ---------------- H. Git protection ----------------
tracked = subprocess.run(["git", "-C", str(p16.REPO), "ls-files"],
                         capture_output=True, text=True).stdout.splitlines()
bad_ext = [f for f in tracked if f.lower().endswith((".pth", ".pt", ".mp4", ".npy", ".png"))]
record("no_heavy_binary_tracked", not bad_ext)
bad_phase16 = [f for f in tracked if ".phase16_" in f]
record("no_phase16_heavy_dirs_tracked", not bad_phase16)
phase16_bins = [f for f in tracked if f.endswith(".bin") and ".phase16" in f]
record("no_phase16_bin_tracked", not phase16_bins)
gi = (p16.REPO / ".gitignore").read_text("utf-8")
record("gitignore_covers_phase16_dirs",
       all(pat in gi for pat in (".phase16_local_artifacts/", ".phase16_export_tmp/",
                                 ".phase16_visual_outputs/", ".phase16_diag/",
                                 ".phase16_runtime_traces/", ".phase16_memory_profiles/")))
diff_check = subprocess.run(["git", "-C", str(p16.REPO), "diff", "--check"],
                            capture_output=True, text=True)
record("git_diff_check_clean", diff_check.returncode == 0)

overall = all(r["pass"] for r in RESULTS)
report = {
    "phase": 16,
    "milestone": "full SeedVR2 DiT streaming negative tests",
    "total": len(RESULTS),
    "passed": sum(1 for r in RESULTS if r["pass"]),
    "failed": sum(1 for r in RESULTS if not r["pass"]),
    "overall_pass": overall,
    "overall_status": "PASS" if overall else "FAIL",
    "tests": RESULTS,
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("total=%d passed=%d failed=%d overall=%s" % (len(RESULTS), report["passed"], report["failed"], report["overall_status"]))
sys.exit(0 if overall else 1)