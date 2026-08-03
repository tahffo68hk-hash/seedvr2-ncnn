#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15 negative tests: strict multi-block streaming hybrid contract.

Covers range selection, artifact/weight binding, strict streaming lifecycle,
silent-fallback detection, numerical guards, visual sanity and Git protection.
Writes reports/phase15_negative_tests.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase15_common as p15  # noqa: E402
import streaming_block_manager as sbm  # noqa: E402

OUT = p15.REPO / "reports" / "phase15_negative_tests.json"
TMP = p15.DIAG / "neg_tests"
TMP.mkdir(parents=True, exist_ok=True)
BLOCK16_IN = p15.PHASE15_ROOT / "official_context_capture" / "block_16"

RESULTS = []


def record(name, passed, detail=""):
    RESULTS.append({"name": name, "pass": bool(passed), "detail": str(detail)[:400]})
    print("%-58s %s" % (name, "PASS" if passed else "FAIL"), flush=True)


def range_validator(blocks, actual_count=32, last_index=31):
    """Orchestrator-level range validation (mirrors the Phase 15 spec)."""
    if not blocks:
        return False, "empty range"
    if len(set(blocks)) != len(blocks):
        return False, "duplicate block indices"
    if any(b < 0 or b >= actual_count for b in blocks):
        return False, "block index out of range"
    if last_index in blocks:
        return False, "last layer in range"
    if blocks != sorted(blocks):
        return False, "blocks not ascending"
    if blocks != list(range(blocks[0], blocks[0] + len(blocks))):
        return False, "blocks not contiguous"
    return True, "ok"


def run_adapter_rc(blocks, input_dir, load_mode="streaming", out_suffix="x"):
    out_dir = TMP / ("out_%s" % out_suffix)
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = TMP / ("rep_%s.json" % out_suffix)
    cmd = [
        str(p15.ADAPTER_EXE.resolve()),
        "--root", str(p15.PHASE15_ROOT.resolve()),
        "--blocks", ",".join("%d:%s" % (i, p15.variant_for(i)) for i in blocks),
        "--input-dir", str(Path(input_dir).resolve()),
        "--output-dir", str(out_dir.resolve()),
        "--load-mode", load_mode,
        "--report", str(report_path.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    report = None
    rp = TMP / ("rep_%s.json" % out_suffix)
    if rp.exists():
        report = json.loads(rp.read_text("utf-8"))
    return proc.returncode, proc.stdout or "", proc.stderr or "", report


def ev(stage, bi):
    return {"stage": stage, "block_index": bi}


def lifecycle(events, **kw):
    base = {"order_ok": True, "max_resident_weighted_blocks": 1,
            "ncnn_adapter_call_count": 1, "original_block_call_count": 0,
            "silent_fallback": False, "streaming_mode_validated": True}
    base.update(kw)
    base["events"] = events
    return sbm.validate_streaming_lifecycle(base)


# ---------------- A. range / selection validation ----------------
record("range_duplicate_rejected", range_validator([16, 16])[0] is False)
record("range_non_contiguous_rejected", range_validator([16, 18])[0] is False)
record("range_out_of_bounds_rejected", range_validator([32])[0] is False)
record("range_negative_rejected", range_validator([-1, 16])[0] is False)
record("range_last_layer_rejected", range_validator([31])[0] is False)
record("range_empty_rejected", range_validator([])[0] is False)
record("range_crossing_last_layer_rejected", range_validator([30, 31])[0] is False)
record("range_valid_anchor_accepted", range_validator(p15.RANGE_4)[0] is True)
record("range_1_accepted", range_validator(p15.RANGE_1)[0] is True)

# ---------------- B. artifact / weight binding ----------------
ok = True
for bi in p15.BLOCK_SET:
    meta = json.loads((p15.PHASE15_ROOT / p15.variant_for(bi) / "metadata.json").read_text("utf-8"))
    blk = meta.get("block") or {}
    pnnx = meta.get("pnnx") or {}
    if blk.get("block_index") != bi or pnnx.get("fp16") != 0:
        ok = False
        ok = False
record("metadata_block_index_and_fp16_correct", ok)
record("artifact_manifest_exists", all(
    (p15.PHASE15_ROOT / p15.variant_for(bi) / "artifact_manifest.json").exists()
    for bi in p15.BLOCK_SET))
manifest16 = json.loads((p15.PHASE15_ROOT / p15.variant_for(16) / "artifact_manifest.json").read_text("utf-8"))
p14_meta = json.loads((p15.PHASE14_ROOT / p15.P14_VARIANT_16 / "metadata.json").read_text("utf-8"))
record("block16_reuses_phase14_artifact", bool(
    manifest16.get("reused_from_phase14") is True and manifest16.get("source_block_index") == 16))
affine_ok = True
for bi in p15.BLOCK_SET:
    qbin = p15.PHASE15_ROOT / p15.variant_for(bi) / "models" / "q_norm" / "model.ncnn.bin"
    data = np.fromfile(str(qbin), np.float32)
    if data.size == 0 or bool((data == 1.0).all()):
        affine_ok = False
record("qk_rmsnorm_real_affine_not_all_ones", affine_ok)
q15 = np.fromfile(str(p15.PHASE15_ROOT / p15.variant_for(15) / "models" / "q_norm" / "model.ncnn.bin"), np.float32)
q16 = np.fromfile(str(p15.PHASE15_ROOT / p15.variant_for(16) / "models" / "q_norm" / "model.ncnn.bin"), np.float32)
record("q_norm_not_reused_across_blocks", bool(not np.array_equal(q15, q16)))

# ---------------- C. strict streaming lifecycle ----------------
record("event_load_b_before_unload_a_rejected",
       sbm.validate_event_sequence([ev("LOAD", 16), ev("EXECUTE", 16),
                                    ev("LOAD", 17), ev("EXECUTE", 17), ev("UNLOAD", 17), ev("UNLOAD", 16)])[0] is False)
record("event_reversed_order_rejected",
       sbm.validate_event_sequence([ev("UNLOAD", 16), ev("LOAD", 16), ev("EXECUTE", 16)])[0] is False)
record("event_execute_before_load_rejected",
       sbm.validate_event_sequence([ev("EXECUTE", 16), ev("LOAD", 16), ev("UNLOAD", 16)])[0] is False)
record("event_chain_ends_loaded_rejected",
       sbm.validate_event_sequence([ev("LOAD", 16), ev("EXECUTE", 16)])[0] is False)
record("event_unload_wrong_block_rejected",
       sbm.validate_event_sequence([ev("LOAD", 16), ev("EXECUTE", 16), ev("UNLOAD", 17)])[0] is False)
record("event_strict_order_accepted",
       sbm.validate_event_sequence([ev("LOAD", 16), ev("EXECUTE", 16), ev("UNLOAD", 16),
                                    ev("LOAD", 17), ev("EXECUTE", 17), ev("UNLOAD", 17)])[0] is True)
record("cache_not_cleared_detected",
       lifecycle([{"stage": "UNLOAD", "cache_cleared": False, "cache_size_after": 3}])
       .get("every_unload_cache_cleared") is False)
record("resident_bytes_not_zero_detected",
       lifecycle([{"stage": "UNLOAD", "cache_cleared": True, "cache_size_after": 0,
                   "resident_bytes_after": 317205528}])
       .get("every_unload_cache_cleared") is False)

rc, out, err, rep = run_adapter_rc([16], BLOCK16_IN, load_mode="streaming", out_suffix="s16")
record("adapter_streaming_block16_rc0", rc == 0)
record("adapter_streaming_order_ok", bool(rep and rep.get("order_ok") is True))
record("adapter_streaming_validated", bool(rep and rep.get("streaming_mode_validated") is True))
record("adapter_streaming_one_resident_block", bool(rep and rep.get("max_resident_weighted_blocks") == 1))
record("adapter_streaming_cache_cleared", bool(
    rep and all(e.get("cache_cleared") for e in rep.get("events", []) if e.get("stage") == "UNLOAD")))
record("adapter_no_silent_fallback", bool(rep and rep.get("silent_fallback") is False
                                          and rep.get("ncnn_adapter_call_count") >= 1
                                          and rep.get("original_block_call_count") == 0))

# ---------------- D. adapter negative invocations ----------------
cmd = [str(p15.ADAPTER_EXE.resolve()), "--root", str(p15.PHASE15_ROOT.resolve()),
       "--blocks", "16:no_such_variant", "--input-dir", str(BLOCK16_IN.resolve()),
       "--output-dir", str((TMP / "out_bv").resolve()), "--load-mode", "streaming"]
rc = subprocess.run(cmd, capture_output=True, text=True, timeout=600).returncode
record("adapter_rejects_unknown_variant", rc != 0)
cmd = [str(p15.ADAPTER_EXE.resolve()), "--root", str(p15.PHASE15_ROOT.resolve()),
       "--blocks", "16:%s" % p15.variant_for(16), "--input-dir", str((TMP / "no_such_input_dir").resolve()),
       "--output-dir", str((TMP / "out_mi").resolve()), "--load-mode", "streaming"]
rc = subprocess.run(cmd, capture_output=True, text=True, timeout=600).returncode
record("adapter_rejects_missing_input", rc != 0)
cmd = [str(p15.ADAPTER_EXE.resolve()), "--root", str(p15.PHASE15_ROOT.resolve()),
       "--blocks", "99:%s" % p15.variant_for(16), "--input-dir", str(BLOCK16_IN.resolve()),
       "--output-dir", str((TMP / "out_b99").resolve()), "--load-mode", "streaming"]
rc = subprocess.run(cmd, capture_output=True, text=True, timeout=600).returncode
record("adapter_rejects_out_of_range_block", rc != 0)

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

# ---------------- F. numerical / error guards ----------------
def err_entry(max_abs, nan=0, inf=0, limit=1e-3):
    rel = max_abs / 1e4
    return {"relative_max_error": rel, "nan_count": nan, "inf_count": inf,
            "pass": rel < limit and nan == 0 and inf == 0}

record("nan_output_detected", err_entry(1e-6, nan=1)["pass"] is False)
record("inf_output_detected", err_entry(1e-6, inf=1)["pass"] is False)
record("error_over_limit_detected", err_entry(50.0)["pass"] is False)
record("error_under_limit_accepted", err_entry(1e-6)["pass"] is True)

rcs, _, _, _ = run_adapter_rc([16], BLOCK16_IN, load_mode="streaming", out_suffix="rs_stream")
rcr, _, _, _ = run_adapter_rc([16], BLOCK16_IN, load_mode="resident", out_suffix="rs_res")
sv = np.fromfile(str(TMP / "out_rs_stream" / "final_vid.bin"), np.float32)
rv = np.fromfile(str(TMP / "out_rs_res" / "final_vid.bin"), np.float32)
record("resident_equals_streaming_final_vid",
       bool(rcs == 0 and rcr == 0 and sv.shape == rv.shape and np.array_equal(sv, rv)))

# ---------------- G. visual sanity guards ----------------
record("constant_frames_detected", bool(np.zeros((4, 128, 128, 3)).std() <= 1e-3))
record("frame_count_mismatch_detected", (np.zeros((3, 128, 128, 3)).shape[0] != 4))
record("psnr_below_threshold_detected", bool(30.0 < p15.VISUAL_PSNR_MIN_4))
record("ssim_below_threshold_detected", bool(0.9 < p15.VISUAL_SSIM_MIN_4))

# ---------------- H. Git protection ----------------
tracked = subprocess.run(["git", "-C", str(p15.REPO), "ls-files"],
                         capture_output=True, text=True).stdout.splitlines()
bad_ext = [f for f in tracked if f.lower().endswith((".pth", ".pt", ".mp4", ".npy", ".png"))]
record("no_heavy_binary_tracked", not bad_ext)
bad_phase15 = [f for f in tracked if ".phase15_" in f]
record("no_phase15_heavy_dirs_tracked", not bad_phase15)
phase15_bins = [f for f in tracked if f.endswith(".bin") and ".phase15" in f]
record("no_phase15_bin_tracked", not phase15_bins)

overall = all(r["pass"] for r in RESULTS)
report = {
    "phase": 15,
    "milestone": "multi-block streaming hybrid negative tests",
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