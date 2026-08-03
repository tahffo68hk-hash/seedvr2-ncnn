#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: streaming hybrid block manager (Python orchestration).

Same role as the Phase 15 streaming_block_manager but drives the Phase 16
full_dit_streaming_adapter (blocks 0..31 including the last layer) with
optional --denoising-step, and validates the richer Phase 16 lifecycle:
strict LOAD/EXECUTE/UNLOAD order, max_resident_weighted_blocks=1, cache
cleared, state machine final IDLE, expected_next_block_index progression.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase16_common as p16  # noqa: E402


def blocks_arg(block_indices):
    return ",".join(f"{i}:{p16.variant_for(i)}" for i in block_indices)


def run_adapter(block_indices, input_dir, output_dir, load_mode="streaming",
                report_path=None, trace_dir=None, memory_profile=False,
                probe_files=True, denoising_step=0, timeout_s=3600):
    """Run the Phase 16 full-chain adapter for one chain."""
    cmd = [
        str(p16.ADAPTER_EXE.resolve()),
        "--root", str(p16.PHASE16_ROOT.resolve()),
        "--blocks", blocks_arg(block_indices),
        "--input-dir", str(Path(input_dir).resolve()),
        "--output-dir", str(Path(output_dir).resolve()),
        "--load-mode", load_mode,
        "--denoising-step", str(denoising_step),
    ]
    if report_path:
        cmd += ["--report", str(Path(report_path).resolve())]
    if trace_dir:
        cmd += ["--trace-dir", str(Path(trace_dir).resolve())]
    if memory_profile:
        cmd.append("--memory-profile")
    if not probe_files:
        cmd.append("--no-file-probe")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    report = None
    if report_path and Path(report_path).exists():
        report = json.loads(Path(report_path).read_text("utf-8"))
    return report, (proc.stdout or ""), proc.returncode


def validate_event_sequence(events):
    """Strict streaming order: LOAD i, EXECUTE i, UNLOAD i, then next LOAD."""
    seen_unload = True
    expected_block = -1
    ok = True
    reason = ""
    for ev in events:
        stage = ev.get("stage")
        bi = ev.get("block_index")
        if stage == "LOAD":
            if not seen_unload:
                ok, reason = False, f"LOAD {bi} before UNLOAD of previous block"
                break
            seen_unload = False
            expected_block = bi
        elif stage == "EXECUTE":
            if expected_block != bi:
                ok, reason = False, f"EXECUTE {bi} out of order (expected {expected_block})"
                break
        elif stage == "UNLOAD":
            if expected_block != bi or seen_unload:
                ok, reason = False, f"UNLOAD {bi} out of order (expected {expected_block})"
                break
            seen_unload = True
    if ok and not seen_unload:
        ok, reason = False, "chain ended with a block still loaded"
    return ok, reason


def validate_streaming_lifecycle(report):
    """Check unload semantics: cache cleared, resident bytes zero, order ok,
    state machine final IDLE and expected_next progression."""
    checks = {
        "order_ok": bool(report.get("order_ok")),
        "max_resident_weighted_blocks": int(report.get("max_resident_weighted_blocks", -1)),
        "ncnn_adapter_call_count": int(report.get("ncnn_adapter_call_count", 0)),
        "original_block_call_count": int(report.get("original_block_call_count", -1)),
        "silent_fallback": bool(report.get("silent_fallback")),
        "streaming_mode_validated": bool(report.get("streaming_mode_validated")),
        "state_machine_final": report.get("state_machine_final"),
        "state_machine_valid": bool(report.get("state_machine_valid")),
        "expected_next_block_index": int(report.get("expected_next_block_index", -1)),
    }
    events = report.get("events", [])
    all_unloaded = True
    for ev in events:
        if ev.get("stage") == "UNLOAD":
            if not ev.get("cache_cleared", False):
                all_unloaded = False
            if ev.get("cache_size_after") != 0:
                all_unloaded = False
            if ev.get("resident_bytes_after") not in (None, 0):
                all_unloaded = False
    checks["every_unload_cache_cleared"] = all_unloaded
    seq_ok, reason = validate_event_sequence(events)
    checks["event_sequence_ok"] = seq_ok
    checks["event_sequence_reason"] = reason
    return checks
