#!/usr/bin/env python3
"""One-command Host entry for real Phase 18D full-photo inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import shutil
import sys
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

MODEL_VERSION = "phase18d-host-product-v1"
SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]+$")
CALIBRATION_SOURCE = "HOST_MEASURED_256_512_1024_2026-08-09"


class CliError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_task_id() -> str:
    return time.strftime("photo_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def read_log_tail(path: Path, limit: int = 65536) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END); size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read().decode("utf-8", errors="replace")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Enhance one photo with the real SeedVR2 ncnn Host runtime")
    result.add_argument("--input"); result.add_argument("--output")
    result.add_argument("--model-root"); result.add_argument("--task-root")
    result.add_argument("--runner", default="build/phase18d-host/Release/phase18d_full_photo_runner.exe")
    result.add_argument("--runner-prefix", action="append", default=[])
    result.add_argument("--runtime-mode", choices=("tile_major", "stage_major", "hybrid", "auto"), default="auto")
    result.add_argument("--tile-cohort-size", type=int, default=0)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--pause-after-dit-block", type=int, default=-1)
    result.add_argument("--cancel", action="store_true")
    result.add_argument("--memory-budget-mb", type=int, default=2048)
    return result


def estimate_resources(width: int, height: int, runtime_mode: str, cohort_size: int,
                       available_disk_bytes: int = (1 << 63) - 1,
                       resume_enabled: bool = False) -> dict:
    columns = ((max(width - 128, 0) + 95) // 96) + 1
    rows = ((max(height - 128, 0) + 95) // 96) + 1
    tiles = columns * rows
    selected = runtime_mode
    selected_k = cohort_size
    reason = "EXPLICIT_RUNTIME_OVERRIDE"
    if runtime_mode == "auto":
        if resume_enabled:
            selected, selected_k, reason = "hybrid", min(4, tiles), "RESUME_REQUIRES_DURABLE_HYBRID"
        elif tiles <= 2:
            selected, selected_k, reason = "tile_major", 0, "MEASURED_SMALL_TASK_TILE_MAJOR"
        elif tiles <= 25:
            selected, selected_k, reason = "stage_major", 0, "MEASURED_MEDIUM_TASK_STAGE_MAJOR"
        else:
            selected, selected_k, reason = "hybrid", min(4, tiles), "MEASURED_LONG_TASK_BOUNDED_DISK"
    final_bytes = tiles * 2_114_073
    def storage(mode: str, k: int) -> tuple[int, int]:
        live_tiles = tiles if mode == "stage_major" else (k if mode == "hybrid" else 1)
        peak = live_tiles * 84_918_726 + final_bytes
        return peak, peak + 1_073_741_824
    peak_bytes, required_bytes = storage(selected, selected_k)
    if runtime_mode == "auto" and selected == "stage_major" and required_bytes > available_disk_bytes:
        selected, selected_k, reason = "hybrid", min(4, tiles), "AVAILABLE_DISK_SELECTED_HYBRID"
        peak_bytes, required_bytes = storage(selected, selected_k)
    while (runtime_mode == "auto" and selected == "hybrid" and selected_k > 1 and
           required_bytes > available_disk_bytes):
        selected_k -= 1
        peak_bytes, required_bytes = storage(selected, selected_k)
        if required_bytes <= available_disk_bytes:
            reason = "AVAILABLE_DISK_REDUCED_HYBRID_COHORT"
    runtime_ms = (15_190.4 * tiles if selected == "tile_major" else
                  41_799.59302326 + 13_743.10465116 * tiles + 8_413.613 *
                  ((((tiles + selected_k - 1) // selected_k) - 1) if selected == "hybrid" else 0))
    return {"tile_count": tiles, "selected_mode": selected, "cohort_size": selected_k,
            "selection_reason": reason, "estimated_peak_task_bytes": peak_bytes,
            "estimated_final_task_bytes": final_bytes,
            "required_free_bytes": required_bytes,
            "estimated_runtime_ms": runtime_ms, "estimated_peak_rss_mb": 1480.41,
            "source_label": CALIBRATION_SOURCE}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.cancel:
        if not args.task_root: raise CliError("TASK_ROOT_REQUIRED_FOR_CANCEL")
        task_root = Path(args.task_root).resolve()
        if not task_root.is_dir(): raise CliError("TASK_NOT_FOUND")
        payload = {"schema_version": 1, "action": "cancel",
                   "requested_unix_ns": time.time_ns()}
        atomic_write(task_root / "runtime_control.json",
                     (json.dumps(payload, separators=(",", ":")) + "\n").encode())
        print(f"Cancel requested at next committed boundary: {task_root}", flush=True)
        return 0
    if not args.input or not args.output or not args.model_root:
        raise CliError("INPUT_OUTPUT_MODEL_REQUIRED")
    source = Path(args.input).resolve(); output = Path(args.output).resolve()
    model_root = Path(args.model_root).resolve()
    if not source.is_file(): raise CliError("INPUT_NOT_FOUND")
    manifest = model_root / "manifest.json"
    if not manifest.is_file(): raise CliError("MODEL_MANIFEST_NOT_FOUND")
    if args.runtime_mode == "hybrid" and args.tile_cohort_size <= 0:
        raise CliError("INVALID_COHORT_SIZE")
    task_root = Path(args.task_root).resolve() if args.task_root else (
        Path(".phase18d_host_run/tasks") / new_task_id()).resolve()
    task_id = task_root.name
    if not SAFE_TASK_ID.fullmatch(task_id): raise CliError("INVALID_TASK_ID")

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size; rgb = image.tobytes()
    identity = {
        "schema_version": 1, "task_id": task_id, "model_version": MODEL_VERSION,
        "source_image_sha256": sha256_file(source), "normalized_image_sha256": sha256_bytes(rgb),
        "model_manifest_sha256": sha256_file(manifest), "width": width, "height": height,
        "runtime_mode": args.runtime_mode, "tile_cohort_size": args.tile_cohort_size,
    }
    identity_path = task_root / "task_identity.json"
    if task_root.exists() and any(task_root.iterdir()):
        if not args.resume: raise CliError("TASK_EXISTS_USE_RESUME")
        if not identity_path.is_file() or json.loads(identity_path.read_text("utf-8")) != identity:
            raise CliError("TASK_IDENTITY_MISMATCH")
    else:
        task_root.mkdir(parents=True, exist_ok=True)
        atomic_write(identity_path, (json.dumps(identity, indent=2) + "\n").encode())
    rgb_path = task_root / "normalized.rgb"
    if not args.resume or not rgb_path.is_file(): atomic_write(rgb_path, rgb)

    columns = ((max(width - 128, 0) + 95) // 96) + 1
    rows = ((max(height - 128, 0) + 95) // 96) + 1
    tile_count = columns * rows
    free_bytes = shutil.disk_usage(task_root.parent).free
    resource = estimate_resources(width, height, args.runtime_mode, args.tile_cohort_size,
                                  free_bytes, args.resume)
    print(f"Estimate [ESTIMATED from {resource['source_label']}]: "
          f"runtime={resource['estimated_runtime_ms']/1000:.1f}s "
          f"peak_disk={resource['estimated_peak_task_bytes']/1073741824:.2f}GiB "
          f"final_disk={resource['estimated_final_task_bytes']/1073741824:.2f}GiB "
          f"peak_rss={resource['estimated_peak_rss_mb']:.2f}MiB "
          f"free={free_bytes/1073741824:.2f}GiB", flush=True)
    if args.runtime_mode == "auto":
        print(f"Auto plan: {resource['selected_mode']}" +
              (f" K={resource['cohort_size']}" if resource['selected_mode'] == "hybrid" else "") +
              f" reason={resource['selection_reason']}", flush=True)
    if resource["required_free_bytes"] > free_bytes: raise CliError("INSUFFICIENT_ESTIMATED_DISK")
    if args.memory_budget_mb <= 0 or resource["estimated_peak_rss_mb"] > args.memory_budget_mb:
        raise CliError("INSUFFICIENT_MEMORY_BUDGET")
    request = dict(identity)
    request.update({"max_pixels": 16_000_000, "max_tiles": 2048,
                    "encoder_threads": 5, "dit_threads": 5, "decoder_threads": 5,
                    "resume": bool(args.resume),
                    "memory_budget_bytes": args.memory_budget_mb * 1024 * 1024})
    if args.pause_after_dit_block >= 0: request["stop_after_dit_block"] = args.pause_after_dit_block
    print(f"Input: {width}x{height}\nTiles: {tile_count}\nRuntime: {args.runtime_mode}" +
          (f" K={args.tile_cohort_size}" if args.runtime_mode == "hybrid" else ""), flush=True)
    command = [args.runner, *args.runner_prefix, str(model_root), str(task_root), str(rgb_path),
               str(width), str(height), json.dumps(request, separators=(",", ":"))]
    started = time.monotonic()
    stdout_path = task_root / "native_stdout.log"
    stderr_path = task_root / "native_stderr.log"
    progress_path = task_root / "runtime_progress.json"; last_progress = None
    with stdout_path.open("w", encoding="utf-8") as stdout_log, \
         stderr_path.open("w", encoding="utf-8") as stderr_log:
        process = subprocess.Popen(command, text=True, stdout=stdout_log, stderr=stderr_log)
        while process.poll() is None:
            try:
                progress = json.loads(progress_path.read_text("utf-8"))
                signature = (progress.get("phase"), progress.get("stage_name"),
                             progress.get("dit_block_index"), progress.get("tile_index", progress.get("model_files_done")),
                             progress.get("tile_count", progress.get("model_files_total")))
                if progress.get("state") == "running" and signature != last_progress:
                    phase, stage, block, tile, total = signature
                    print(f"Progress: {phase}/{stage} block={block} tile={tile}/{total}", flush=True)
                    last_progress = signature
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            time.sleep(0.1)
        process.wait()
    stdout = read_log_tail(stdout_path)
    stderr = read_log_tail(stderr_path)
    if stdout: print(stdout.strip(), flush=True)
    if process.returncode != 0:
        raise CliError(f"NATIVE_RUN_FAILED: {stderr.strip() or stdout.strip()}")
    report_path = task_root / "task_report.json"
    if not report_path.is_file(): raise CliError("TASK_REPORT_MISSING")
    report = json.loads(report_path.read_text("utf-8"))
    if report.get("state") != "completed" or report.get("silent_fallback") or not report.get("finite"):
        raise CliError("TASK_REPORT_NOT_COMPLETED")
    output_rgb = (task_root / "output_full.rgb").read_bytes()
    if len(output_rgb) != width * height * 3: raise CliError("OUTPUT_RGB_SIZE_MISMATCH")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", (width, height), output_rgb).save(output, format="PNG")
    print(f"Output: {output}\nTask: {task_root}\nWall time: {time.monotonic()-started:.3f}s\nSHA: {sha256_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except CliError as error:
        print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(1)
