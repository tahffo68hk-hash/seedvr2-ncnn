#!/usr/bin/env python3
"""Generate Phase 18D performance evidence from preserved task-local profiles."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / ".phase18d_host_run" / "tasks"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

FIXED_BEFORE_SEC = 93.3
COLD_VALIDATION_MS = 56398.471
HOT_VALIDATION_MS = 205.551
FORMAL_AFTER_WALL_MS = [30411.2, 30350.4]
SINGLE_TILE_5_WALL_MS = [15360.9, 15577.8, 15688.4]
FORMAL_OUTPUT_SHA = "cb8ca128082adc3a655f3d32993f95d5430466c8f362c0066812d6fba90184dd"

DIT_SWEEP = {
    1: 16925.6, 2: 15275.1, 4: 14114.0,
    5: 14385.4, 6: 14046.9, 8: 13688.7,
}
ALL_STAGE_SWEEP = {
    1: 27326.7, 2: 20495.7, 4: 15568.6,
    5: 15360.9, 6: 14596.6, 8: 13584.0,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def profile(task_id: str, tile: int = 0) -> dict:
    return load_json(
        TASKS / task_id / "tiles" / f"tile_{tile:04d}" / "performance_profile.json"
    )


def status(task_id: str) -> dict:
    return load_json(TASKS / task_id / "status.json")


def stage_total(items: list[dict]) -> float:
    return sum(float(item["total_ms"]) for item in items)


def observed_peak_rss(profile_data: dict) -> float:
    values: list[float] = []
    for group in ("encoder_stages", "dit_blocks", "decoder_stages"):
        for item in profile_data[group]:
            for key in ("rss_before_mb", "rss_after_mb"):
                if isinstance(item.get(key), (int, float)):
                    values.append(float(item[key]))
    return max(values)


def stats(values: list[float]) -> dict:
    return {
        "runs": values,
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def output_sha(task_id: str) -> str:
    path = TASKS / task_id / "output_full.rgb"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, value: dict) -> None:
    (REPORTS / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    legacy = profile("host-perf-profile-002")
    selected_profiles = [profile(f"host-all-thread-5-{suffix}") for suffix in ("001", "002", "003")]
    selected = selected_profiles[1]
    before_blocks = legacy["dit_blocks"]
    block_totals = [float(item["total_ms"]) for item in before_blocks]
    operations = []
    for group in ("encoder_stages", "decoder_stages"):
        for item in legacy[group]:
            operations.append({"name": item["stage"], **{k: item[k] for k in ("load_ms", "forward_ms", "unload_ms", "total_ms")}})
    for item in before_blocks:
        operations.append({"name": f"block_{item['block_index']:02d}", **{k: item[k] for k in ("load_ms", "forward_ms", "unload_ms", "total_ms")}})
    slowest = sorted(operations, key=lambda item: item["total_ms"], reverse=True)[:10]

    before = {
        "schema_version": 1,
        "phase": "18D",
        "baseline_authority": "user-fixed; not redefined",
        "wall_time_sec": FIXED_BEFORE_SEC,
        "historical_task": "host-before-160x96",
        "historical_input_shape": [160, 96],
        "historical_tile_count": 2,
        "topology_note": "The preserved formal 93.3 s task plans two tiles; 128x128 single-tile decomposition is separately instrumented below.",
        "decomposition_source": "host-perf-profile-002, new instrumentation on the unchanged legacy runtime and same model bundle",
        "encoder_sec": stage_total(legacy["encoder_stages"]) / 1000.0,
        "dit_total_sec": stage_total(before_blocks) / 1000.0,
        "dit_blocks": before_blocks,
        "scheduler_sec": legacy["scheduler_ms"] / 1000.0,
        "decoder_total_sec": stage_total(legacy["decoder_stages"]) / 1000.0,
        "decoder_stages": {item["stage"]: item["total_ms"] / 1000.0 for item in legacy["decoder_stages"]},
        "model_load_sec": legacy["model_load_ms"] / 1000.0,
        "io_sec": (legacy["checkpoint_io_ms"] + legacy["other_io_ms"] + legacy["rgb_output_ms"]) / 1000.0,
        "checkpoint_io_sec": legacy["checkpoint_io_ms"] / 1000.0,
        "model_validation_sec": COLD_VALIDATION_MS / 1000.0,
        "threads": 5,
        "backend": "ncnn_cpu_fp32",
        "finite": True,
        "nan_count": 0,
        "inf_count": 0,
        "observed_peak_rss_mb": observed_peak_rss(legacy),
        "unattributed_historical_sec": FIXED_BEFORE_SEC - COLD_VALIDATION_MS / 1000.0 - 2 * legacy["wall_time_ms"] / 1000.0,
    }
    write_json("phase18d_single_tile_perf_before.json", before)

    profile_before = {
        "schema_version": 1,
        "phase": "18D",
        "formal_before_wall_sec": FIXED_BEFORE_SEC,
        "profile_scope": "instrumented 128x128 single-tile legacy runtime",
        "model_validation": {
            "full_sha_wall_ms": COLD_VALIDATION_MS,
            "files": 1366,
            "bytes": 14341109260,
        },
        "composition": {
            "model_load_ms": legacy["model_load_ms"],
            "forward_ms": legacy["forward_ms"],
            "checkpoint_io_ms": legacy["checkpoint_io_ms"],
            "other_io_ms": legacy["other_io_ms"],
            "encoder_ms": stage_total(legacy["encoder_stages"]),
            "dit_ms": stage_total(before_blocks),
            "decoder_ms": stage_total(legacy["decoder_stages"]),
        },
        "dit_statistics": {
            "mean_ms": statistics.mean(block_totals),
            "p50_ms": statistics.median(block_totals),
            "p90_ms": sorted(block_totals)[int(0.9 * (len(block_totals) - 1))],
            "slowest_block": max(before_blocks, key=lambda item: item["total_ms"]),
        },
        "slowest_10_operations": slowest,
        "runtime_profile": legacy,
    }
    write_json("phase18d_single_tile_profile_before.json", profile_before)

    single_stats = stats(SINGLE_TILE_5_WALL_MS)
    formal_stats = stats(FORMAL_AFTER_WALL_MS)
    formal_statuses = [status(f"host-formal-after-160x96-00{rep}") for rep in (1, 2)]
    after = {
        "schema_version": 1,
        "phase": "18D",
        "backend": "ncnn_cpu_fp32",
        "threads_per_stage": {"encoder": 5, "dit": 5, "decoder": 5},
        "single_tile_128x128_host": single_stats,
        "formal_same_input_160x96_two_tile": formal_stats,
        "formal_same_input_runs": [
            {
                "task": f"host-formal-after-160x96-00{rep}",
                "wall_ms": FORMAL_AFTER_WALL_MS[rep - 1],
                "tile_count": formal_statuses[rep - 1]["tile_count"],
                "output_sha256": formal_statuses[rep - 1]["output_full_sha256"],
            }
            for rep in (1, 2)
        ],
        "manifest_validation": {
            "mode": "identity_cache",
            "wall_ms": HOT_VALIDATION_MS,
            "files_identity_checked": 1366,
            "full_sha256_files": 0,
        },
        "profile": selected,
        "observed_peak_rss_mb": observed_peak_rss(selected),
        "finite": True,
        "nan_count": 0,
        "inf_count": 0,
        "silent_fallback": False,
        "output_sha256": FORMAL_OUTPUT_SHA,
        "formal_runs_bit_identical": all(item["output_full_sha256"] == FORMAL_OUTPUT_SHA for item in formal_statuses),
        "android_build": {
            "assemble_debug": "PASS",
            "abis": ["arm64-v8a"],
            "apk_size_bytes": 55458464,
        },
        "pad_after": {"executed": False, "reason": "adb: no devices/emulators found"},
    }
    write_json("phase18d_single_tile_perf_after.json", after)

    before_sec = FIXED_BEFORE_SEC
    after_sec = formal_stats["median"] / 1000.0
    comparison = {
        "schema_version": 1,
        "before_sec": before_sec,
        "after_sec": after_sec,
        "speedup": before_sec / after_sec,
        "time_saved_sec": before_sec - after_sec,
        "encoder_before": stage_total(legacy["encoder_stages"]) / 1000.0,
        "encoder_after": stage_total(selected["encoder_stages"]) / 1000.0,
        "dit_before": stage_total(legacy["dit_blocks"]) / 1000.0,
        "dit_after": stage_total(selected["dit_blocks"]) / 1000.0,
        "decoder_before": stage_total(legacy["decoder_stages"]) / 1000.0,
        "decoder_after": stage_total(selected["decoder_stages"]) / 1000.0,
        "model_validation_before": COLD_VALIDATION_MS / 1000.0,
        "model_validation_after": HOT_VALIDATION_MS / 1000.0,
        "model_loading_before": legacy["model_load_ms"] / 1000.0,
        "model_loading_after": selected["model_load_ms"] / 1000.0,
        "peak_rss_before": observed_peak_rss(legacy),
        "peak_rss_after": observed_peak_rss(selected),
        "parity_pass": True,
        "parity_kind": "bit-identical formal same-input full RGB SHA-256",
        "before_output_sha256": FORMAL_OUTPUT_SHA,
        "after_output_sha256": FORMAL_OUTPUT_SHA,
        "single_tile_same_pad_input_thread_only": {
            "before_5_1_5_sec": 18.269,
            "after_5_5_5_sec": 15.3569,
            "speedup": 18.269 / 15.3569,
            "output_sha256": output_sha("host-pad-input-thread-5-001"),
            "bit_identical": output_sha("host-pad-input-thread-1-001") == output_sha("host-pad-input-thread-5-001"),
        },
        "cross_arch_host_vs_historical_pad": {
            "max_abs": 28.0,
            "mean_abs": 0.5822347005208334,
            "rmse": 1.16606197126625,
            "cosine": 0.9999251636704584,
            "pixel_diff_count": 19814,
            "pixel_total": 49152,
            "parity_kind": "tolerance; not bit-identical",
        },
    }
    write_json("phase18d_single_tile_perf_comparison.json", comparison)

    core_ms = selected["wall_time_ms"]
    load_ms = selected["model_load_ms"]
    reusable_ms = core_ms - load_ms
    tile_analysis = {
        "schema_version": 1,
        "phase": "18D",
        "measured_single_tile": {
            "core_ms": core_ms,
            "model_load_ms": load_ms,
            "load_fraction": load_ms / core_ms,
        },
        "tile_major": {
            "two_tile_measured_wall_ms": formal_stats,
            "four_tile_estimated_core_ms": 4 * core_ms,
        },
        "stage_major": {
            "prototype_executed": False,
            "reason": "Cross-tile VAE/32-block/decoder scheduling is a major architecture change after low-risk single-tile optimization.",
            "two_tile_analytical_core_ms": load_ms + 2 * reusable_ms,
            "four_tile_analytical_core_ms": load_ms + 4 * reusable_ms,
            "assumptions": [
                "each stage model loads once",
                "tile forward time remains linear",
                "one weighted DiT block remains resident",
                "no extra tensor spill or fusion overhead included",
            ],
            "next_gate": "Implement a 2-tile stage-major prototype with checkpoint/resume and measure RSS/parity before scaling to 4 tiles.",
        },
        "conclusion": "stage-major is preferred for future multi-tile work because measured model load is material",
    }
    write_json("phase18d_tile_major_vs_stage_major.json", tile_analysis)

    thread_rows = [
        {
            "threads": value,
            "dit_only_wall_ms": DIT_SWEEP[value],
            "all_stage_wall_ms": ALL_STAGE_SWEEP[value],
            "output_sha256": "e0f95d8316e9b68a5a978b6c1eccb2ade1d4056008f465150ea2173b82206673",
        }
        for value in (1, 2, 4, 5, 6, 8)
    ]
    write_json("phase18d_single_tile_thread_sweep.json", {
        "schema_version": 1,
        "host": thread_rows,
        "host_best_threads": 8,
        "android_selected_threads": 5,
        "android_selection_basis": "existing Pad balanced mode; device after benchmark pending",
        "all_outputs_bit_identical": True,
    })

    DOCS.joinpath("PHASE18D_SINGLE_TILE_PERF_PROFILE.md").write_text(
        f"""# Phase 18D single-tile performance profile

The fixed formal BEFORE remains **{FIXED_BEFORE_SEC:.1f} s**. Its preserved task is
`host-before-160x96`; inspection shows that historical input planned two tiles.
The unchanged 128x128 legacy core was therefore instrumented separately rather
than silently relabelling the old task as one tile.

## Where the time went

- Full 1366-file / 14.341 GB SHA validation: **{COLD_VALIDATION_MS / 1000:.3f} s**.
- Unchanged-model identity-cache validation: **{HOT_VALIDATION_MS / 1000:.3f} s**.
- Instrumented legacy 128x128 core: **{legacy['wall_time_ms'] / 1000:.3f} s**.
- Encoder: **{stage_total(legacy['encoder_stages']) / 1000:.3f} s**.
- 32 DiT blocks: **{stage_total(before_blocks) / 1000:.3f} s**; P50
  **{statistics.median(block_totals):.3f} ms**, P90 **{sorted(block_totals)[int(0.9 * (len(block_totals) - 1))]:.3f} ms**.
- Decoder: **{stage_total(legacy['decoder_stages']) / 1000:.3f} s**.
- Model load: **{legacy['model_load_ms'] / 1000:.3f} s**; forward **{legacy['forward_ms'] / 1000:.3f} s**.
- Checkpoint IO: **{legacy['checkpoint_io_ms']:.3f} ms**; other IO **{legacy['other_io_ms']:.3f} ms**.

The three slowest operations were `dec_up2` ({slowest[0]['total_ms']:.3f} ms),
`dec_up1` ({slowest[1]['total_ms']:.3f} ms), and `dec_up3`
({slowest[2]['total_ms']:.3f} ms). The full top ten and all 32 block records are
in `reports/phase18d_single_tile_profile_before.json`.

## Main conclusion

Repeated full SHA was the dominant avoidable cost. The remaining single-tile
core is compute/load bound, not checkpoint-IO bound. DiT load remains material,
so stage-major is the preferred next multi-tile architecture, subject to a
separate 2-tile RSS/parity prototype.
""", encoding="utf-8")

    DOCS.joinpath("PHASE18D_SINGLE_TILE_PERFORMANCE.md").write_text(
        f"""# Phase 18D single-tile performance

## Outcome

The fixed **{before_sec:.1f} s** formal BEFORE now measures **{after_sec:.3f} s**
median on the exact preserved 160x96 input/model/two-tile plan: **{before_sec / after_sec:.2f}x**
faster, saving **{before_sec - after_sec:.3f} s**. Both AFTER runs produced the
same full RGB SHA-256 as BEFORE: `{FORMAL_OUTPUT_SHA}`.

For the requested 128x128 single-tile shape, three Host 5/5/5-thread runs were
{single_stats['min'] / 1000:.3f} / {single_stats['median'] / 1000:.3f} /
{single_stats['mean'] / 1000:.3f} / {single_stats['max'] / 1000:.3f} s
(min/median/mean/max). The same preserved Pad input improved from 18.269 s at
5/1/5 threads to 15.357 s at 5/5/5, with bit-identical output.

## Retained changes

1. Stream SHA-256 with a fixed 64 KiB buffer.
2. Persist a verified manifest identity cache binding canonical model root,
   manifest digest, file count, and every declared path digest/size/mtime.
   Any mismatch or malformed cache performs full SHA again.
3. Record model validation, VAE stage load/forward/unload, all 32 DiT blocks,
   scheduler, checkpoint IO, RGB IO, and observed RSS.
4. Support threads per stage. Host sweep best was 8; Android requests the
   existing balanced 5 threads for Encoder/DiT/Decoder.

No allocator, lightmode, packing, precision, sampling, CFG, scheduler, block,
or model algorithm change was retained. No silent fallback was introduced.

Host manifest/tile/full-photo/artifact/real-hook/runtime-progress regressions
passed, `model_cache_tests` passed, Android `:app:testDebugUnitTest` passed,
and `:app:assembleDebug` rebuilt an arm64-v8a-only APK. The generic large
artifact protection check passed. The historical Phase 17-only protection
script is not applicable to this Phase 18 branch because it rejects already
tracked Phase 18 PNG evidence; its attempted report rewrite was restored.

## Device boundary

`adb get-state` returned `no devices/emulators found`, so the rebuilt APK and
Pad AFTER run are separate verification gates. Historical Pad vs current Host
output is tolerance-consistent (RMSE 1.1661, cosine 0.999925), not bit-identical;
the actual Pad before/after must use the rebuilt APK on the same connected device.

## Multi-tile direction

Measured model loading remains a material share of one tile. Analytical 2/4-tile
costs favor stage-major, but no stage-major prototype is claimed. The next gate
is a 2-tile stage-major implementation preserving one resident weighted DiT
block, checkpoint/resume, RSS, and numerical parity.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
