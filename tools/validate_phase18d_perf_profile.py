#!/usr/bin/env python3
"""Validate one production Phase 18D single-tile performance profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


VAE_ENCODER_STAGES = [
    "enc_conv_in", "enc_down0", "enc_down1", "enc_down2",
    "enc_down3", "enc_mid", "enc_out",
]
VAE_DECODER_STAGES = [
    "dec_conv_in", "dec_mid", "dec_up0", "dec_up1",
    "dec_up2", "dec_up3", "dec_out",
]


def nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def validate_profile(path: Path) -> list[str]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"profile missing: {path}"]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"profile unreadable: {exc}"]

    errors: list[str] = []
    required_scalars = [
        "wall_time_ms", "image_prepare_ms", "manifest_validation_ms",
        "model_sha_validation_ms", "model_load_ms", "forward_ms",
        "checkpoint_io_ms", "other_io_ms", "scheduler_ms",
        "rgb_output_ms",
    ]
    if profile.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if profile.get("backend") != "ncnn_cpu_fp32":
        errors.append("backend must be ncnn_cpu_fp32")
    if not isinstance(profile.get("threads"), int) or profile["threads"] <= 0:
        errors.append("threads must be a positive integer")
    for key in required_scalars:
        if not nonnegative_number(profile.get(key)):
            errors.append(f"{key} must be finite and nonnegative")
    for key in ("finite", "silent_fallback"):
        if not isinstance(profile.get(key), bool):
            errors.append(f"{key} must be boolean")
    if profile.get("nan_count") != 0 or profile.get("inf_count") != 0:
        errors.append("nan_count and inf_count must both be zero")

    def validate_stages(key: str, names: list[str]) -> None:
        stages = profile.get(key)
        if not isinstance(stages, list) or len(stages) != len(names):
            errors.append(f"{key} must contain {len(names)} stages")
            return
        if [stage.get("stage") for stage in stages] != names:
            errors.append(f"{key} stage order mismatch")
        for index, stage in enumerate(stages):
            for timing in ("load_ms", "forward_ms", "unload_ms", "total_ms"):
                if not nonnegative_number(stage.get(timing)):
                    errors.append(f"{key}[{index}].{timing} invalid")
            if "rss_before_mb" not in stage or "rss_after_mb" not in stage:
                errors.append(f"{key}[{index}] missing RSS fields")

    validate_stages("encoder_stages", VAE_ENCODER_STAGES)
    validate_stages("decoder_stages", VAE_DECODER_STAGES)

    blocks = profile.get("dit_blocks")
    if not isinstance(blocks, list) or len(blocks) != 32:
        errors.append("dit_blocks must contain exactly 32 entries")
    else:
        if [block.get("block_index") for block in blocks] != list(range(32)):
            errors.append("dit_blocks must be ordered block 0 through 31")
        for index, block in enumerate(blocks):
            for timing in ("load_ms", "forward_ms", "unload_ms", "total_ms"):
                if not nonnegative_number(block.get(timing)):
                    errors.append(f"dit_blocks[{index}].{timing} invalid")
            if "rss_before_mb" not in block or "rss_after_mb" not in block:
                errors.append(f"dit_blocks[{index}] missing RSS fields")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    errors = validate_profile(args.profile)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
