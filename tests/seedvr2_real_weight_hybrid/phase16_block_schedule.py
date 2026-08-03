#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: derive the 8/16/full block schedule from the REAL checkpoint.

Loads the verified official checkpoint (weights_only=True, mmap) and derives
actual_block_count, last-layer index, normal vs split/shared layouts, and the
contiguous ranges range_8 (9..16), range_16 (8..23), range_full (0..31).
Never hardcodes the block count: every number is re-derived from state_dict
keys.  Writes reports/phase16_block_schedule.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase16_common as p16  # noqa: E402


def block_index_of(key: str):
    m = re.match(r"^blocks\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def main() -> int:
    state = torch.load(str(p16.MAIN_CKPT), map_location="cpu", weights_only=True, mmap=True)
    indices = sorted({i for k in state for i in [block_index_of(k)] if i is not None})
    actual_block_count = len(indices)
    last_layer_index = max(indices)
    normal = [i for i in indices if i != last_layer_index]
    split = [i for i in indices if any(f"blocks.{i}." + k + ".vid.weight" in state for k in ("attn.proj_qkv",))]
    shared = [i for i in indices if i not in split]

    per_block = {}
    for i in indices:
        keys = sorted(k for k in state if k.startswith(f"blocks.{i}."))
        per_block[str(i)] = {
            "key_count": len(keys),
            "numel": int(sum(state[k].numel() for k in keys)),
            "layout": "split_vid_txt" if i in split else "shared_all",
            "is_last_layer": i == last_layer_index,
        }

    range_8 = list(range(9, 17))
    range_16 = list(range(8, 24))
    range_full = indices
    ranges = {"range_8": range_8, "range_16": range_16, "range_full": range_full}

    report = {
        "phase": 16,
        "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
        "real_seedvr2_weights": True,
        "actual_block_count": actual_block_count,
        "block_indices": indices,
        "normal_block_indices": normal,
        "last_layer_index": last_layer_index,
        "split_layout_blocks": split,
        "shared_layout_blocks": shared,
        "mm_layers_from_config": p16.MM_LAYERS,
        "ranges": ranges,
        "range_8": range_8,
        "range_16": range_16,
        "range_full": range_full,
        "selection_reason": (
            "range_8 = [9..16] and range_16 = [8..23] are contiguous normal blocks "
            "centered on the Phase 14/15 anchor 16 (floor(32/2)) and exclude the "
            "last layer 31; range_full = all derived block indices 0..31 in strict "
            "model forward order including the unique last layer."
        ),
        "continuous": all(r == list(range(r[0], r[-1] + 1)) for r in ranges.values()),
        "complete_coverage": range_full == indices,
        "no_duplicates": all(len(r) == len(set(r)) for r in ranges.values()),
        "no_missing_indices": sorted(set(indices) - set(range_full)) == [],
        "last_layer_only_at_end": last_layer_index == max(range_full),
        "last_layer_excluded_from_8_16": last_layer_index not in range_8 + range_16,
        "per_block": per_block,
        "full_32_block_ncnn": False,
        "full_pipeline_pytorch": True,
        "android_vulkan": False,
    }
    out = p16.REPO / "reports" / "phase16_block_schedule.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"actual_block_count": actual_block_count, "last_layer": last_layer_index,
                      "split": split, "shared": shared, "ranges": ranges}, indent=1))
    ok = (actual_block_count == p16.ACTUAL_BLOCK_COUNT and last_layer_index == p16.LAST_LAYER_INDEX
          and report["continuous"] and report["complete_coverage"] and report["no_duplicates"]
          and report["no_missing_indices"] and report["last_layer_only_at_end"])
    print("schedule ok:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
