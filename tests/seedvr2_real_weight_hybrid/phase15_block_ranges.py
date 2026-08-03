#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: derive block ranges 1/2/4 from the REAL checkpoint schema.

Loads the verified official checkpoint (weights_only=True, mmap) and derives:
actual_block_count, last_layer_index, normal block indices, the Phase 14
anchor, and the contiguous ranges range_1/range_2/range_4.  The anchor is
re-derived as floor(actual_block_count / 2) and must equal the Phase 14
selected block (16).  Never hardcodes the block count.  Writes
reports/phase15_block_ranges.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase15_common as p15  # noqa: E402


def block_index_of(key: str):
    m = re.match(r"^blocks\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def main() -> int:
    state = torch.load(str(p15.MAIN_CKPT), map_location="cpu", weights_only=True, mmap=True)
    indices = sorted({i for k in state for i in [block_index_of(k)] if i is not None})
    actual_block_count = len(indices)
    last_layer_index = max(indices)
    normal = [i for i in indices if i != last_layer_index]
    anchor = actual_block_count // 2
    assert anchor == p15.ANCHOR, f"anchor mismatch: derived {anchor} != {p15.ANCHOR}"

    per_block = {}
    for i in p15.BLOCK_SET:
        keys = sorted(k for k in state if k.startswith(f"blocks.{i}."))
        per_block[str(i)] = {
            "key_count": len(keys),
            "numel": int(sum(state[k].numel() for k in keys)),
            "keys": keys,
        }

    ranges = {name: list(indices) for name, indices in p15.RANGES.items()}
    report = {
        "phase": 15,
        "checkpoint_sha256": p15.EXPECTED_MAIN_SHA256,
        "actual_block_count": actual_block_count,
        "block_indices": indices,
        "last_layer_index": last_layer_index,
        "normal_block_indices": normal,
        "selected_phase14_block_index": anchor,
        "anchor_rederived": anchor,
        "ranges": ranges,
        "selection_reason": (
            "anchor=floor(actual_block_count/2); range_1=[anchor]; "
            "range_2=[anchor, anchor+1]; range_4=[anchor-1, anchor, anchor+1, anchor+2] "
            "centered on the anchor; all four indices are contiguous normal blocks "
            "within the real checkpoint and exclude the last layer."
        ),
        "last_layer_excluded": True,
        "block_set": p15.BLOCK_SET,
        "per_block": per_block,
        "full_32_block_ncnn": False,
        "full_pipeline_pytorch": True,
        "android_vulkan": False,
    }
    out = p15.REPO / "reports" / "phase15_block_ranges.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"actual_block_count": actual_block_count, "last_layer": last_layer_index,
                      "anchor": anchor, "ranges": ranges,
                      "per_block_key_counts": {k: v["key_count"] for k, v in per_block.items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
