#!/usr/bin/env python3
"""Materialize the ignored deployment tree with same-volume hardlinks."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / ".phase18c_device_models"
STAGING = BUNDLE / "staging"


def main() -> int:
    index = json.loads((BUNDLE / "deploy_index.json").read_text(encoding="utf-8"))["files"]
    for position, item in enumerate(index, 1):
        source = Path(item["source_path"])
        target = STAGING / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size == source.stat().st_size:
                continue
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            target.write_bytes(source.read_bytes())
        if position % 100 == 0:
            print(f"materialized {position}/{len(index)}", flush=True)
    print(STAGING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
