#!/usr/bin/env python3
"""Classify actual Phase 18D task bytes by recoverable lifecycle category."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

CATEGORIES = ("source", "normalized_rgb", "tile_inputs", "vae_latent", "dit_state",
              "scheduler_state", "decoder_state", "tile_outputs", "fusion_outputs",
              "checkpoint_metadata", "debug_gates", "runtime_metadata", "other")

def category(relative: Path) -> str:
    name=relative.name; text=relative.as_posix()
    if name.startswith("gate_") or "debug" in name: return "debug_gates"
    if text.startswith("checkpoints/") or name.startswith("stage_major_checkpoint"): return "checkpoint_metadata"
    if name.startswith("source_image") or name=="input.png": return "source"
    if name=="normalized.rgb": return "normalized_rgb"
    if name in ("tile_input.rgb","input_norm.f32"): return "tile_inputs"
    if name.startswith("state_encoder_"): return "vae_latent"
    if name.startswith("state_dit_"): return "dit_state"
    if name.startswith("state_scheduler"): return "scheduler_state"
    if name.startswith("state_decoder_"): return "decoder_state"
    if name in ("output_valid.rgb","status.json") and text.startswith("tiles/"): return "tile_outputs"
    if name in ("output_full.rgb","output.png","side_by_side.png","diff.png","preview.png"): return "fusion_outputs"
    if name.endswith(".json"): return "runtime_metadata"
    return "other"

def analyze_task(root: Path):
    root=Path(root); totals=defaultdict(int); total=0; tile_bytes=0
    tile_dirs=set()
    for path in root.rglob("*"):
        if not path.is_file(): continue
        size=path.stat().st_size; relative=path.relative_to(root); total+=size
        totals[category(relative)]+=size
        parts=relative.parts
        if len(parts)>=2 and parts[0]=="tiles" and parts[1].startswith("tile_"):
            tile_dirs.add(parts[1]); tile_bytes+=size
    return {"schema_version":1,"task_root":str(root.resolve()),"total_bytes":total,
            "tile_count":len(tile_dirs),"bytes_per_tile":tile_bytes//len(tile_dirs) if tile_dirs else 0,
            "categories":{name:totals[name] for name in CATEGORIES}}

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("task_root"); parser.add_argument("--output")
    args=parser.parse_args(argv); report=analyze_task(Path(args.task_root)); text=json.dumps(report,indent=2)
    if args.output: Path(args.output).write_text(text+"\n",encoding="utf-8")
    print(text); return 0

if __name__=="__main__": raise SystemExit(main())
