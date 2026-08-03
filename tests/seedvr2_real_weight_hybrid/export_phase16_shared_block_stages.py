#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: stream-export REAL-weight shared-layout blocks to FP32 ncnn.

Exports blocks 10..14 and 19..31 (15-key .all. shared layout) with the same
12-unique-stage contract as Phase 15, one graph at a time
(`pnnx ... fp16=0`).  Block 31 is the unique last layer: its metadata carries
is_last_layer=true (vid_only MLP semantics handled by the runtime); the txt
swiglu stages point at the same shared physical artifacts and are never
executed.  Weight-free stages (attn_norm, mlp_norm, rope_apply, swiglu_act,
head) are hardlinked from the Phase 15 block-16 artifact (bin=0, same graph
contract).  Outputs land in `.phase16_local_artifacts/` (git-ignored).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_ncnn"))

import phase16_common as p16  # noqa: E402
from export_synthetic_dit_block import (  # noqa: E402
    LinearUnit, RMSNormUnit, RopeApplyUnit, HeadUnit, SwigluActUnit,
    trace_and_save, run_pnnx, make_gemm_dynamic, write_raw,
)
from export_real_weight_block_stages import (  # noqa: E402
    sha256, cleanup_stage, stream_linear, stream_rms,
)

C, HD, MLP = p16.C, p16.HD, p16.MLP_HIDDEN
Q_MAX, PAIRS = p16.Q_MAX, p16.PAIRS


def link_weight_free(src_root: Path, dst_models: Path) -> None:
    """Hardlink the 5 weight-free stages from the Phase 15 block-16 artifact."""
    for tag in ("attn_norm", "mlp_norm", "rope_apply", "swiglu_act", "head"):
        sdir = src_root / "models" / tag
        ddir = dst_models / tag
        ddir.mkdir(parents=True, exist_ok=True)
        for suffix in ("model.ncnn.param", "model.ncnn.bin", "reference_output.bin"):
            s = sdir / suffix
            if not s.exists():
                continue
            d = ddir / suffix
            if d.exists():
                d.unlink()
            os.link(str(s), str(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-index", type=int, required=True)
    ap.add_argument("--pnnx", type=Path, default=p16.PNNX_EXE)
    ap.add_argument("--libtorch", type=Path, default=p16.LIBTORCH)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    bi = args.block_index
    if bi not in p16.SHARED_BLOCKS:
        raise SystemExit(f"block {bi} is not a shared-layout block (10..31)")
    if bi in p16.P15_EXISTING:
        raise SystemExit(f"block {bi} already exists from Phase 15; use hardlink reuse, not export")
    variant = p16.variant_for(bi)
    is_last_layer = (bi == p16.LAST_LAYER_INDEX)
    root = args.root or (p16.PHASE16_ROOT / variant)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    B = f"blocks.{bi}."
    stage_log = {}

    # Weight-free stages: hardlink from Phase 15 block-16 (bin=0, same contract).
    link_weight_free(p16.PHASE15_ROOT / p16.variant_for(16), models)
    for tag in p16.WEIGHT_FREE_STAGES:
        info = cleanup_stage(models / tag)
        info["stage"] = tag
        info["export_time_s"] = 0.0
        info["source_params"] = []
        info["weight_free_shared"] = True
        stage_log[tag] = info

    sd = torch.load(str(p16.MAIN_CKPT), map_location="cpu", weights_only=True, mmap=True)

    def key(name):
        return sd[B + name].detach().cpu().contiguous().float().clone()

    def done(tag):
        if args.skip_existing and (models / tag / "model.ncnn.bin").exists():
            info = cleanup_stage(models / tag)
            info["stage"] = tag
            info["export_time_s"] = 0.0
            info["source_params"] = []
            stage_log[tag] = info
            return True
        return False

    # 1. qkv: real [7680,2560] shared weight (bias=False).
    if not done("qkv"):
        stream_linear("qkv", key("attn.proj_qkv.all.weight"), None, 8, f"[8,{C}]",
                      models / "qkv", args.pnnx, args.libtorch,
                      [f"{B}attn.proj_qkv.all.weight"], stage_log)

    # 2. q/k RMSNorm: real affine (never folded, block-specific).
    for tag, name in (("q_norm", "attn.norm_q.all.weight"), ("k_norm", "attn.norm_k.all.weight")):
        if done(tag):
            continue
        stream_rms(tag, HD, 1e-5, key(name), 8, f"[8,{HD}]",
                   models / tag, args.pnnx, args.libtorch, [f"{B}{name}"], stage_log)

    # 3. attn_proj: real [2560,2560]+bias.
    if not done("attn_proj"):
        stream_linear("attn_proj", key("attn.proj_out.all.weight"), key("attn.proj_out.all.bias"),
                      8, f"[8,{C}]", models / "attn_proj", args.pnnx, args.libtorch,
                      [f"{B}attn.proj_out.all.weight", f"{B}attn.proj_out.all.bias"], stage_log)

    # 4. SwiGLU gate/up/out (bias=False).
    for tag, name, rows in (("swiglu_gate", "mlp.all.proj_in_gate.weight", C),
                            ("swiglu_up", "mlp.all.proj_in.weight", C),
                            ("swiglu_out", "mlp.all.proj_out.weight", MLP)):
        if done(tag):
            continue
        stream_linear(tag, key(name), None, 8, f"[8,{rows}]", models / tag,
                      args.pnnx, args.libtorch, [f"{B}{name}"], stage_log)

    # AdaSingle real params (12 bins; 6 unique .all. values duplicated to sides).
    params = root / "params"
    params.mkdir(parents=True, exist_ok=True)
    ada_written = []
    for layer in ("attn", "mlp"):
        for part in ("shift", "scale", "gate"):
            t = key(f"ada.all.{layer}_{part}")
            for side in ("vid", "txt"):
                t.numpy().tofile(params / f"ada_{side}_{layer}_{part}.bin")
                ada_written.append(f"params/ada_{side}_{layer}_{part}.bin")

    # Metadata: runtime stage names -> shared physical artifacts.
    def entry(name, stage):
        return {"name": name,
                "ncnn_param": f"models/{stage}/model.ncnn.param",
                "ncnn_bin": f"models/{stage}/model.ncnn.bin"}

    models_map = {
        "attn_norm": entry("attn_norm", "attn_norm"),
        "qkv_vid": entry("qkv_vid", "qkv"), "qkv_txt": entry("qkv_txt", "qkv"),
        "q_norm_vid": entry("q_norm_vid", "q_norm"), "q_norm_txt": entry("q_norm_txt", "q_norm"),
        "k_norm_vid": entry("k_norm_vid", "k_norm"), "k_norm_txt": entry("k_norm_txt", "k_norm"),
        "rope_apply": entry("rope_apply", "rope_apply"),
        "attn_proj_vid": entry("attn_proj_vid", "attn_proj"), "attn_proj_txt": entry("attn_proj_txt", "attn_proj"),
        "mlp_norm": entry("mlp_norm", "mlp_norm"),
        "swiglu_gate_vid": entry("swiglu_gate_vid", "swiglu_gate"), "swiglu_gate_txt": entry("swiglu_gate_txt", "swiglu_gate"),
        "swiglu_up_vid": entry("swiglu_up_vid", "swiglu_up"), "swiglu_up_txt": entry("swiglu_up_txt", "swiglu_up"),
        "swiglu_act": entry("swiglu_act", "swiglu_act"),
        "swiglu_out_vid": entry("swiglu_out_vid", "swiglu_out"), "swiglu_out_txt": entry("swiglu_out_txt", "swiglu_out"),
    }
    for i in range(p16.HEADS):
        models_map[f"head{i}"] = entry(f"head{i}", "head")

    meta = {
        "name": variant, "variant": variant, "execution": "staged_dit_block",
        "window": [4, 3, 3], "grid": [1, 1, 1], "dtype": "fp32",
        "block": {
            "hidden_dim": C, "txt_dim": C, "heads": p16.HEADS, "head_dim": HD,
            "configured_rope_dim": 128, "effective_rotary_dim": 126, "identity_tail_dim": 2,
            "norm_eps": 1e-5, "mlp_hidden_dim": MLP, "mlp_type": "swiglu",
            "is_last_layer": is_last_layer, "weight_seed": 0, "real_seedvr2_weights": True,
            "block_index": bi, "shared_weights": True,
            "q_max": Q_MAX, "head_pairs": PAIRS,
        },
        "params_dir": "params",
        "models": models_map,
        "pnnx": {"command": "pnnx model.pt inputshape=[...] fp16=0", "fp16": 0},
        "ada_bins": sorted(ada_written),
    }
    (root / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8", newline="\n")
    (root / "export_stage_log.json").write_text(json.dumps(stage_log, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")

    manifest = {
        "phase": 16,
        "variant": variant,
        "block_index": bi,
        "source_block_index": bi,
        "is_last_layer": is_last_layer,
        "layout": "shared_all",
        "reused_from_phase15": False,
        "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
        "seedvr2_commit": p16.SEEDVR2_HEAD,
        "pnnx_fp16": 0,
        "qk_affine_mode": "real_affine_exported",
        "weighted_stage_count": 7,
        "shared_weight_free_stage_count": 5,
        "weight_free_stages_shared_with_phase15_block16": True,
        "stages": {tag: {"ncnn_bin_sha256": stage_log[tag]["sha256_bin"],
                         "ncnn_param_sha256": stage_log[tag]["sha256_param"],
                         "bin_bytes": stage_log[tag]["ncnn_bin_bytes"],
                         "param_bytes": stage_log[tag]["ncnn_param_bytes"]}
                   for tag in stage_log},
        "ada_bins": sorted(ada_written),
        "total_bin_bytes": int(sum(stage_log[t]["ncnn_bin_bytes"] for t in stage_log)),
        "total_param_bytes": int(sum(stage_log[t]["ncnn_param_bytes"] for t in stage_log)),
        "large_artifacts_committed": False,
    }
    (root / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    total = sum(v["ncnn_bin_bytes"] for v in stage_log.values())
    print(f"block {bi}: last_layer={is_last_layer} stages={len(stage_log)} total_bin={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
