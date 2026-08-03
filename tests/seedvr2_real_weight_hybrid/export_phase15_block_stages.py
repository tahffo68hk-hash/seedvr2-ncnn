#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: stream-export REAL-weight stages for blocks 15/17/18 to FP32 ncnn.

Loads ONLY `blocks.<i>.*` (shared .all. weights, 15 state_dict keys, real
q/k RMSNorm affine - never folded) from the verified official checkpoint and
exports the 12 unique FP32 stage artifacts with `pnnx ... fp16=0`, one graph
at a time, then writes the 12 AdaSingle runtime params bins.  One Python
module graph is released before the next stage (per-block streaming export).
Outputs land in `.phase15_local_artifacts/real_seedvr2_block<i>_c2560_h20_hd128/`
(git-ignored).  The Phase 13 random and Phase 14 block-16 artifacts are not
touched; block 16 is reused read-only from Phase 14 by a separate script.
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

import phase15_common as p15  # noqa: E402
from export_synthetic_dit_block import (  # noqa: E402
    LinearUnit, RMSNormUnit, RopeApplyUnit, HeadUnit, SwigluActUnit,
    trace_and_save, run_pnnx, make_gemm_dynamic, write_raw,
)
from export_real_weight_block_stages import (  # noqa: E402
    sha256, cleanup_stage, stream_linear, stream_rms,
)

C, HD, MLP = p15.C, p15.HD, p15.MLP_HIDDEN
Q_MAX, PAIRS = p15.Q_MAX, p15.PAIRS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-index", type=int, required=True,
                    choices=[b for b in p15.BLOCK_SET if b != p15.ANCHOR])
    ap.add_argument("--pnnx", type=Path, default=p15.PNNX_EXE)
    ap.add_argument("--libtorch", type=Path, default=p15.LIBTORCH)
    ap.add_argument("--root", type=Path, default=None,
                    help="defaults to .phase15_local_artifacts/<variant>")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    bi = args.block_index
    variant = p15.variant_for(bi)
    root = args.root or (p15.PHASE15_ROOT / variant)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    B = f"blocks.{bi}."
    stage_log = {}

    sd = torch.load(str(p15.MAIN_CKPT), map_location="cpu", weights_only=True, mmap=True)

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

    # 1. attn_norm / mlp_norm: no-affine RMSNorm (shared by vid/txt).
    for tag, dim in (("attn_norm", C), ("mlp_norm", C)):
        if done(tag):
            continue
        stream_rms(tag, dim, 1e-5, None, 8, f"[8,{dim}]", models / tag,
                   args.pnnx, args.libtorch, [], stage_log)

    # 2. qkv: real [7680,2560] shared weight (bias=False).
    if not done("qkv"):
        stream_linear("qkv", key("attn.proj_qkv.all.weight"), None, 8, f"[8,{C}]",
                      models / "qkv", args.pnnx, args.libtorch,
                      [f"{B}attn.proj_qkv.all.weight"], stage_log)

    # 3. q/k RMSNorm: REAL affine weights (never folded; block-specific).
    for tag, name in (("q_norm", "attn.norm_q.all.weight"), ("k_norm", "attn.norm_k.all.weight")):
        if done(tag):
            continue
        stream_rms(tag, HD, 1e-5, key(name), 8, f"[8,{HD}]",
                   models / tag, args.pnnx, args.libtorch, [f"{B}{name}"], stage_log)

    # 4. rope_apply: fixed padded capacity, no weights.
    if not done("rope_apply"):
        rope = RopeApplyUnit()
        x = torch.randn(Q_MAX, HD)
        cos = torch.randn(Q_MAX, PAIRS)
        sin = torch.randn(Q_MAX, PAIRS)
        t0 = time.perf_counter()
        with torch.no_grad():
            expected = rope(x, cos, sin)
        trace_and_save(rope, (x, cos, sin), models / "rope_apply")
        run_pnnx(args.pnnx, models / "rope_apply",
                 f"[{Q_MAX},{HD}],[{Q_MAX},{PAIRS}],[{Q_MAX},{PAIRS}]", args.libtorch)
        write_raw(expected, models / "rope_apply" / "reference_output.bin")
        info = cleanup_stage(models / "rope_apply")
        info["export_time_s"] = round(time.perf_counter() - t0, 3)
        info["source_params"] = []
        info["stage"] = "rope_apply"
        stage_log["rope_apply"] = info
        del rope, expected, x, cos, sin
        gc.collect()
        print(f"  rope_apply: export={info['export_time_s']}s")

    # 5. attn_proj: real [2560,2560]+bias.
    if not done("attn_proj"):
        stream_linear("attn_proj", key("attn.proj_out.all.weight"), key("attn.proj_out.all.bias"),
                      8, f"[8,{C}]", models / "attn_proj", args.pnnx, args.libtorch,
                      [f"{B}attn.proj_out.all.weight", f"{B}attn.proj_out.all.bias"], stage_log)

    # 6. SwiGLU gate/up/out (bias=False).
    for tag, name, rows in (("swiglu_gate", "mlp.all.proj_in_gate.weight", C),
                            ("swiglu_up", "mlp.all.proj_in.weight", C),
                            ("swiglu_out", "mlp.all.proj_out.weight", MLP)):
        if done(tag):
            continue
        stream_linear(tag, key(name), None, 8, f"[8,{rows}]", models / tag,
                      args.pnnx, args.libtorch, [f"{B}{name}"], stage_log)

    # 7. silu(gate)*up.
    if not done("swiglu_act"):
        act = SwigluActUnit()
        g = torch.randn(8, MLP)
        u = torch.randn(8, MLP)
        t0 = time.perf_counter()
        with torch.no_grad():
            expected = act(g, u)
        trace_and_save(act, (g, u), models / "swiglu_act")
        run_pnnx(args.pnnx, models / "swiglu_act", f"[8,{MLP}],[8,{MLP}]", args.libtorch)
        write_raw(expected, models / "swiglu_act" / "reference_output.bin")
        info = cleanup_stage(models / "swiglu_act")
        info["export_time_s"] = round(time.perf_counter() - t0, 3)
        info["source_params"] = []
        info["stage"] = "swiglu_act"
        stage_log["swiglu_act"] = info
        del act, expected, g, u
        gc.collect()
        print(f"  swiglu_act: export={info['export_time_s']}s")

    # 8. attention head (no weights, fixed capacity).
    if not done("head"):
        head = HeadUnit()
        q = torch.randn(Q_MAX, HD); k = torch.randn(Q_MAX, HD); v = torch.randn(Q_MAX, HD)
        mask = torch.zeros(Q_MAX, Q_MAX)
        t0 = time.perf_counter()
        with torch.no_grad():
            expected = head(q, k, v, mask)
        trace_and_save(head, (q, k, v, mask), models / "head")
        run_pnnx(args.pnnx, models / "head",
                 f"[{Q_MAX},{HD}],[{Q_MAX},{HD}],[{Q_MAX},{HD}],[{Q_MAX},{Q_MAX}]", args.libtorch)
        write_raw(expected, models / "head" / "reference_output.bin")
        info = cleanup_stage(models / "head")
        info["export_time_s"] = round(time.perf_counter() - t0, 3)
        info["source_params"] = []
        info["stage"] = "head"
        stage_log["head"] = info
        del head, expected, q, k, v, mask
        gc.collect()
        print(f"  head: export={info['export_time_s']}s")

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
    for i in range(p15.HEADS):
        models_map[f"head{i}"] = entry(f"head{i}", "head")

    meta = {
        "name": variant, "variant": variant, "execution": "staged_dit_block",
        "window": [4, 3, 3], "grid": [1, 1, 1], "dtype": "fp32",
        "block": {
            "hidden_dim": C, "txt_dim": C, "heads": p15.HEADS, "head_dim": HD,
            "configured_rope_dim": 128, "effective_rotary_dim": 126, "identity_tail_dim": 2,
            "norm_eps": 1e-5, "mlp_hidden_dim": MLP, "mlp_type": "swiglu",
            "is_last_layer": False, "weight_seed": 0, "real_seedvr2_weights": True,
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

    # Phase 15 artifact manifest (block binding, provenance, SHAs).
    manifest = {
        "phase": 15,
        "variant": variant,
        "block_index": bi,
        "source_block_index": bi,
        "reused_from_phase14": False,
        "checkpoint_sha256": p15.EXPECTED_MAIN_SHA256,
        "seedvr2_commit": p15.SEEDVR2_HEAD,
        "pnnx_fp16": 0,
        "qk_affine_mode": "real_affine_exported",
        "weighted_stage_count": 7,   # qkv, q_norm, k_norm, attn_proj, swiglu_gate/up/out
        "shared_weight_free_stage_count": 5,  # attn_norm, mlp_norm, rope_apply, swiglu_act, head
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
    print(f"block {bi}: stages exported: {len(stage_log)} total ncnn bin bytes: {total}")
    print(f"  ada bins: {len(ada_written)} manifest: {root / 'artifact_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
