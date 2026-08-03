#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14: stream-export the REAL-weight SeedVR2 block 16 stages to FP32 ncnn.

Loads ONLY `blocks.16.*` from the real checkpoint (shared .all. weights, 15
state_dict keys, q/k RMSNorm real affine - never folded), then exports the 12
unique FP32 stage artifacts with `pnnx ... fp16=0` one graph at a time.  The
runtime stage names (19 + 20 heads) map vid/txt pairs to the SAME physical
artifact because block 16 uses shared weights.  AdaSingle real shift/scale/gate
values are written as runtime params bins (12 files, 6 unique duplicated to
vid/txt sides).  Outputs go to the git-ignored `.phase14_local_artifacts/`
(never committed); the Phase 13 random-weight artifacts are NOT touched.
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

import phase14_common as p14  # noqa: E402
from export_synthetic_dit_block import (  # noqa: E402
    LinearUnit, RMSNormUnit, RopeApplyUnit, HeadUnit, SwigluActUnit,
    trace_and_save, run_pnnx, make_gemm_dynamic, write_raw,
)

C, HD, MLP = p14.C, p14.HD, p14.MLP_HIDDEN
Q_MAX, PAIRS = p14.Q_MAX, p14.PAIRS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_stage(dir_: Path) -> dict:
    info = {}
    for suffix, key in ((".pt", "torchscript_bytes"), (".pnnx.param", "pnnx_param_bytes"),
                        (".pnnx.bin", "pnnx_bin_bytes"), (".pnnx.py", "pnnx_py_bytes"),
                        (".pnnx.onnx", "pnnx_onnx_bytes")):
        f = dir_ / ("model" + suffix)
        if f.exists():
            info[key] = int(f.stat().st_size)
            f.unlink()
    p = dir_ / "model.ncnn.param"
    b = dir_ / "model.ncnn.bin"
    info["ncnn_param_bytes"] = int(p.stat().st_size) if p.exists() else 0
    info["ncnn_bin_bytes"] = int(b.stat().st_size) if b.exists() else 0
    info["sha256_bin"] = sha256(b) if b.exists() else ""
    info["sha256_param"] = sha256(p) if p.exists() else ""
    return info


def stream_linear(tag, weight, bias, trace_rows, shape, target, pnnx, libtorch,
                  source_params, stage_log):
    unit = LinearUnit(weight, bias)
    x = torch.randn(trace_rows, weight.shape[1])
    t0 = time.perf_counter()
    with torch.no_grad():
        expected = unit(x)
    trace_and_save(unit, x, target)
    run_pnnx(pnnx, target, shape, libtorch)
    make_gemm_dynamic(target / "model.ncnn.param")
    write_raw(expected, target / "reference_output.bin")
    dt = time.perf_counter() - t0
    info = cleanup_stage(target)
    info["export_time_s"] = round(dt, 3)
    info["source_params"] = source_params
    info["stage"] = tag
    stage_log[tag] = info
    del unit, expected, x
    gc.collect()
    print(f"  {tag}: ncnn_bin={info['ncnn_bin_bytes']} export={dt:.1f}s")
    return target


def stream_rms(tag, dim, eps, weight, trace_rows, shape, target, pnnx, libtorch,
               source_params, stage_log):
    unit = RMSNormUnit(dim, eps, weight)
    x = torch.randn(trace_rows, dim)
    t0 = time.perf_counter()
    with torch.no_grad():
        expected = unit(x)
    trace_and_save(unit, x, target)
    run_pnnx(pnnx, target, shape, libtorch)
    write_raw(expected, target / "reference_output.bin")
    dt = time.perf_counter() - t0
    info = cleanup_stage(target)
    info["export_time_s"] = round(dt, 3)
    info["source_params"] = source_params
    info["stage"] = tag
    stage_log[tag] = info
    del unit, expected, x
    gc.collect()
    print(f"  {tag}: ncnn_bin={info['ncnn_bin_bytes']} export={dt:.1f}s")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnnx", type=Path,
                    default=Path(os.environ.get("PNNX_EXE", p14.REPO / "build" / "pnnx-cpu" / "src" / "pnnx.exe")))
    ap.add_argument("--libtorch", type=Path, default=Path(r"V:\AI\LibTorch\2.7.0+cpu\lib"))
    ap.add_argument("--root", type=Path, default=p14.LOCAL_ROOT)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = args.root
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    stage_log = {}

    sd = torch.load(p14.MAIN_CKPT, map_location="cpu", weights_only=True, mmap=True)
    B = p14.BLOCK_PREFIX
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

    # 1. attn_norm / mlp_norm: no-affine RMSNorm (shared, used by vid+txt).
    for tag, dim in (("attn_norm", C), ("mlp_norm", C)):
        if done(tag):
            continue
        stream_rms(tag, dim, 1e-5, None, 8, f"[8,{dim}]", models / tag, args.pnnx, args.libtorch, [], stage_log)

    # 2. qkv: real [7680,2560] shared weight (bias=False).
    if not done("qkv"):
        stream_linear("qkv", key("attn.proj_qkv.all.weight"), None, 8, f"[8,{C}]",
                      models / "qkv", args.pnnx, args.libtorch, ["blocks.16.attn.proj_qkv.all.weight"], stage_log)

    # 3. q/k RMSNorm: REAL affine weights (never folded).
    for tag, name in (("q_norm", "attn.norm_q.all.weight"), ("k_norm", "attn.norm_k.all.weight")):
        if done(tag):
            continue
        stream_rms(tag, HD, 1e-5, key(name), 8, f"[8,{HD}]",
                   models / tag, args.pnnx, args.libtorch, [f"blocks.16.{name}"], stage_log)

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
                      ["blocks.16.attn.proj_out.all.weight", "blocks.16.attn.proj_out.all.bias"], stage_log)

    # 6. SwiGLU gate/up/out (bias=False).
    for tag, name, rows in (("swiglu_gate", "mlp.all.proj_in_gate.weight", C),
                            ("swiglu_up", "mlp.all.proj_in.weight", C),
                            ("swiglu_out", "mlp.all.proj_out.weight", MLP)):
        if done(tag):
            continue
        stream_linear(tag, key(name), None, 8, f"[8,{rows}]", models / tag,
                      args.pnnx, args.libtorch, [f"blocks.16.{name}"], stage_log)

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
    for i in range(p14.HEADS):
        models_map[f"head{i}"] = entry(f"head{i}", "head")

    meta = {
        "name": p14.VARIANT, "variant": p14.VARIANT, "execution": "staged_dit_block",
        "window": [4, 3, 3], "grid": [1, 1, 1], "dtype": "fp32",
        "block": {
            "hidden_dim": C, "txt_dim": C, "heads": p14.HEADS, "head_dim": HD,
            "configured_rope_dim": 128, "effective_rotary_dim": 126, "identity_tail_dim": 2,
            "norm_eps": 1e-5, "mlp_hidden_dim": MLP, "mlp_type": "swiglu",
            "is_last_layer": False, "weight_seed": None, "real_seedvr2_weights": True,
            "block_index": 16, "shared_weights": True,
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
    total = sum(v["ncnn_bin_bytes"] for v in stage_log.values())
    print("stages exported:", len(stage_log), "total ncnn bin bytes:", total)
    print("ada bins:", len(ada_written))
    print("metadata:", root / "metadata.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
