"""Phase 13: stream-export the real-width SeedVR2 DiT block stages to FP32 ncnn.

Rebuilds the REAL `NaMMSRTransformerBlock` at C=2560/H=20/HD=128 (seed
20260803), strict-loads the local random state_dict, then exports each of the
19 stage artifacts with `pnnx ... fp16=0` one at a time: the trace wrapper,
TorchScript and pnnx intermediates are released immediately after each stage
so only one graph is ever live.  Outputs go to the git-ignored
`.phase13_local_artifacts/` directory (never committed).
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "seedvr2_weight_mapping"))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))

import _realwidth_common as rw  # noqa: E402
import probe_lib  # noqa: E402
from export_synthetic_dit_block import (  # noqa: E402
    LinearUnit, RMSNormUnit, RopeApplyUnit, HeadUnit, SwigluActUnit,
    trace_and_save, run_pnnx, make_gemm_dynamic, write_raw,
)

SCALE = rw.SCALE


def peak_working_set_mb() -> float:
    try:
        class P(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwPageFaultCount", ctypes.c_ulong),
                        ("dwPeakWorkingSetSize", ctypes.c_size_t), ("dwWorkingSetSize", ctypes.c_size_t)]
        p = P()
        p.dwLength = ctypes.sizeof(P)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                      ctypes.byref(p), ctypes.sizeof(p))
        return (p.dwPeakWorkingSetSize / (1024.0 * 1024.0)) if ok else -1.0
    except Exception:
        return -1.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_stage(dir_: Path) -> dict:
    """Delete TorchScript / pnnx intermediates; keep ncnn + reference.  Returns sizes."""
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
    print(f"  {tag}: ncnn_bin={info['ncnn_bin_bytes']} export={dt:.1f}s peak_ws={peak_working_set_mb():.0f}MB")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=Path,
                        default=Path(os.environ.get("PNNX_EXE", HERE.parent.parent / "build" / "pnnx-cpu" / "src" / "pnnx.exe")))
    parser.add_argument("--libtorch", type=Path, default=Path(r"V:\AI\LibTorch\2.7.0+cpu\lib"))
    parser.add_argument("--root", type=Path, default=rw.LOCAL_ROOT)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    probe_lib.ensure_compat()
    torch.Tensor.bfloat16 = lambda self: self
    block = probe_lib.build_block(dict(rw.REAL_CFG), seed=rw.SEED).eval()
    if rw.STATE_DICT.exists():
        sd_pt = torch.load(rw.STATE_DICT, map_location="cpu")
        block.load_state_dict({k: v for k, v in sd_pt.items()}, strict=True)
    sd = {k: v.detach() for k, v in block.state_dict().items()}

    root = args.root
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    stage_log = {}

    def done(tag):
        if args.skip_existing and (models / tag / "model.ncnn.bin").exists():
            info = cleanup_stage(models / tag)
            info["stage"] = tag
            info["export_time_s"] = 0.0
            info["source_params"] = []
            stage_log[tag] = info
            return True
        return False

    C, HD, MLP = rw.C, rw.HD, rw.MLP_HIDDEN

    # no-affine branch norms (weights=None)
    for tag in ("attn_norm", "mlp_norm"):
        if done(tag):
            continue
        stream_rms(tag, C, 1e-5, None, 8, f"[8,{C}]", models / tag, args.pnnx, args.libtorch, [], stage_log)

    # qkv (bias=False)
    for tag, key in (("qkv_vid", "attn.proj_qkv.vid.weight"), ("qkv_txt", "attn.proj_qkv.txt.weight")):
        if done(tag):
            continue
        stream_linear(tag, sd[key], None, 8, f"[8,{C}]", models / tag, args.pnnx, args.libtorch, [key], stage_log)

    # q/k RMSNorm (affine, per-head dim 128)
    for tag, key in (("q_norm_vid", "attn.norm_q.vid.weight"), ("q_norm_txt", "attn.norm_q.txt.weight"),
                     ("k_norm_vid", "attn.norm_k.vid.weight"), ("k_norm_txt", "attn.norm_k.txt.weight")):
        if done(tag):
            continue
        stream_rms(tag, HD, 1e-5, sd[key], 8, f"[8,{HD}]", models / tag, args.pnnx, args.libtorch, [key], stage_log)

    # rope_apply (fixed padded capacity, no weights)
    if not done("rope_apply"):
        rope = RopeApplyUnit()
        x = torch.randn(rw.Q_MAX, HD)
        cos = torch.randn(rw.Q_MAX, rw.PAIRS)
        sin = torch.randn(rw.Q_MAX, rw.PAIRS)
        t0 = time.perf_counter()
        with torch.no_grad():
            expected = rope(x, cos, sin)
        trace_and_save(rope, (x, cos, sin), models / "rope_apply")
        run_pnnx(args.pnnx, models / "rope_apply",
                 f"[{rw.Q_MAX},{HD}],[{rw.Q_MAX},{rw.PAIRS}],[{rw.Q_MAX},{rw.PAIRS}]", args.libtorch)
        write_raw(expected, models / "rope_apply" / "reference_output.bin")
        info = cleanup_stage(models / "rope_apply")
        info["export_time_s"] = round(time.perf_counter() - t0, 3)
        info["source_params"] = []
        info["stage"] = "rope_apply"
        stage_log["rope_apply"] = info
        del rope, expected, x, cos, sin
        gc.collect()
        print(f"  rope_apply: export={info['export_time_s']}s")

    # attention head (attention only, no weights, fixed capacity)
    if not done("head"):
        head = HeadUnit()
        q = torch.randn(rw.Q_MAX, HD); k = torch.randn(rw.Q_MAX, HD); v = torch.randn(rw.Q_MAX, HD)
        mask = torch.zeros(rw.Q_MAX, rw.Q_MAX)
        t0 = time.perf_counter()
        with torch.no_grad():
            expected = head(q, k, v, mask)
        trace_and_save(head, (q, k, v, mask), models / "head")
        run_pnnx(args.pnnx, models / "head",
                 f"[{rw.Q_MAX},{HD}],[{rw.Q_MAX},{HD}],[{rw.Q_MAX},{HD}],[{rw.Q_MAX},{rw.Q_MAX}]", args.libtorch)
        write_raw(expected, models / "head" / "reference_output.bin")
        info = cleanup_stage(models / "head")
        info["export_time_s"] = round(time.perf_counter() - t0, 3)
        info["source_params"] = []
        info["stage"] = "head"
        stage_log["head"] = info
        del head, expected, q, k, v, mask
        gc.collect()
        print(f"  head: export={info['export_time_s']}s")

    # attention output projections (bias=True)
    for tag, wk, bk in (("attn_proj_vid", "attn.proj_out.vid.weight", "attn.proj_out.vid.bias"),
                        ("attn_proj_txt", "attn.proj_out.txt.weight", "attn.proj_out.txt.bias")):
        if done(tag):
            continue
        stream_linear(tag, sd[wk], sd[bk], 8, f"[8,{C}]", models / tag, args.pnnx, args.libtorch, [wk, bk], stage_log)

    # SwiGLU gate/up/out vid+txt (bias=False)
    for tag, key in (("swiglu_gate_vid", "mlp.vid.proj_in_gate.weight"),
                     ("swiglu_gate_txt", "mlp.txt.proj_in_gate.weight"),
                     ("swiglu_up_vid", "mlp.vid.proj_in.weight"),
                     ("swiglu_up_txt", "mlp.txt.proj_in.weight"),
                     ("swiglu_out_vid", "mlp.vid.proj_out.weight"),
                     ("swiglu_out_txt", "mlp.txt.proj_out.weight")):
        if done(tag):
            continue
        if tag.startswith("swiglu_out"):
            stream_linear(tag, sd[key], None, 8, f"[8,{MLP}]", models / tag, args.pnnx, args.libtorch, [key], stage_log)
        else:
            stream_linear(tag, sd[key], None, 8, f"[8,{C}]", models / tag, args.pnnx, args.libtorch, [key], stage_log)

    # silu(gate)*up
    if not done("swiglu_act"):
        act = SwigluActUnit()
        g = torch.randn(8, MLP); u = torch.randn(8, MLP)
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

    # metadata (same 19 model entries as Phase 12, real-width block config)
    meta = {
        "name": rw.VARIANT, "variant": rw.VARIANT, "execution": "staged_dit_block",
        "window": [4, 3, 3], "grid": [1, 1, 1], "dtype": "fp32",
        "block": {
            "hidden_dim": C, "txt_dim": C, "heads": rw.HEADS, "head_dim": HD,
            "configured_rope_dim": 128, "effective_rotary_dim": 126, "identity_tail_dim": 2,
            "norm_eps": 1e-5, "mlp_hidden_dim": MLP, "mlp_type": "swiglu",
            "is_last_layer": False, "weight_seed": rw.SEED, "real_seedvr2_weights": False,
            "q_max": rw.Q_MAX, "head_pairs": rw.PAIRS,
        },
        "params_dir": "params",
        "models": {
            "attn_norm": {"name": "attn_norm", "ncnn_param": "models/attn_norm/model.ncnn.param", "ncnn_bin": "models/attn_norm/model.ncnn.bin"},
            "qkv_vid": {"name": "qkv_vid", "ncnn_param": "models/qkv_vid/model.ncnn.param", "ncnn_bin": "models/qkv_vid/model.ncnn.bin"},
            "qkv_txt": {"name": "qkv_txt", "ncnn_param": "models/qkv_txt/model.ncnn.param", "ncnn_bin": "models/qkv_txt/model.ncnn.bin"},
            "q_norm_vid": {"name": "q_norm_vid", "ncnn_param": "models/q_norm_vid/model.ncnn.param", "ncnn_bin": "models/q_norm_vid/model.ncnn.bin"},
            "q_norm_txt": {"name": "q_norm_txt", "ncnn_param": "models/q_norm_txt/model.ncnn.param", "ncnn_bin": "models/q_norm_txt/model.ncnn.bin"},
            "k_norm_vid": {"name": "k_norm_vid", "ncnn_param": "models/k_norm_vid/model.ncnn.param", "ncnn_bin": "models/k_norm_vid/model.ncnn.bin"},
            "k_norm_txt": {"name": "k_norm_txt", "ncnn_param": "models/k_norm_txt/model.ncnn.param", "ncnn_bin": "models/k_norm_txt/model.ncnn.bin"},
            "rope_apply": {"name": "rope_apply", "ncnn_param": "models/rope_apply/model.ncnn.param", "ncnn_bin": "models/rope_apply/model.ncnn.bin"},
            "heads": [{"name": f"head{i}", "ncnn_param": "models/head/model.ncnn.param", "ncnn_bin": "models/head/model.ncnn.bin"} for i in range(rw.HEADS)],
            "attn_proj_vid": {"name": "attn_proj_vid", "ncnn_param": "models/attn_proj_vid/model.ncnn.param", "ncnn_bin": "models/attn_proj_vid/model.ncnn.bin"},
            "attn_proj_txt": {"name": "attn_proj_txt", "ncnn_param": "models/attn_proj_txt/model.ncnn.param", "ncnn_bin": "models/attn_proj_txt/model.ncnn.bin"},
            "mlp_norm": {"name": "mlp_norm", "ncnn_param": "models/mlp_norm/model.ncnn.param", "ncnn_bin": "models/mlp_norm/model.ncnn.bin"},
            "swiglu_gate_vid": {"name": "swiglu_gate_vid", "ncnn_param": "models/swiglu_gate_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_gate_vid/model.ncnn.bin"},
            "swiglu_gate_txt": {"name": "swiglu_gate_txt", "ncnn_param": "models/swiglu_gate_txt/model.ncnn.param", "ncnn_bin": "models/swiglu_gate_txt/model.ncnn.bin"},
            "swiglu_up_vid": {"name": "swiglu_up_vid", "ncnn_param": "models/swiglu_up_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_up_vid/model.ncnn.bin"},
            "swiglu_up_txt": {"name": "swiglu_up_txt", "ncnn_param": "models/swiglu_up_txt/model.ncnn.param", "ncnn_bin": "models/swiglu_up_txt/model.ncnn.bin"},
            "swiglu_act": {"name": "swiglu_act", "ncnn_param": "models/swiglu_act/model.ncnn.param", "ncnn_bin": "models/swiglu_act/model.ncnn.bin"},
            "swiglu_out_vid": {"name": "swiglu_out_vid", "ncnn_param": "models/swiglu_out_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_out_vid/model.ncnn.bin"},
            "swiglu_out_txt": {"name": "swiglu_out_txt", "ncnn_param": "models/swiglu_out_txt/model.ncnn.param", "ncnn_bin": "models/swiglu_out_txt/model.ncnn.bin"},
        },
        "pnnx": {"command": "pnnx model.pt inputshape=[...] fp16=0", "fp16": 0},
    }
    (root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (root / "export_stage_log.json").write_text(json.dumps(stage_log, indent=2), encoding="utf-8")
    print("export complete:", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
