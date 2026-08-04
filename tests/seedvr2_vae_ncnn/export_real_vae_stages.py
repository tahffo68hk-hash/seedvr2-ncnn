#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: stream-export REAL SeedVR2 VAE encoder/decoder stages to FP32 ncnn.

For each profile (A/B) and each stage, builds the graph-equivalent export wrapper
from the official modules, traces with the official reference input, converts
with `pnnx ... fp16=0`, runs an ncnn smoke via vae_stage_runner against the
wrapper output, records sizes/SHA/smoke-error, and removes all intermediate
TorchScript/pnnx files.  Artifacts live under the git-ignored
.phase17_local_artifacts/ (never committed).
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
import phase17_common as pc  # noqa: E402
import vae_export_model as vem  # noqa: E402

# torch BCTHW stage input shapes per profile
SHAPES = {
    "A": {
        "enc_conv_in": [1, 3, 5, 128, 128], "enc_down0": [1, 128, 5, 128, 128],
        "enc_down1": [1, 128, 5, 64, 64], "enc_down2": [1, 256, 3, 32, 32],
        "enc_down3": [1, 512, 2, 16, 16], "enc_mid": [1, 512, 2, 16, 16],
        "enc_out": [1, 512, 2, 16, 16],
        "dec_conv_in": [1, 16, 2, 16, 16], "dec_mid": [1, 512, 2, 16, 16],
        "dec_up0": [1, 512, 2, 16, 16], "dec_up1": [1, 512, 3, 32, 32],
        "dec_up2": [1, 512, 5, 64, 64], "dec_up3": [1, 256, 5, 128, 128],
        "dec_out": [1, 128, 5, 128, 128],
    },
    "B": {
        "enc_conv_in": [1, 3, 9, 192, 256], "enc_down0": [1, 128, 9, 192, 256],
        "enc_down1": [1, 128, 9, 96, 128], "enc_down2": [1, 256, 5, 48, 64],
        "enc_down3": [1, 512, 3, 24, 32], "enc_mid": [1, 512, 3, 24, 32],
        "enc_out": [1, 512, 3, 24, 32],
        "dec_conv_in": [1, 16, 3, 24, 32], "dec_mid": [1, 512, 3, 24, 32],
        "dec_up0": [1, 512, 3, 24, 32], "dec_up1": [1, 512, 5, 48, 64],
        "dec_up2": [1, 512, 9, 96, 128], "dec_up3": [1, 256, 9, 192, 256],
        "dec_out": [1, 128, 9, 192, 256],
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_raw(value: torch.Tensor, path: Path) -> None:
    value.detach().cpu().contiguous().float().numpy().tofile(str(path))


def trace_and_save(model, example, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.jit.trace(model, example, check_trace=True).save(str(target / "model.pt"))


def run_pnnx(pnnx: Path, directory: Path, shape: str, libtorch: Path) -> None:
    env = dict(os.environ)
    env["PATH"] = str(libtorch) + os.pathsep + env.get("PATH", "")
    subprocess.run([str(pnnx), "model.pt", f"inputshape={shape}", "fp16=0"],
                   cwd=directory, check=True, env=env)


def ncnn_dims(shape) -> tuple:
    """BCTHW torch -> ncnn Mat dims (w,h,d,c) for batch=1."""
    _, c, t, h, w = shape
    return (w, h, t, c)


def run_runner(runner: Path, param: Path, bin_: Path, in_blob: Path, out_blob: Path,
               in_dims, out_dims) -> int:
    cmd = [str(runner), str(param), str(bin_), str(in_blob), str(out_blob)]
    cmd += [str(d) for d in in_dims] + ["--"] + [str(d) for d in out_dims]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"runner failed rc={r.returncode}: {r.stderr.strip()}")
    return 0


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
    info["sha256_param"] = sha256_file(p) if p.exists() else ""
    info["sha256_bin"] = sha256_file(b) if b.exists() else ""
    return info


def stage_source_keys(vae, stage: str) -> list:
    enc, dec = vae.encoder, vae.decoder
    maps = {
        "enc_conv_in": lambda: enc.conv_in,
        "enc_down0": lambda: enc.down_blocks[0],
        "enc_down1": lambda: enc.down_blocks[1],
        "enc_down2": lambda: enc.down_blocks[2],
        "enc_down3": lambda: enc.down_blocks[3],
        "enc_mid": lambda: enc.mid_block,
        "enc_out": lambda: enc,
        "dec_conv_in": lambda: dec.conv_in,
        "dec_mid": lambda: dec.mid_block,
        "dec_up0": lambda: dec.up_blocks[0],
        "dec_up1": lambda: dec.up_blocks[1],
        "dec_up2": lambda: dec.up_blocks[2],
        "dec_up3": lambda: dec.up_blocks[3],
        "dec_out": lambda: dec,
    }
    mod = maps[stage]()
    if stage == "enc_out":
        keep = ("conv_norm_out", "conv_act", "conv_out")
        return [k for k in mod.state_dict() if k.startswith(keep)]
    if stage == "dec_out":
        keep = ("conv_norm_out", "conv_act", "conv_out")
        return [k for k in mod.state_dict() if k.startswith(keep)]
    return list(mod.state_dict().keys())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnnx", type=Path,
                    default=Path(os.environ.get("PNNX_EXE", pc.REPO_ROOT + r"\build\pnnx-cpu\src\pnnx.exe")))
    ap.add_argument("--libtorch", type=Path, default=Path(r"V:\AI\LibTorch\2.7.0+cpu\lib"))
    ap.add_argument("--runner", type=Path,
                    default=Path(r"D:\AIProjects\seedvr2-ncnn\build\phase17-vae-smoke\vae_stage_runner.exe"))
    ap.add_argument("--root", type=Path, default=Path(pc.ART))
    ap.add_argument("--profiles", default="A,B")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    vae = pc.build_real_vae()
    sd, _ = pc.load_vae_state_dict()
    missing, unexpected = vae.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    vae.eval().to(torch.float32)
    vae.disable_slicing()
    vae.set_memory_limit(None, None)

    enc_stages = vem.build_encoder_stages(vae.encoder)
    dec_stages = vem.build_decoder_stages(vae.decoder)
    all_stages = {**enc_stages, **dec_stages}

    inventory = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "pnnx_command": "pnnx model.pt inputshape=[...] fp16=0",
        "fp16": 0,
        "profiles": {},
    }

    for pname in [p.strip() for p in args.profiles.split(",")]:
        shapes = SHAPES[pname]
        ref_root = args.root / f"reference_{pname}"
        out_root = args.root / f"stages_{pname}"
        inv_p = inventory["profiles"][pname] = {}
        for stage in list(all_stages.keys()):
            shape = shapes[stage]
            target = out_root / stage
            if args.skip_existing and (target / "model.ncnn.bin").exists():
                info = cleanup_stage(target)
                info["stage"] = stage
                info["export_time_s"] = 0.0
                info["source_keys"] = []
                inv_p[stage] = info
                print(f"  skip {stage}", flush=True)
                continue
            mod = all_stages[stage]
            # load the official reference input blob
            in_blob = ref_root / stage / "input.bin"
            if not in_blob.exists():
                raise FileNotFoundError(f"missing reference input {in_blob}")
            x = torch.from_numpy(
                __import__("numpy").fromfile(in_blob, dtype="<f4")
            ).reshape(shape)
            t0 = time.perf_counter()
            with torch.no_grad():
                expected = mod(x)
            target.mkdir(parents=True, exist_ok=True)
            trace_and_save(mod, x, target)
            run_pnnx(args.pnnx, target,
                     "[" + ",".join(str(s) for s in shape) + "]", args.libtorch)
            in_dims = ncnn_dims(shape)
            out_dims = ncnn_dims(expected.shape)
            write_raw(x, target / "smoke_input.bin")
            write_raw(expected, target / "expected_output.bin")
            run_runner(args.runner, target / "model.ncnn.param", target / "model.ncnn.bin",
                       target / "smoke_input.bin", target / "smoke_output.bin",
                       in_dims, out_dims)
            got = torch.from_numpy(
                __import__("numpy").fromfile(target / "smoke_output.bin", dtype="<f4")
            ).reshape(expected.shape)
            max_abs = float((expected.float() - got.float()).abs().max().item())
            rel_max = float(max_abs / expected.float().abs().max().item())
            info = cleanup_stage(target)
            info["stage"] = stage
            info["export_time_s"] = round(time.perf_counter() - t0, 3)
            info["input_shape"] = shape
            info["output_shape"] = list(expected.shape)
            info["source_keys"] = stage_source_keys(vae, stage)
            info["smoke_max_abs_diff"] = max_abs
            info["smoke_rel_max"] = rel_max
            inv_p[stage] = info
            (target / "smoke_input.bin").unlink(missing_ok=True)
            (target / "smoke_output.bin").unlink(missing_ok=True)
            (target / "expected_output.bin").unlink(missing_ok=True)
            print(f"  {stage}: bin={info['ncnn_bin_bytes']} smoke={max_abs:.3e} "
                  f"export={info['export_time_s']:.1f}s", flush=True)
            del mod, x, expected, got
            gc.collect()

    pc.write_report(str(Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_vae_artifact_inventory.json")), inventory)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
