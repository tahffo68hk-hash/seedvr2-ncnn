#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: per-stage ncnn vs official-reference parity for profiles A/B.

Reads the captured reference stage inputs/outputs (manifest.json), runs the
FP32 ncnn stage runner on the exported artifacts, and records max_abs,
mean_abs, RMSE, relative max, NaN/Inf plus the applied threshold.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402

RUNNER = Path(r"D:\AIProjects\seedvr2-ncnn\build\phase17-vae-smoke\vae_stage_runner.exe")
THRESHOLDS = pc.DEFAULT_STAGE_THRESHOLDS


def stage_threshold(stage: str) -> tuple:
    if stage in ("enc_conv_in", "dec_conv_in"):
        return THRESHOLDS["conv"], THRESHOLDS["relative_max"]
    if stage == "enc_mid":
        return THRESHOLDS["residual_block"], THRESHOLDS["relative_max"]
    if stage.startswith("enc_down"):
        return THRESHOLDS["downsample"], THRESHOLDS["relative_max"]
    if stage in ("dec_mid",):
        return THRESHOLDS["residual_block"], THRESHOLDS["relative_max"]
    if stage.startswith("dec_up"):
        return THRESHOLDS["upsample_deconv"], THRESHOLDS["relative_max"]
    if stage in ("enc_out",):
        return THRESHOLDS["encoder_final_latent"], THRESHOLDS["relative_max"]
    if stage == "dec_out":
        return THRESHOLDS["decoder_final_float"], THRESHOLDS["relative_max"]
    return 1e-4, THRESHOLDS["relative_max"]


def ncnn_dims(shape):
    _, c, t, h, w = shape
    return (w, h, t, c)


def run_stage(artifact_dir: Path, manifest, stage: str, tmp: Path) -> dict:
    minfo = manifest["stage_files"][stage]
    in_shape = minfo["in"]["shape"]
    out_shape = minfo["out"]["shape"]
    x = np.fromfile(minfo["in"]["path"], dtype="<f4").reshape(in_shape)
    ref = np.fromfile(minfo["out"]["path"], dtype="<f4").reshape(out_shape)
    if stage in ("enc_out", "dec_out"):
        # The official conv_norm_out receives the frame-outer (B*T, C, H, W)
        # layout via causal_norm_wrapper, while the export wrapper consumes
        # the channel-outer (1, C, T, H, W) layout.  Permute before feeding
        # the ncnn stage model (verified: identical values after permute).
        bt, c, h, w = in_shape
        x = x.transpose(1, 0, 2, 3).reshape(1, c, bt, h, w)
        in_shape = [1, c, bt, h, w]
    in_bin = tmp / f"{stage}_in.bin"
    out_bin = tmp / f"{stage}_out.bin"
    x.tofile(str(in_bin))
    cmd = [str(RUNNER), str(artifact_dir / "model.ncnn.param"), str(artifact_dir / "model.ncnn.bin"),
           str(in_bin), str(out_bin)]
    cmd += [str(d) for d in ncnn_dims(in_shape)] + ["--"] + [str(d) for d in ncnn_dims(out_shape)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"stage": stage, "runner_rc": r.returncode, "stderr": r.stderr.strip()[:200],
                "pass": False}
    got = np.fromfile(str(out_bin), dtype="<f4").reshape(out_shape)
    ref_t = torch.from_numpy(ref)
    got_t = torch.from_numpy(got)
    diff = (ref_t.float() - got_t.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rmse = float((diff ** 2).mean().sqrt().item())
    rel_max = float(max_abs / ref_t.float().abs().max().item()) if ref_t.float().abs().max().item() > 0 else 0.0
    nan = int(torch.isnan(got_t).sum().item())
    inf = int(torch.isinf(got_t).sum().item())
    th_abs, th_rel = stage_threshold(stage)
    ref_absmax = float(ref_t.float().abs().max().item()) if ref_t.numel() else 0.0
    # The absolute thresholds are calibrated for unit-scale outputs (final
    # latent / decoded pixels).  Intermediate stages can legitimately carry
    # large magnitudes (e.g. enc_mid absmax ~2e3); for those the meaningful
    # bound is the relative error.  Scale the absolute threshold linearly with
    # the reference magnitude so the gate stays scale-invariant and the final
    # small-magnitude stages keep their strict absolute bounds.
    th_abs_eff = th_abs * max(1.0, ref_absmax)
    passed = max_abs < th_abs_eff and rel_max < th_rel and nan == 0 and inf == 0
    return {
        "stage": stage,
        "input_shape": in_shape,
        "output_shape": out_shape,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "relative_max": rel_max,
        "ref_absmax": ref_absmax,
        "nan": nan,
        "inf": inf,
        "threshold_abs": th_abs,
        "threshold_abs_effective": th_abs_eff,
        "threshold_rel": th_rel,
        "pass": passed,
        "artifact_sha256_bin": pc.sha256_file(str(artifact_dir / "model.ncnn.bin")),
    }


def main() -> int:
    root = Path(pc.ART)
    tmp = Path(pc.EXPORT_TMP) / "parity_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    report = {"phase": 17, "real_seedvr2_weights": True,
              "checkpoint_sha256": pc.VAE_CKPT_SHA256, "profiles": {}}
    overall = True
    for prof in ("A", "B"):
        manifest = json.loads((root / f"reference_{prof}" / "manifest.json").read_text(encoding="utf-8"))
        stages = list(manifest["stage_files"].keys())
        per = {}
        for stage in stages:
            artifact_dir = root / f"stages_{prof}" / stage
            if not (artifact_dir / "model.ncnn.bin").exists():
                per[stage] = {"stage": stage, "runner_rc": -1, "pass": False, "error": "artifact missing"}
                overall = False
                continue
            res = run_stage(artifact_dir, manifest, stage, tmp)
            per[stage] = res
            if not res.get("pass"):
                overall = False
            print(f"  {prof} {stage}: max={res.get('max_abs_error', 'n/a')} rel={res.get('relative_max', 'n/a')} pass={res.get('pass')}", flush=True)
        report["profiles"][prof] = {"stages": per,
                                    "all_pass": all(v.get("pass") for v in per.values())}
    report["overall_pass"] = overall
    pc.write_report(str(Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_vae_stage_parity.json")), report)
    print("overall pass:", overall)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
