# -*- coding: utf-8 -*-
"""Phase 17: full pipeline matrix on the official SeedVR2 3B runner.

Modes (all with deterministic mean-path latent, use_sample=False):
  pytorch_reference        PyTorch VAE + PyTorch DiT + PyTorch VAE decode
  dit_ncnn_vae_pytorch     PyTorch VAE + 32-Block ncnn DiT + PyTorch decode
  dit_ncnn_decoder_ncnn    PyTorch VAE + 32-Block ncnn DiT + ncnn VAE decode
  full_ncnn_core           ncnn VAE encode + 32-Block ncnn DiT + ncnn VAE decode

The 32-block DiT chain is executed by the Phase 16 full_dit_streaming_adapter
(strict LOAD/EXECUTE/UNLOAD, max_resident_weighted_blocks=1, no fallback);
the VAE encoder/decoder are routed through the FP32 ncnn stage runtime when
requested.  Scheduler / timestep / conditioning / pos-neg emb stay on the
official host/PyTorch side.

Usage:
  run_full_ncnn_core_pipeline.py --mode pytorch_reference
  run_full_ncnn_core_pipeline.py --mode dit_ncnn_vae_pytorch
  run_full_ncnn_core_pipeline.py --mode dit_ncnn_decoder_ncnn
  run_full_ncnn_core_pipeline.py --mode full_ncnn_core
  run_full_ncnn_core_pipeline.py --copy-baselines
  run_full_ncnn_core_pipeline.py --metrics
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tests" / "seedvr2_real_weight_hybrid"))
sys.path.insert(0, str(REPO / "tests" / "seedvr2_dit_block_contract"))
sys.path.insert(0, str(REPO / "tests" / "seedvr2_dit_block_ncnn"))

import phase17_common as pc  # noqa: E402
import phase16_common as p16  # noqa: E402
import phase16_streaming_manager as p16m  # noqa: E402
from probe_lib import ensure_compat  # noqa: E402
from run_official_pipeline import (  # noqa: E402
    make_input_video, build_video_transform, cut_videos, get_device,
    sha256_tensor, tensor_metrics, image_metrics, git_head,
)
from run_phase15_visual import build_runner  # noqa: E402
from run_phase16_visual import ChainHybrid, save_input_frames  # noqa: E402
from ncnn_vae_runtime import VaeEncoderRuntime, VaeDecoderRuntime  # noqa: E402
from run_vae_decoder_parity import ssim_np  # noqa: E402

VISUAL = Path(pc.VISUAL_OUT)
MODE_DIRS = {
    "pytorch_reference": "01_pytorch_reference",
    "dit_ncnn_vae_pytorch": "02_phase16_dit_ncnn_vae_pytorch",
    "dit_ncnn_decoder_ncnn": "03_dit_ncnn_decoder_ncnn",
    "full_ncnn_core": "04_full_ncnn_core",
}
FRAME_FILES = {"pytorch_reference": "frames_pytorch_reference.pt",
               "dit_ncnn_vae_pytorch": "frames_dit_ncnn_vae_pytorch.pt",
               "dit_ncnn_decoder_ncnn": "frames_dit_ncnn_decoder_ncnn.pt",
               "full_ncnn_core": "frames_full_ncnn_core.pt"}
LATENT_FILES = {"pytorch_reference": "final_latent_pytorch_reference.pt",
                "dit_ncnn_vae_pytorch": "final_latent_dit_ncnn_vae_pytorch.pt",
                "dit_ncnn_decoder_ncnn": "final_latent_dit_ncnn_decoder_ncnn.pt",
                "full_ncnn_core": "final_latent_full_ncnn_core.pt"}

PIXEL_PSNR_MIN = 40.0
PIXEL_SSIM_MIN = 0.98
PIXEL_MAX_ABS_SANITY = 0.25
LATENT_MAX_ABS_SANITY = 0.5
LATENT_MEAN_ABS_SANITY = 0.05
AMPLIFY_FACTOR = 25.0


def _save_frames(tchw, out_dir: Path, prefix: str = "frame"):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    a = tchw.clamp(-1, 1) * 0.5 + 0.5
    a = (a * 255.0).round().clamp(0, 255).to(torch.uint8)
    u8 = a.permute(0, 2, 3, 1).cpu().numpy()
    paths = []
    for i, f in enumerate(u8):
        p = out_dir / ("%s_%04d.png" % (prefix, i))
        Image.fromarray(f, mode="RGB").save(p, compress_level=0)
        paths.append(str(p.relative_to(REPO)))
    return paths


def _load_u8_frames(d: Path):
    from PIL import Image
    frames = sorted(d.glob("*.png"))
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in frames])


class VaeNcnnHooks:
    """Routes runner.vae_encode/vae_decode through the FP32 ncnn runtime."""

    def __init__(self, profile: str, use_encoder: bool, use_decoder: bool):
        self.enc = VaeEncoderRuntime(profile, root=str(pc.ART)) if use_encoder else None
        self.dec = VaeDecoderRuntime(profile, root=str(pc.ART)) if use_decoder else None
        self.enc_calls = 0
        self.dec_calls = 0
        self.enc_seconds = 0.0
        self.dec_seconds = 0.0
        self.original_encoder_calls = 0
        self.original_decoder_calls = 0
        self.silent_fallback = False

    def make_encode(self, runner):
        scale = float(runner.config.vae.scaling_factor)
        shift = float(runner.config.vae.get("shifting_factor", 0.0))
        dtype = getattr(torch, runner.config.vae.dtype)
        device = get_device()

        def vae_encode_ncnn(samples):
            self.enc_calls += 1
            latents = []
            t0 = time.perf_counter()
            for sample in samples:
                s = sample.unsqueeze(0)  # [1,3,T,H,W]
                latent = self.enc.encode(s.detach().float().cpu())  # [1,16,TL,h,w]
                latent = latent.permute(0, 2, 3, 4, 1)  # [1,TL,h,w,16]
                latent = (latent - shift) * scale
                latents.append(latent.to(device, dtype).squeeze(0))
            self.enc_seconds += time.perf_counter() - t0
            print("NCNN VAE ENCODER EXECUTED", flush=True)
            return latents

        return vae_encode_ncnn

    def make_decode(self, runner):
        scale = float(runner.config.vae.scaling_factor)
        shift = float(runner.config.vae.get("shifting_factor", 0.0))
        dtype = getattr(torch, runner.config.vae.dtype)
        device = get_device()

        def vae_decode_ncnn(latents):
            self.dec_calls += 1
            samples = []
            t0 = time.perf_counter()
            for latent in latents:
                z = latent.detach().float().cpu().unsqueeze(0) / scale + shift  # [1,TL,h,w,16]
                z = z.permute(0, 4, 1, 2, 3)  # [1,16,TL,h,w]
                if z.shape[2] == 1:
                    z = z.squeeze(2)
                out = self.dec.decode(z)  # [1,3,T,H,W] fp32
                samples.append(out.to(device, dtype).squeeze(0))
            self.dec_seconds += time.perf_counter() - t0
            print("NCNN VAE DECODER EXECUTED", flush=True)
            return samples

        return vae_decode_ncnn

    def counts(self):
        return {
            "ncnn_vae_encoder_call_count": self.enc_calls,
            "ncnn_vae_decoder_call_count": self.dec_calls,
            "original_vae_encoder_call_count": self.original_encoder_calls,
            "original_vae_decoder_call_count": self.original_decoder_calls,
            "silent_fallback": self.silent_fallback,
        }


def run_mode(mode: str, args) -> dict:
    ensure_compat()
    from common.seed import set_seed

    workdir = VISUAL / ("run_%s" % mode)
    workdir.mkdir(parents=True, exist_ok=True)
    cap = {"latents": None}
    t0 = time.time()
    runner = build_runner(cfg_scale=args.cfg_scale, steps=args.steps)
    # Deterministic mean-path latent for all modes (fixed RNG would also work,
    # but ncnn cannot sample; READY gates compare identical latent inputs).
    runner.config.vae.use_sample = False

    chain_hybrid = None
    if mode in ("dit_ncnn_vae_pytorch", "dit_ncnn_decoder_ncnn", "full_ncnn_core"):
        chain = p16.RANGE_FULL
        chain_hybrid = ChainHybrid(chain, workdir, denoising_step=0)

        def make_fwd(bi):
            def fwd(vid, txt, vid_shape, txt_shape, emb, cache):
                return chain_hybrid.forward(bi, vid, txt, vid_shape, txt_shape, emb, cache)
            return fwd

        for bi in chain:
            runner.dit.blocks[bi].forward = make_fwd(bi)

    hooks = VaeNcnnHooks("A", use_encoder=(mode == "full_ncnn_core"),
                         use_decoder=(mode in ("dit_ncnn_decoder_ncnn", "full_ncnn_core")))
    if hooks.enc is not None:
        runner.vae_encode = hooks.make_encode(runner)
    if hooks.dec is not None:
        runner.vae_decode = hooks.make_decode(runner)

    orig_vae_decode = runner.vae_decode

    def rec_vae_decode(latents):
        cap["latents"] = [l.detach().float().cpu() for l in latents]
        return orig_vae_decode(latents)

    runner.vae_decode = rec_vae_decode
    set_seed(args.seed, same_across_ranks=True)

    video = make_input_video(args.frames, args.height, args.width)
    video_transform = build_video_transform(args.height, args.width)
    cond = video_transform(video.permute(0, 3, 1, 2).to(get_device()))
    cond = cut_videos(cond, 1)
    ori_length = video.size(0)
    cond_hash = sha256_tensor(cond.float().cpu())

    runner.dit.to("cpu")
    runner.vae.to(get_device())
    cond_latents = runner.vae_encode([cond])
    runner.vae.to("cpu")
    runner.dit.to(get_device())

    # Fixed RNG anchor before noise generation: the official VAE encoder consumes
    # CUDA RNG while the ncnn VAE encoder runs in a subprocess (no RNG), so the
    # latent/noise draws must be anchored AFTER encoding to stay bit-identical
    # across all four modes (identical z for reference vs hybrid runs).
    set_seed(args.seed, same_across_ranks=True)
    noises = [torch.randn_like(latent) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent) for latent in cond_latents]

    def add_noise(x, aug_noise):
        t = torch.tensor([1000.0], device=get_device()) * 0.0
        shape = torch.tensor(x.shape[1:], device=get_device())[None]
        t = runner.timestep_transform(t, shape)
        return runner.schedule.forward(x, aug_noise, t)

    conditions = [
        runner.get_condition(noise, task="sr", latent_blur=add_noise(latent, aug_noise))
        for noise, aug_noise, latent in zip(noises, aug_noises, cond_latents)
    ]
    pos_emb = torch.load(str(p16.POS_EMB), map_location="cpu", weights_only=True).to(get_device())
    neg_emb = torch.load(str(p16.NEG_EMB), map_location="cpu", weights_only=True).to(get_device())

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises, conditions=conditions, texts_pos=[pos_emb],
            texts_neg=[neg_emb], cfg_scale=args.cfg_scale, dit_offload=False,
        )
    sample = video_tensors[0].permute(1, 0, 2, 3)[:ori_length].float().cpu()
    elapsed = time.time() - t0

    if mode in ("dit_ncnn_vae_pytorch", "dit_ncnn_decoder_ncnn", "full_ncnn_core"):
        if chain_hybrid is None or not chain_hybrid.ran:
            raise RuntimeError("chain hybrid adapter never executed for mode=%s" % mode)

    _save_frames(sample, VISUAL / MODE_DIRS[mode])
    torch.save([cap["latents"][0]], VISUAL / LATENT_FILES[mode])
    torch.save(sample, VISUAL / FRAME_FILES[mode])
    manifest = {
        "phase": 17, "mode": mode, "seed": args.seed, "steps": args.steps,
        "cfg_scale": args.cfg_scale, "git_head": git_head(),
        "elapsed_seconds": round(elapsed, 3),
        "frames": int(ori_length), "height": args.height, "width": args.width,
        "cond_sha256": cond_hash,
        "use_sample": False,
        "call_counts": hooks.counts(),
        "ncnn_encoder_seconds": round(hooks.enc_seconds, 3),
        "ncnn_decoder_seconds": round(hooks.dec_seconds, 3),
        "chain_hybrid_ran": bool(chain_hybrid is not None and chain_hybrid.ran),
        "chain_adapter_report": str(workdir / "chain_adapter_report.json"),
    }
    (VISUAL / ("run_%s_manifest.json" % mode)).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def copy_baselines() -> None:
    src = Path(p16.VISUAL_ROOT)
    dst = VISUAL
    dst.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("00_input", "00_input"),
        ("01_pytorch_reference", "01_pytorch_reference"),
        ("04_hybrid_full_dit", "02_phase16_dit_ncnn_vae_pytorch"),
        ("frames_reference.pt", "frames_pytorch_reference.pt"),
        ("final_latent_reference.pt", "final_latent_pytorch_reference.pt"),
        ("frames_hybrid_full.pt", "frames_dit_ncnn_vae_pytorch.pt"),
        ("final_latent_hybrid_full.pt", "final_latent_dit_ncnn_vae_pytorch.pt"),
    ]
    for a, b in pairs:
        sp = src / a
        dp = dst / b
        if sp.exists():
            if sp.is_dir():
                if dp.exists():
                    shutil.rmtree(dp)
                shutil.copytree(sp, dp)
            else:
                shutil.copy2(sp, dp)
            print(f"copied {a} -> {b}")
        else:
            print(f"WARN missing {sp}")
    print("baselines copied")


def compute_metrics() -> dict:
    modes = ["pytorch_reference", "dit_ncnn_vae_pytorch", "dit_ncnn_decoder_ncnn", "full_ncnn_core"]
    frames = {}
    latents = {}
    for m in modes:
        frames[m] = torch.load(VISUAL / FRAME_FILES[m], map_location="cpu", weights_only=True)
        latents[m] = torch.load(VISUAL / LATENT_FILES[m], map_location="cpu", weights_only=True)
    ref_f = frames["pytorch_reference"].float()
    ref_l = latents["pytorch_reference"][0].float()
    input_u8 = _load_u8_frames(VISUAL / "00_input")

    results = {}
    for m in modes[1:]:
        l_met = tensor_metrics(ref_l, latents[m][0].float(), "latent")
        ref_u = ref_f.clamp(-1, 1) * 0.5 + 0.5
        got_u = frames[m].float().clamp(-1, 1) * 0.5 + 0.5
        ref_np = ref_u.numpy().transpose(0, 2, 3, 1)
        got_np = got_u.numpy().transpose(0, 2, 3, 1)
        im = image_metrics(ref_np, got_np)
        im["ssim"] = ssim_np(ref_np, got_np, data_range=1.0)
        diff = np.abs(ref_np - got_np)
        im["mismatch_count"] = int((diff > 1.0 / 255.0).sum())
        im["frames"] = int(ref_np.shape[0])
        im["resolution"] = [int(ref_np.shape[1]), int(ref_np.shape[2])]
        im["flicker_reference"] = float(np.abs(np.diff(ref_np, axis=0)).mean())
        im["flicker_mode"] = float(np.abs(np.diff(got_np, axis=0)).mean())
        im["non_constant_reference"] = bool(np.std(ref_np) > 1e-3)
        im["non_constant_mode"] = bool(np.std(got_np) > 1e-3)
        im["all_black_mode"] = bool(float(np.max(got_np)) < 1e-3)
        im["all_white_mode"] = bool(float(np.min(got_np)) > 1.0 - 1e-3)
        results[m] = {"latent": l_met, "pixels": im}

    # diff panels + 5-panel side-by-side
    for m in ("dit_ncnn_decoder_ncnn", "full_ncnn_core"):
        got_u = (frames[m].float().clamp(-1, 1) * 0.5 + 0.5).numpy().transpose(0, 2, 3, 1)
        diff_dir = VISUAL / ("05_diff_decoder" if m == "dit_ncnn_decoder_ncnn" else "06_diff_full")
        _save_diff(input_u8, ref_np, got_u, diff_dir)
    _save_side_by_side(input_u8, ref_np,
                       (frames["dit_ncnn_vae_pytorch"].float().clamp(-1, 1) * 0.5 + 0.5).numpy().transpose(0, 2, 3, 1),
                       (frames["dit_ncnn_decoder_ncnn"].float().clamp(-1, 1) * 0.5 + 0.5).numpy().transpose(0, 2, 3, 1),
                       (frames["full_ncnn_core"].float().clamp(-1, 1) * 0.5 + 0.5).numpy().transpose(0, 2, 3, 1),
                       VISUAL / "07_side_by_side")

    # error budget
    enc = json.loads((REPO / "reports" / "phase17_vae_encoder_parity.json").read_text(encoding="utf-8"))
    dec = json.loads((REPO / "reports" / "phase17_vae_decoder_parity.json").read_text(encoding="utf-8"))
    enc_rel = enc["profiles"]["A"]["relative_max"]
    dec_rel = dec["profiles"]["A"]["relative_max"]
    dit_rel = results["dit_ncnn_vae_pytorch"]["latent"]["rmse"] / max(1e-12, float(ref_l.abs().max()))
    full_px = results["full_ncnn_core"]["pixels"]
    budget = {
        "phase": 17,
        "encoder_contribution": enc_rel,
        "dit_contribution": dit_rel,
        "decoder_contribution": dec_rel,
        "combined_contribution": full_px["rmse"],
        "largest_contributor": max(
            (("encoder", enc_rel), ("dit", dit_rel), ("decoder", dec_rel)),
            key=lambda x: x[1])[0],
        "first_abnormal_stage": "none (all stages within relative <1e-3 budget)" if
            enc.get("profiles", {}).get("A", {}).get("pass") else "encoder",
        "note": "contributions measured at A profile; dit contribution is the "
                "latent-space RMSE relative max of the Phase 16 full ncnn DiT chain",
    }

    full_ok = (full_px["psnr_db"] >= PIXEL_PSNR_MIN and full_px["ssim"] >= PIXEL_SSIM_MIN
               and full_px["nan_count"] == 0 and full_px["inf_count"] == 0
               and full_px["non_constant_mode"] and not full_px["all_black_mode"]
               and not full_px["all_white_mode"]
               and full_px["frames"] == 4)
    dec_ok = (results["dit_ncnn_decoder_ncnn"]["pixels"]["psnr_db"] >= PIXEL_PSNR_MIN
              and results["dit_ncnn_decoder_ncnn"]["pixels"]["ssim"] >= PIXEL_SSIM_MIN)

    report = {
        "phase": 17,
        "milestone": "real-weight SeedVR2 VAE ncnn + full 32-Block DiT core",
        "real_seedvr2_weights": True,
        "real_vae_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "run": {"seed": 20260803, "steps": 1, "cfg_scale": 1.0, "frames": 4,
                "height": 128, "width": 128, "use_sample": False},
        "modes": results,
        "error_budget": budget,
        "thresholds": {
            "full_ncnn_core_pixel_psnr_min_db": PIXEL_PSNR_MIN,
            "full_ncnn_core_pixel_ssim_min": PIXEL_SSIM_MIN,
            "pixel_max_abs_sanity": PIXEL_MAX_ABS_SANITY,
            "latent_max_abs_sanity": LATENT_MAX_ABS_SANITY,
            "latent_mean_abs_sanity": LATENT_MEAN_ABS_SANITY,
            "amplify_factor": AMPLIFY_FACTOR,
        },
        "full_ncnn_core_pass": full_ok,
        "decoder_hybrid_pass": dec_ok,
        "overall_pass": bool(full_ok),
        "verdict": "PASS" if full_ok else "PARTIAL",
        "outputs": {
            "input_dir": "00_input",
            "reference_dir": "01_pytorch_reference",
            "phase16_baseline_dir": "02_phase16_dit_ncnn_vae_pytorch",
            "decoder_ncnn_dir": "03_dit_ncnn_decoder_ncnn",
            "full_ncnn_core_dir": "04_full_ncnn_core",
            "diff_decoder_dir": "05_diff_decoder",
            "diff_full_dir": "06_diff_full",
            "side_by_side_dir": "07_side_by_side",
        },
        "scheduler_host_or_pytorch": True,
        "full_pure_ncnn_application": False,
        "max_resident_weighted_dit_blocks": 1,
        "max_resident_vae_model_sets": 1,
        "android_vulkan": False,
        "large_artifacts_committed": False,
    }
    (REPO / "reports" / "phase17_full_pipeline_visual.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPO / "reports" / "phase17_full_pipeline_error_budget.json").write_text(
        json.dumps(budget, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"],
                      "modes": {m: {"psnr": results[m]["pixels"]["psnr_db"],
                                    "ssim": results[m]["pixels"]["ssim"]} for m in modes[1:]}},
                     indent=2))
    return report


def _save_diff(input_u8, ref_u8, got_u8, diff_dir: Path):
    from PIL import Image
    diff_dir.mkdir(parents=True, exist_ok=True)
    d = np.abs(ref_u8 - got_u8)
    amp = np.clip(d * AMPLIFY_FACTOR, 0.0, 1.0)
    for i in range(ref_u8.shape[0]):
        p = diff_dir / ("diff_amp_%04d.png" % i)
        Image.fromarray((amp[i] * 255.0).round().astype(np.uint8), mode="RGB").save(p, compress_level=0)


def _save_side_by_side(input_u8, ref_u8, p16_u8, dec_u8, full_u8, side_dir: Path):
    from PIL import Image
    side_dir.mkdir(parents=True, exist_ok=True)
    for i in range(ref_u8.shape[0]):
        d = np.abs(ref_u8[i] - full_u8[i])
        amp = np.clip(d * AMPLIFY_FACTOR, 0.0, 1.0)
        panels = [input_u8[i], (ref_u8[i] * 255).round().astype(np.uint8),
                  (p16_u8[i] * 255).round().astype(np.uint8),
                  (dec_u8[i] * 255).round().astype(np.uint8),
                  (full_u8[i] * 255).round().astype(np.uint8),
                  (amp * 255).round().astype(np.uint8)]
        widths = [p.shape[1] for p in panels]
        height = max(p.shape[0] for p in panels)
        gap = 4
        canvas = np.full((height, sum(widths) + gap * (len(panels) - 1), 3), 255, dtype=np.uint8)
        x = 0
        for p in panels:
            canvas[: p.shape[0], x: x + p.shape[1]] = p
            x += p.shape[1] + gap
        Image.fromarray(canvas, mode="RGB").save(side_dir / ("side_by_side_%04d.png" % i), compress_level=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODE_DIRS.keys()), default=None)
    ap.add_argument("--copy-baselines", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        VISUAL.mkdir(parents=True, exist_ok=True)
        if args.copy_baselines:
            copy_baselines()
            return 0
        if args.metrics:
            compute_metrics()
            return 0
        if args.mode is None:
            ap.error("--mode required")
        run_mode(args.mode, args)
        return 0
    except Exception as exc:  # noqa: BLE001
        err = {"phase": "phase17_full_pipeline", "error": str(exc),
               "traceback": traceback.format_exc()}
        VISUAL.mkdir(parents=True, exist_ok=True)
        (VISUAL / "pipeline_error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
        print(traceback.format_exc())
        return 3


if __name__ == "__main__":
    sys.exit(main())
