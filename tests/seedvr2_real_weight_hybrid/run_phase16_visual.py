#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: official SeedVR2 3B pipeline with 8/16/full DiT blocks replaced
by the full_dit_streaming_adapter (one strict chain adapter call per hybrid
run, max_resident_weighted_blocks=1, no silent fallback).

Modes
-----
reference
    Official PyTorch reference run (all 32 blocks, bf16 autocast, seed
    20260803, steps=1, 4 frames 128x128, cfg 1.0).  Saves input/reference
    frames, final latent and a manifest under .phase16_visual_outputs/.
    (The Phase 16 capture run already produced these artifacts; this mode is
    kept for reproducibility.)

hybrid --hybrid-block-count 8|16|full
    Official pipeline with the chain range [9..16] / [8..23] / [0..31]
    replaced by ONE full_dit_streaming_adapter call (strict streaming, no
    fallback).  Saves hybrid frames, final latent, per-block ncnn outputs and
    the adapter report under .phase16_visual_outputs/.

metrics
    Load saved reference + hybrid artifacts, compute latent/pixel
    RMSE/PSNR/SSIM/flicker/non-constant metrics, build diff and side-by-side
    panels, aggregate the error-growth curve (E1/2/4 from Phase 15 reports,
    E8/16/full from this phase) and write
    reports/full_dit_visual_comparison.json + reports/full_dit_error_growth.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_ncnn"))

import phase16_common as p16  # noqa: E402
import phase16_streaming_manager as p16m  # noqa: E402
from probe_lib import ensure_compat  # noqa: E402
from run_official_pipeline import (  # noqa: E402
    capture_inputs, write_adapter_input, make_input_video, build_video_transform,
    cut_videos, get_device, tensor_metrics, image_metrics, load_png_frames,
    make_diff_and_side_by_side, git_head, sha256_file, sha256_tensor,
)
from run_phase15_visual import build_runner, save_frames, save_input_frames  # noqa: E402

C = p16.C
EMB = p16.EMB
VISUAL = p16.VISUAL_ROOT
CHAIN_RANGES = {"8": p16.RANGE_8, "16": p16.RANGE_16, "full": p16.RANGE_FULL}
HYBRID_DIRS = {"8": "02_hybrid_8_blocks", "16": "03_hybrid_16_blocks", "full": "04_hybrid_full_dit"}
DIFF_DIRS = {"8": "05_diff_8", "16": "06_diff_16", "full": "07_diff_full"}
LATENT_FILES = {"8": "final_latent_hybrid_8.pt", "16": "final_latent_hybrid_16.pt",
                "full": "final_latent_hybrid_full.pt"}
FRAME_FILES = {"8": "frames_hybrid_8.pt", "16": "frames_hybrid_16.pt",
               "full": "frames_hybrid_full.pt"}
MANIFEST_FILES = {"8": "run_hybrid_8_manifest.json", "16": "run_hybrid_16_manifest.json",
                  "full": "run_hybrid_full_manifest.json"}

PIXEL_MAX_ABS_SANITY = 0.25
LATENT_MAX_ABS_SANITY = 0.5
LATENT_MEAN_ABS_SANITY = 0.05
NON_CONSTANT_MIN_STD = 1e-3
AMPLIFY_FACTOR = 25.0


class ChainHybrid:
    """Replaces a contiguous chain of block forwards with ONE adapter call.

    The adapter executes the whole chain (strict LOAD/EXECUTE/UNLOAD order,
    max resident = 1) from the input captured at the chain's first block; the
    per-block outputs are then served to the pipeline block forwards so the
    ncnn chain result flows end-to-end without any PyTorch fallback.
    """

    def __init__(self, chain_blocks, workdir: Path, denoising_step: int = 0):
        self.chain = list(chain_blocks)
        self.first = self.chain[0]
        self.workdir = workdir
        self.denoising_step = denoising_step
        self.outputs = {}
        self.output_hashes = {}
        self.input_hashes = {}
        self.report = None
        self.stdout = ""
        self.rc = -1
        self.calls = {bi: 0 for bi in self.chain}
        self.ran = False

    def forward(self, bi, vid, txt, vid_shape, txt_shape, emb, cache=None):
        if bi not in self.calls:
            raise RuntimeError("chain hybrid called for out-of-range block %d" % bi)
        self.calls[bi] += 1
        if self.calls[bi] > 1:
            raise RuntimeError("block %d called %d times (expected 1 with steps=1)"
                               % (bi, self.calls[bi]))
        if not self.ran and bi != self.first:
            raise RuntimeError("chain hybrid first call must be block %d, got %d"
                               % (self.first, bi))
        if not self.ran:
            if cache is not None:
                self._prime_vid_ada_cache(cache, emb, vid_shape)
            inp = capture_inputs(vid, txt, vid_shape, txt_shape, emb)
            self.input_hashes = {k: sha256_tensor(inp[k]) for k in ("vid", "txt", "emb")}
            input_dir = self.workdir / "chain_input"
            output_dir = self.workdir / "chain_output"
            report_path = self.workdir / "chain_adapter_report.json"
            write_adapter_input(inp, input_dir)
            self.report, self.stdout, self.rc = p16m.run_adapter(
                self.chain, input_dir, output_dir, load_mode="streaming",
                report_path=report_path, denoising_step=self.denoising_step,
                timeout_s=7200)
            if self.rc != 0:
                raise RuntimeError("adapter rc=%d for chain %s stderr=%s"
                                   % (self.rc, self.chain, (self.stdout or "")[-2000:]))
            if "HYBRID BLOCK EXECUTED" not in (self.stdout or ""):
                raise RuntimeError("adapter did not print HYBRID BLOCK EXECUTED for chain %s"
                                   % (self.chain,))
            if self.report is None:
                raise RuntimeError("adapter report missing for chain %s" % (self.chain,))
            for bj in self.chain:
                v = np.fromfile(str(output_dir / ("block_%d" % bj) / "output_vid.bin"),
                                dtype=np.float32).reshape(-1, C)
                t = np.fromfile(str(output_dir / ("block_%d" % bj) / "output_txt.bin"),
                                dtype=np.float32).reshape(-1, C)
                self.outputs[bj] = (v, t)
                self.output_hashes[str(bj)] = {
                    "vid": sha256_file(output_dir / ("block_%d" % bj) / "output_vid.bin"),
                    "txt": sha256_file(output_dir / ("block_%d" % bj) / "output_txt.bin"),
                }
            self.ran = True
        v, t = self.outputs[bi]
        vid_out = torch.from_numpy(v).to(device=vid.device, dtype=vid.dtype)
        txt_out = torch.from_numpy(t).to(device=txt.device, dtype=txt.dtype)
        return vid_out, txt_out, vid_shape, txt_shape

    def _prime_vid_ada_cache(self, cache, emb, vid_shape):
        """Recreate the official cache entry 'emb_repeat_0_vid'.

        The official vid_out_ada reuses the cache key that a real block's
        attn-in vid modulation writes (cache keys only carry idx+branch_tag,
        not layer/mode).  With the whole chain replaced by ncnn no real block
        runs, so this weight-independent entry (pure function of emb and
        vid_len) is rebuilt exactly as a real block would have built it:
        rearrange with l=len(['attn','mlp'])=2, g=3, then repeat by vid_len.
        """
        from einops import rearrange
        from common.distributed.ops import slice_inputs
        emb_r = rearrange(emb, "b (d l g) -> b d l g", l=2, g=3)[..., 0, :]
        vid_len = int(vid_shape.reshape(-1).prod())
        cached = slice_inputs(
            torch.cat([e.repeat(int(vl), *([1] * e.ndim))
                       for e, vl in zip(emb_r, [vid_len])]), dim=0)
        cache("emb_repeat_0_vid", lambda: cached)


def run_official_reference(args) -> dict:
    ensure_compat()
    from common.seed import set_seed
    cap = {"latents": None}
    t0 = time.time()
    runner = build_runner(cfg_scale=args.cfg_scale, steps=args.steps, fp32_blocks=())
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

    runner.dit.to("cpu")
    runner.vae.to(get_device())
    cond_latents = runner.vae_encode([cond])
    runner.vae.to("cpu")
    runner.dit.to(get_device())

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

    (VISUAL / "00_input").mkdir(parents=True, exist_ok=True)
    save_input_frames(video, VISUAL / "00_input")
    save_frames(sample, VISUAL / "01_pytorch_reference")
    torch.save([cap["latents"][0]], VISUAL / "final_latent_reference.pt")
    torch.save(sample, VISUAL / "frames_reference.pt")

    manifest = {
        "phase": 16, "mode": "reference", "seed": args.seed, "steps": args.steps,
        "cfg_scale": args.cfg_scale, "git_head": git_head(),
        "elapsed_seconds": round(elapsed, 3),
        "frames": int(ori_length), "height": args.height, "width": args.width,
    }
    (VISUAL / "run_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def run_hybrid(args, count: str) -> dict:
    chain = CHAIN_RANGES[count]
    ensure_compat()
    from common.seed import set_seed
    cap = {"latents": None}
    workdir = VISUAL / ("run_hybrid_%s" % count)
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runner = build_runner(cfg_scale=args.cfg_scale, steps=args.steps, fp32_blocks=())
    chain_hybrid = ChainHybrid(chain, workdir, denoising_step=0)
    orig_forwards = {bi: runner.dit.blocks[bi].forward for bi in chain}

    def make_fwd(bi):
        def fwd(vid, txt, vid_shape, txt_shape, emb, cache):
            return chain_hybrid.forward(bi, vid, txt, vid_shape, txt_shape, emb, cache)
        return fwd

    for bi in chain:
        runner.dit.blocks[bi].forward = make_fwd(bi)

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

    runner.dit.to("cpu")
    runner.vae.to(get_device())
    cond_latents = runner.vae_encode([cond])
    runner.vae.to("cpu")
    runner.dit.to(get_device())

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

    if not chain_hybrid.ran:
        raise RuntimeError("chain hybrid adapter never executed for count=%s" % count)
    if any(c != 1 for c in chain_hybrid.calls.values()):
        raise RuntimeError("unexpected block call counts: %s" % chain_hybrid.calls)

    save_frames(sample, VISUAL / HYBRID_DIRS[count])
    torch.save([cap["latents"][0]], VISUAL / LATENT_FILES[count])
    torch.save(sample, VISUAL / FRAME_FILES[count])

    rep = chain_hybrid.report
    adapter_evidence = {
        "ncnn_adapter_call_count": int(rep.get("ncnn_adapter_call_count", -1)),
        "original_block_call_count": int(rep.get("original_block_call_count", -1)),
        "silent_fallback": bool(rep.get("silent_fallback", True)),
        "order_ok": bool(rep.get("order_ok", False)),
        "streaming_mode_validated": bool(rep.get("streaming_mode_validated", False)),
        "max_resident_weighted_blocks": int(rep.get("max_resident_weighted_blocks", -1)),
        "state_machine_final": rep.get("state_machine_final"),
        "state_machine_valid": bool(rep.get("state_machine_valid", False)),
        "expected_next_block_index": int(rep.get("expected_next_block_index", -1)),
        "events": len(rep.get("events", [])),
        "rc": chain_hybrid.rc,
        "stdout_has_hybrid_block_executed": "HYBRID BLOCK EXECUTED" in (chain_hybrid.stdout or ""),
        "chain": chain,
        "input_hashes": chain_hybrid.input_hashes,
        "output_hashes": chain_hybrid.output_hashes,
        "block_call_count_map": chain_hybrid.calls,
    }

    manifest = {
        "phase": 16, "mode": "hybrid", "count": count, "blocks": chain,
        "seed": args.seed, "steps": args.steps, "cfg_scale": args.cfg_scale,
        "git_head": git_head(), "elapsed_seconds": round(elapsed, 3),
        "frames": int(ori_length), "height": args.height, "width": args.width,
        "adapter": adapter_evidence,
    }
    (VISUAL / MANIFEST_FILES[count]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def _flicker(frames):
    """frames: float32 [T,H,W,C] in [0,1].  Mean frame-to-frame abs change."""
    if frames.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(frames, axis=0)).mean())


def compute_metrics() -> dict:
    ref_man = json.loads((VISUAL / "run_reference_manifest.json").read_text("utf-8")) \
        if (VISUAL / "run_reference_manifest.json").exists() else {}
    ref_frames = load_png_frames(VISUAL / "01_pytorch_reference")
    input_u8 = load_png_frames(VISUAL / "00_input")
    ref_lat = torch.load(VISUAL / "final_latent_reference.pt", map_location="cpu",
                         weights_only=True)[0]
    ref_lat_np = ref_lat.float().numpy()

    modes = {}
    verdict_reasons = []
    for count in ("8", "16", "full"):
        man_file = VISUAL / MANIFEST_FILES[count]
        if not man_file.exists():
            verdict_reasons.append("%s: hybrid manifest missing" % count)
            continue
        man = json.loads(man_file.read_text("utf-8"))
        hyb_frames = load_png_frames(VISUAL / HYBRID_DIRS[count])
        hyb_lat = torch.load(VISUAL / LATENT_FILES[count], map_location="cpu",
                             weights_only=True)[0]
        hyb_lat_np = hyb_lat.float().numpy()
        assert hyb_frames.shape == ref_frames.shape, (hyb_frames.shape, ref_frames.shape)
        assert hyb_lat_np.shape == ref_lat_np.shape, (hyb_lat_np.shape, ref_lat_np.shape)
        pixel = image_metrics(ref_frames, hyb_frames)
        pixel["output_shape"] = list(ref_frames.shape)
        pixel["frames"] = int(ref_frames.shape[0])
        pixel["flicker_reference"] = _flicker(ref_frames)
        pixel["flicker_hybrid"] = _flicker(hyb_frames)
        pixel["non_constant"] = bool(float(hyb_frames.std()) > NON_CONSTANT_MIN_STD)
        latent = tensor_metrics(ref_lat, hyb_lat, "final_latent")
        modes[count] = {
            "blocks": man.get("blocks", CHAIN_RANGES[count]),
            "pixel": pixel,
            "latent": latent,
            "adapter": man["adapter"],
            "elapsed_seconds": man.get("elapsed_seconds"),
        }
        if pixel["nan_count"] or pixel["inf_count"] or latent["nan_count"] or latent["inf_count"]:
            verdict_reasons.append("%s: NaN/Inf detected" % count)
        if count == "full":
            if pixel["psnr_db"] < p16.VISUAL_PSNR_MIN_FULL:
                verdict_reasons.append("full PSNR %.3f < %.1f" % (pixel["psnr_db"], p16.VISUAL_PSNR_MIN_FULL))
            if pixel["ssim"] < p16.VISUAL_SSIM_MIN_FULL:
                verdict_reasons.append("full SSIM %.4f < %.2f" % (pixel["ssim"], p16.VISUAL_SSIM_MIN_FULL))
            if pixel["max_abs_error"] > PIXEL_MAX_ABS_SANITY:
                verdict_reasons.append("full pixel max_abs %.4f > %.2f sanity" % (pixel["max_abs_error"], PIXEL_MAX_ABS_SANITY))
        print("%s hybrid: pixel_psnr=%.3f ssim=%.4f rmse=%.6f | latent_rmse=%.6f"
              % (count, pixel["psnr_db"], pixel["ssim"], pixel["rmse"], latent["rmse"]), flush=True)

    diff_dir_root = VISUAL / "09_error_growth"
    diff_dir_root.mkdir(parents=True, exist_ok=True)
    for count in modes:
        hyb_frames = load_png_frames(VISUAL / HYBRID_DIRS[count])
        make_diff_and_side_by_side(ref_frames, hyb_frames, input_u8,
                                   VISUAL / DIFF_DIRS[count],
                                   VISUAL / ("08_side_by_side_%s" % count),
                                   factor=AMPLIFY_FACTOR)
    side_paths = sorted(str(p) for p in (VISUAL / "08_side_by_side_full").glob("*.png")) \
        if (VISUAL / "08_side_by_side_full").exists() else []

    # Error-growth curve: E1/2/4 reuse Phase 15 (multiblock_visual_comparison.json).
    growth = {}
    p15rep = None
    p15_path = p16.REPO / "reports" / "multiblock_visual_comparison.json"
    if p15_path.exists():
        p15rep = json.loads(p15_path.read_text("utf-8"))
        for n in ("1", "2", "4"):
            m = p15rep.get("modes", {}).get(n)
            if m:
                px = m.get("pixels") or m.get("pixel") or {}
                growth["E%s" % n] = {
                    "blocks": m.get("blocks"),
                    "latent_rmse": m.get("latent", {}).get("rmse"),
                    "latent_psnr_db": m.get("latent", {}).get("psnr_db"),
                    "pixel_rmse": px.get("rmse"),
                    "pixel_psnr_db": px.get("psnr_db"),
                    "pixel_ssim": px.get("ssim"),
                }
    for count in modes:
        m = modes[count]
        growth["E%s" % count] = {
            "blocks": m["blocks"],
            "latent_rmse": m["latent"]["rmse"],
            "latent_psnr_db": m["latent"]["psnr_db"],
            "pixel_rmse": m["pixel"]["rmse"],
            "pixel_psnr_db": m["pixel"]["psnr_db"],
            "pixel_ssim": m["pixel"]["ssim"],
        }
    growth_report = {
        "phase": 16,
        "real_seedvr2_weights": True,
        "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
        "error_growth": growth,
        "notes": {
            "E1_E2_E4": "reused from reports/multiblock_visual_comparison.json (Phase 15, blocks 15/16/17/18)",
            "E8": "blocks 9..16", "E16": "blocks 8..23", "EFull": "blocks 0..31",
        },
    }
    (p16.REPO / "reports" / "full_dit_error_growth.json").write_text(
        json.dumps(growth_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    full_ok = "full" in modes and not verdict_reasons
    report = {
        "phase": 16,
        "milestone": "full SeedVR2 DiT streaming visual hybrid",
        "real_seedvr2_weights": True,
        "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
        "run": {
            "seed": ref_man.get("seed", 20260803), "steps": ref_man.get("steps", 1),
            "cfg_scale": ref_man.get("cfg_scale", 1.0),
            "frames": ref_man.get("frames", 4), "height": ref_man.get("height", 128),
            "width": ref_man.get("width", 128),
        },
        "modes": modes,
        "thresholds": {
            "full_psnr_min_db": p16.VISUAL_PSNR_MIN_FULL,
            "full_ssim_min": p16.VISUAL_SSIM_MIN_FULL,
            "pixel_max_abs_sanity": PIXEL_MAX_ABS_SANITY,
            "latent_max_abs_sanity": LATENT_MAX_ABS_SANITY,
            "latent_mean_abs_sanity": LATENT_MEAN_ABS_SANITY,
            "amplify_factor": AMPLIFY_FACTOR,
        },
        "overall_pass": bool(full_ok),
        "verdict": "PASS" if full_ok else "FAIL",
        "verdict_reasons": verdict_reasons or ["all Phase 16 visual thresholds satisfied"],
        "outputs": {
            "diff_dirs": {c: str((VISUAL / DIFF_DIRS[c]).relative_to(p16.REPO)) for c in modes},
            "side_by_side_dir": "08_side_by_side_full" if (VISUAL / "08_side_by_side_full").exists() else None,
            "side_by_side_frames": side_paths,
            "error_growth_report": "reports/full_dit_error_growth.json",
        },
    }
    (p16.REPO / "reports" / "full_dit_visual_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "modes": {c: {
        "pixel_psnr": m["pixel"]["psnr_db"], "pixel_ssim": m["pixel"]["ssim"],
        "latent_rmse": m["latent"]["rmse"]} for c, m in modes.items()}}, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "hybrid", "metrics"], required=True)
    ap.add_argument("--hybrid-block-count", choices=["8", "16", "full"], default=None)
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
        if args.mode == "reference":
            run_official_reference(args)
            return 0
        if args.mode == "hybrid":
            if args.hybrid_block_count is None:
                raise SystemExit("--hybrid-block-count required for hybrid mode")
            run_hybrid(args, args.hybrid_block_count)
            return 0
        compute_metrics()
        return 0
    except Exception as exc:  # noqa: BLE001
        err = {"phase": "phase16_visual", "error": str(exc), "traceback": traceback.format_exc()}
        VISUAL.mkdir(parents=True, exist_ok=True)
        (VISUAL / "pipeline_error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
        print(traceback.format_exc())
        return 3


if __name__ == "__main__":
    sys.exit(main())
