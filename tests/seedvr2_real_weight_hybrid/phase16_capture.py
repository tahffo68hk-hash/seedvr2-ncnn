#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: capture official-context inputs for ALL 32 real blocks.

Runs the official SeedVR2 3B pipeline (reference mode, all PyTorch blocks,
bf16 autocast, seed=20260803, steps=1, 4 frames 128x128, cfg 1.0) with
observation-only pre-hooks on every block 0..31 recording the input
vid/txt/emb/shapes of the FIRST invocation of each block.  Saves the inputs
under .phase16_local_artifacts/official_context_capture/block_<i>/ and writes
a capture manifest.  No block forward is replaced; the numeric path is the
official reference run (also saved as the visual reference).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_ncnn"))

import phase16_common as p16  # noqa: E402
from probe_lib import ensure_compat  # noqa: E402
from run_official_pipeline import (  # noqa: E402
    capture_inputs, make_input_video, build_video_transform, cut_videos,
    get_device, git_head, sha256_tensor,
)
from run_phase15_visual import build_runner, save_frames, save_input_frames  # noqa: E402

CAPTURE = p16.PHASE16_ROOT / "official_context_capture"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    args = ap.parse_args()

    ensure_compat()
    from common.seed import set_seed
    cap = {"calls": {i: 0 for i in p16.ALL_BLOCKS}, "inputs": {}}
    t0 = time.time()
    runner = build_runner(cfg_scale=args.cfg_scale, steps=args.steps, fp32_blocks=())

    for bi in p16.ALL_BLOCKS:
        blk = runner.dit.blocks[bi]
        blk.register_forward_pre_hook(
            lambda module, args, kwargs, bi=bi: _capture(cap, bi, args, kwargs),
            with_kwargs=True)

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

    missing = [bi for bi in p16.ALL_BLOCKS if bi not in cap["inputs"]]
    if missing:
        print("MISSING BLOCK INPUTS:", missing)
        return 1

    CAPTURE.mkdir(parents=True, exist_ok=True)
    per_block = {}
    for bi in p16.ALL_BLOCKS:
        bdir = CAPTURE / f"block_{bi}"
        bdir.mkdir(parents=True, exist_ok=True)
        inp = cap["inputs"][bi]
        from run_official_pipeline import write_adapter_input
        hashes = write_adapter_input(inp, bdir)
        per_block[str(bi)] = {"hashes": hashes, "calls": cap["calls"][bi]}

    (p16.VISUAL_ROOT / "00_input").mkdir(parents=True, exist_ok=True)
    save_input_frames(video, p16.VISUAL_ROOT / "00_input")
    save_frames(sample, p16.VISUAL_ROOT / "01_pytorch_reference")
    torch.save([cap["latents"][0]], p16.VISUAL_ROOT / "final_latent_reference.pt")
    torch.save(sample, p16.VISUAL_ROOT / "frames_reference.pt")

    manifest = {
        "phase": 16, "mode": "capture", "seed": args.seed, "steps": args.steps,
        "cfg_scale": args.cfg_scale, "git_head": git_head(),
        "elapsed_seconds": round(elapsed, 3),
        "frames": int(ori_length), "height": args.height, "width": args.width,
        "block_count": len(per_block), "per_block": per_block,
        "input_video_sha256": _sha(video),
    }
    (CAPTURE / "capture_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({"elapsed": round(elapsed, 3), "blocks": len(per_block),
                      "missing": missing}, indent=2))
    return 0


def _capture(cap, bi, args, kwargs):
    if args:
        vid, txt, vid_shape, txt_shape, emb = args[0], args[1], args[2], args[3], args[4]
    else:
        vid = kwargs["vid"]; txt = kwargs["txt"]
        vid_shape = kwargs["vid_shape"]; txt_shape = kwargs["txt_shape"]
        emb = kwargs["emb"]
    if cap["inputs"].get(bi) is None:
        cap["calls"][bi] += 1
        cap["inputs"][bi] = capture_inputs(vid, txt, vid_shape, txt_shape, emb)
    else:
        cap["calls"][bi] += 1


def _sha(video):
    import hashlib
    return hashlib.sha256(video.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
