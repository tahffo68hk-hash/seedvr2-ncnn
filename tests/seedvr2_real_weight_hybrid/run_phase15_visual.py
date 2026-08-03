#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: official SeedVR2 3B pipeline with 0/1/2/4 target blocks swapped
to the ncnn multiblock streaming hybrid adapter.

Modes
-----
reference
    Run the official pipeline (all 32 PyTorch blocks, bf16 autocast) with
    block 15/16/17/18 input capture, then post-hoc FP32 reference evals for
    the four target blocks.  Saves input frames, reference frames, the final
    latent and per-block FP32 reference outputs under
    .phase15_visual_outputs/.

hybrid --blocks N (N in 1/2/4)
    Run the official pipeline with the target block range replaced by the
    multiblock_streaming_adapter (one strict streaming adapter call per
    block, no silent fallback).  Saves hybrid frames, final latent, adapter
    reports and per-block ncnn outputs.

metrics
    Load all saved artifacts, compute per-mode pixel/latent metrics, per-block
    FP32-vs-ncnn parity, input consistency, build the 6-panel side-by-side
    (Input | PyTorch | 1B | 2B | 4B | Amplified 4B diff) and write
    reports/multiblock_visual_comparison.json.

Fixed settings (Phase 15 spec): seed=20260803, steps=1, frames=4, 128x128,
cfg_scale=1.0.
"""
from __future__ import annotations

import argparse
import json
import subprocess
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

import phase15_common as p15  # noqa: E402
import streaming_block_manager as sbm  # noqa: E402
from probe_lib import ensure_compat  # noqa: E402
from run_official_pipeline import (  # noqa: E402
    capture_inputs, write_adapter_input, make_input_video, build_video_transform,
    cut_videos, get_device, tensor_metrics, image_metrics, load_png_frames,
    git_head, sha256_file, sha256_tensor,
)

C = p15.C
EMB = p15.EMB
REF_BLOCKS = p15.BLOCK_SET          # [15, 16, 17, 18]
HYBRID_RANGES = {1: p15.RANGE_1, 2: p15.RANGE_2, 4: p15.RANGE_4}
VISUAL = p15.VISUAL_ROOT
PER_BLOCK_REL_LIMIT = 1.0e-3        # per-block final output relative limit

BLOCK_PARITY_DIR = VISUAL / "09_block_parity"



def tensor_to_uint8(tchw):
    a = tchw.clamp(-1, 1) * 0.5 + 0.5
    a = (a * 255.0).round().clamp(0, 255).to(torch.uint8)
    return a.permute(0, 2, 3, 1).cpu().numpy()


def save_frames(tchw, out_dir: Path, prefix: str = "frame"):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    u8 = tensor_to_uint8(tchw)
    out = []
    for i, f in enumerate(u8):
        p = out_dir / ("%s_%04d.png" % (prefix, i))
        Image.fromarray(f, mode="RGB").save(p, compress_level=0)
        out.append(str(p))
    return out


def save_input_frames(video_thwc, out_dir: Path):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    u8 = (video_thwc.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
    out = []
    for i, f in enumerate(u8):
        p = out_dir / ("input_%04d.png" % i)
        Image.fromarray(f, mode="RGB").save(p, compress_level=0)
        out.append(str(p))
    return out

def build_runner(cfg_scale: float = 1.0, steps: int = 1, fp32_blocks=()):
    """Mirror the official configure_runner, minus distributed init.

    Target blocks stay FP32 (the ncnn hybrid contract is FP32); all other
    blocks run bf16 under the same bf16 autocast context.
    """
    from omegaconf import OmegaConf
    from common.config import load_config
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer
    from common.seed import set_seed

    os_env = __import__("os")
    os_env.chdir(str(p15.SEEDVR2_ROOT))
    os_env.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    os_env.environ["MASTER_ADDR"] = "127.0.0.1"
    os_env.environ["MASTER_PORT"] = "29501"
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1, init_method="env://")

    config = load_config(str(p15.SEEDVR2_ROOT / "configs_3b" / "main.yaml"))
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    runner.config.vae.checkpoint = str(p15.VAE_CKPT)
    runner.configure_dit_model(device="cpu", checkpoint=str(p15.MAIN_CKPT))
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

    fp32_set = set(fp32_blocks)
    for i, blk in enumerate(runner.dit.blocks):
        if i not in fp32_set:
            blk.to(torch.bfloat16)
    runner.dit.to("cuda")
    runner.config.diffusion.cfg.scale = float(cfg_scale)
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = int(steps)
    runner.configure_diffusion()
    return runner


def run_official_reference(args) -> dict:
    """Full PyTorch reference; capture target-block inputs + FP32 ref evals."""
    ensure_compat()
    from common.seed import set_seed
    cap = {"calls": {bi: 0 for bi in REF_BLOCKS}, "inputs": {},
           "latents": None, "input_hashes": {}}
    t0 = time.time()
    runner = build_runner(cfg_scale=args.cfg_scale, steps=args.steps, fp32_blocks=())
    orig_forwards = {bi: runner.dit.blocks[bi].forward for bi in REF_BLOCKS}

    def make_cap(bi, orig):
        def wrap(vid, txt, vid_shape, txt_shape, emb, cache):
            cap["calls"][bi] += 1
            if bi not in cap["inputs"]:
                inp = capture_inputs(vid, txt, vid_shape, txt_shape, emb)
                cap["inputs"][bi] = inp
                cap["input_hashes"][bi] = {
                    "vid": sha256_tensor(inp["vid"]),
                    "txt": sha256_tensor(inp["txt"]),
                    "emb": sha256_tensor(inp["emb"]),
                }
            return orig(vid, txt, vid_shape, txt_shape, emb, cache)
        return wrap

    for bi in REF_BLOCKS:
        runner.dit.blocks[bi].forward = make_cap(bi, orig_forwards[bi])

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
    pos_emb = torch.load(str(p15.POS_EMB), map_location="cpu", weights_only=True).to(get_device())
    neg_emb = torch.load(str(p15.NEG_EMB), map_location="cpu", weights_only=True).to(get_device())

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises, conditions=conditions, texts_pos=[pos_emb],
            texts_neg=[neg_emb], cfg_scale=args.cfg_scale, dit_offload=False,
        )
    sample = video_tensors[0].permute(1, 0, 2, 3)[:ori_length].float().cpu()
    elapsed = time.time() - t0

    missing = [bi for bi in REF_BLOCKS if bi not in cap["inputs"]]
    if missing:
        raise RuntimeError("target blocks never called: %s" % missing)



    (VISUAL / "00_input").mkdir(parents=True, exist_ok=True)
    save_input_frames(video, VISUAL / "00_input")
    save_frames(sample, VISUAL / "01_pytorch_reference")
    torch.save([cap["latents"][0]], VISUAL / "final_latent_reference.pt")
    torch.save(sample, VISUAL / "frames_reference.pt")

    BLOCK_PARITY_DIR.mkdir(parents=True, exist_ok=True)
    per_block = {}
    for bi in REF_BLOCKS:
        bdir = BLOCK_PARITY_DIR / ("block_%d_reference" % bi)
        bdir.mkdir(parents=True, exist_ok=True)
        inp = cap["inputs"][bi]
        write_adapter_input(inp, bdir)
        per_block[str(bi)] = {
            "input_hashes": cap["input_hashes"][bi],
            "calls": cap["calls"][bi],
        }

    manifest = {
        "mode": "reference", "seed": args.seed, "steps": args.steps,
        "cfg_scale": args.cfg_scale, "git_head": git_head(),
        "elapsed_seconds": round(elapsed, 3),
        "frames": int(ori_length), "height": args.height, "width": args.width,
        "block_calls": cap["calls"], "per_block": per_block,
    }
    (VISUAL / "run_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def make_hybrid_forward(orig, cap, workdir: Path, block_index: int):
    def forward(vid, txt, vid_shape, txt_shape, emb, cache):
        cap["calls"][block_index] += 1
        inp = capture_inputs(vid, txt, vid_shape, txt_shape, emb)
        cap["inputs"][block_index] = inp
        cap["input_hashes"][block_index] = {
            "vid": sha256_tensor(inp["vid"]),
            "txt": sha256_tensor(inp["txt"]),
            "emb": sha256_tensor(inp["emb"]),
        }
        input_dir = workdir / ("block_%d" % block_index) / "input"
        output_dir = workdir / ("block_%d" % block_index) / "output"
        report_path = workdir / ("block_%d" % block_index) / "adapter_report.json"
        write_adapter_input(inp, input_dir)
        report, stdout, rc = sbm.run_adapter(
            [block_index], input_dir, output_dir, load_mode="streaming",
            report_path=report_path, timeout_s=1800)
        if rc != 0:
            raise RuntimeError(
                "adapter rc=%d for block %d stderr=%s" % (rc, block_index, (stdout or "")[-2000:]))
        if "HYBRID BLOCK EXECUTED" not in stdout:
            raise RuntimeError("adapter did not print HYBRID BLOCK EXECUTED for block %d" % block_index)
        if report is None:
            raise RuntimeError("adapter report missing for block %d" % block_index)
        cap["adapter_reports"][block_index] = report
        cap["stdout_has_marker"][block_index] = True
        out_vid = np.fromfile(str(output_dir / ("block_%d" % block_index) / "output_vid.bin"), dtype=np.float32).reshape(-1, C)
        out_txt = np.fromfile(str(output_dir / ("block_%d" % block_index) / "output_txt.bin"), dtype=np.float32).reshape(-1, C)
        cap["output_hashes"][block_index] = {
            "vid": sha256_file(output_dir / ("block_%d" % block_index) / "output_vid.bin"),
            "txt": sha256_file(output_dir / ("block_%d" % block_index) / "output_txt.bin"),
        }
        vid_out = torch.from_numpy(out_vid).to(device=vid.device, dtype=vid.dtype)
        txt_out = torch.from_numpy(out_txt).to(device=txt.device, dtype=txt.dtype)
        return vid_out, txt_out, vid_shape, txt_shape
    return forward


def run_hybrid(args, blocks) -> dict:
    """Official pipeline with `blocks` replaced by strict streaming ncnn."""
    ensure_compat()
    from common.seed import set_seed
    cap = {"calls": {bi: 0 for bi in blocks}, "inputs": {}, "input_hashes": {},
           "adapter_reports": {}, "stdout_has_marker": {}, "output_hashes": {},
           "latents": None}
    n = len(blocks)
    workdir = VISUAL / ("run_hybrid_%d" % n)
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runner = build_runner(cfg_scale=args.cfg_scale, steps=args.steps, fp32_blocks=())
    for bi in blocks:
        orig = runner.dit.blocks[bi].forward
        runner.dit.blocks[bi].forward = make_hybrid_forward(orig, cap, workdir, bi)

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
    pos_emb = torch.load(str(p15.POS_EMB), map_location="cpu", weights_only=True).to(get_device())
    neg_emb = torch.load(str(p15.NEG_EMB), map_location="cpu", weights_only=True).to(get_device())

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises, conditions=conditions, texts_pos=[pos_emb],
            texts_neg=[neg_emb], cfg_scale=args.cfg_scale, dit_offload=False,
        )
    sample = video_tensors[0].permute(1, 0, 2, 3)[:ori_length].float().cpu()
    elapsed = time.time() - t0

    hybrid_dir = VISUAL / {1: "02_hybrid_1block", 2: "03_hybrid_2block", 4: "04_hybrid_4block"}[n]
    save_frames(sample, hybrid_dir)
    torch.save([cap["latents"][0]], VISUAL / ("final_latent_hybrid_%d.pt" % n))
    torch.save(sample, VISUAL / ("frames_hybrid_%d.pt" % n))

    adapter_evidence = {}
    for bi in blocks:
        rep = cap["adapter_reports"][bi]
        adapter_evidence[str(bi)] = {
            "ncnn_adapter_call_count": int(rep.get("ncnn_adapter_call_count", -1)),
            "original_block_call_count": int(rep.get("original_block_call_count", -1)),
            "silent_fallback": bool(rep.get("silent_fallback", True)),
            "order_ok": bool(rep.get("order_ok", False)),
            "streaming_mode_validated": bool(rep.get("streaming_mode_validated", False)),
            "max_resident_weighted_blocks": int(rep.get("max_resident_weighted_blocks", -1)),
            "events": len(rep.get("events", [])),
            "rc": 0,
            "stdout_has_hybrid_block_executed": bool(cap["stdout_has_marker"].get(bi, False)),
            "output_hashes": cap["output_hashes"][bi],
            "input_hashes": cap["input_hashes"][bi],
            "block_call_count": cap["calls"][bi],
        }

    manifest = {
        "mode": "hybrid", "blocks": blocks, "block_count": n,
        "seed": args.seed, "steps": args.steps, "cfg_scale": args.cfg_scale,
        "git_head": git_head(), "elapsed_seconds": round(elapsed, 3),
        "adapter": adapter_evidence,
    }
    (VISUAL / ("run_hybrid_%d_manifest.json" % n)).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def block_fp32_eval(orig_forward, inp, cache=None):
    """True-FP32 reference eval of one real block on a captured input.

    A shared official Cache must be reused across evals of the same block
    range: the rope module lru-caches the ~7.9 GiB MM-RoPE3D frequency grid
    per module, and a disabled Cache would force every block to allocate its
    own grid (4 x 7.9 GiB does not fit on the 16 GB GPU).
    """
    from common.cache import Cache
    if cache is None:
        cache = Cache(disable=True)
    dev = get_device()
    vid = inp["vid"].to(dev)
    txt = inp["txt"].to(dev)
    emb = inp["emb"].to(dev)
    vs = torch.tensor(inp["vid_shape"], device=dev, dtype=torch.long).reshape(1, -1)
    ts = torch.tensor(inp["txt_shape"], device=dev, dtype=torch.long).reshape(1, -1)
    torch.Tensor.bfloat16 = lambda self: self  # noqa: E731
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        v_out, t_out, _, _ = orig_forward(vid, txt, vs, ts, emb, cache)
    return v_out.detach().float().cpu().contiguous(), t_out.detach().float().cpu().contiguous()


def rel_error(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    d = np.abs(a - b)
    return {
        "max_abs_error": float(d.max()),
        "mean_abs_error": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "relative_max_error": float(d.max() / max(np.abs(a).max(), 1e-30)),
        "nan_count": int(np.isnan(b).sum()),
        "inf_count": int(np.isinf(b).sum()),
    }


def make_six_panel_side_by_side(input_u8, ref_u8, h1, h2, h4, amp4, side_dir: Path):
    """Panels: Input | PyTorch | 1B | 2B | 4B | Amplified 4B diff."""
    from PIL import Image
    side_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(ref_u8.shape[0]):
        panels = [input_u8[i], ref_u8[i], h1[i], h2[i], h4[i], amp4[i]]
        widths = [p.shape[1] for p in panels]
        height = max(p.shape[0] for p in panels)
        gap = 4
        canvas = np.full((height, sum(widths) + gap * (len(panels) - 1), 3), 255, dtype=np.uint8)
        x = 0
        for p in panels:
            p = np.repeat(p, 3, axis=2) if p.shape[2] == 1 else p
            canvas[: p.shape[0], x : x + p.shape[1]] = p
            x += p.shape[1] + gap
        p = side_dir / ("side_by_side_%04d.png" % i)
        Image.fromarray(canvas, mode="RGB").save(p, compress_level=0)
        paths.append(str(p))
    return paths


def compute_metrics(args) -> dict:
    ref_man = json.loads((VISUAL / "run_reference_manifest.json").read_text("utf-8"))
    ref_frames = load_png_frames(VISUAL / "01_pytorch_reference")
    input_u8 = load_png_frames(VISUAL / "00_input")
    ref_lat = torch.load(VISUAL / "final_latent_reference.pt", map_location="cpu", weights_only=True)
    hybrid_dirs = {1: "02_hybrid_1block", 2: "03_hybrid_2block", 4: "04_hybrid_4block"}

    modes = {}
    for n in (1, 2, 4):
        man = json.loads((VISUAL / ("run_hybrid_%d_manifest.json" % n)).read_text("utf-8"))
        hyb_frames = load_png_frames(VISUAL / hybrid_dirs[n])
        assert hyb_frames.shape == ref_frames.shape, (hyb_frames.shape, ref_frames.shape)
        hyb_lat = torch.load(VISUAL / ("final_latent_hybrid_%d.pt" % n), map_location="cpu", weights_only=True)
        pixel = image_metrics(ref_frames, hyb_frames)
        pixel["output_shape"] = list(ref_frames.shape)
        pixel["frames"] = int(ref_frames.shape[0])
        pixel["non_constant_reference"] = bool(ref_frames.std() > 1e-3)
        pixel["non_constant_hybrid"] = bool(hyb_frames.std() > 1e-3)
        latent = tensor_metrics(ref_lat[0], hyb_lat[0], "final_latent")
        adapter_ok = True
        for bi in man["blocks"]:
            ev = man["adapter"][str(bi)]
            if not (ev["ncnn_adapter_call_count"] >= 1 and ev["original_block_call_count"] == 0
                    and ev["silent_fallback"] is False and ev["order_ok"] is True
                    and ev["streaming_mode_validated"] is True
                    and ev["max_resident_weighted_blocks"] == 1
                    and ev["stdout_has_hybrid_block_executed"]):
                adapter_ok = False
        modes[str(n)] = {
            "blocks": man["blocks"], "pixels": pixel, "latent": latent,
            "adapter_ok": adapter_ok, "adapter": man["adapter"],
            "elapsed_seconds": man["elapsed_seconds"],
        }

    # Input consistency reference vs each hybrid run.  The first block of each
    # chain has an all-PyTorch upstream, so its input must match the reference
    # exactly; later blocks in a chain receive the upstream ncnn output and are
    # expected to differ (their parity is covered by the fixed chain parity).
    input_consistency = {}
    for n in (1, 2, 4):
        man = json.loads((VISUAL / ("run_hybrid_%d_manifest.json" % n)).read_text("utf-8"))
        per_block_consistency = {}
        for bi in man["blocks"]:
            ref_h = ref_man["per_block"][str(bi)]["input_hashes"]
            hyb_h = man["adapter"][str(bi)]["input_hashes"]
            same = (ref_h["vid"] == hyb_h["vid"] and ref_h["txt"] == hyb_h["txt"]
                    and ref_h["emb"] == hyb_h["emb"])
            rv = np.fromfile(str(BLOCK_PARITY_DIR / ("block_%d_reference" % bi) / "input_vid.bin"), np.float32)
            hv = np.fromfile(str(VISUAL / ("run_hybrid_%d" % n) / ("block_%d" % bi) / "input" / "input_vid.bin"), np.float32)
            max_in_diff = float(np.abs(rv - hv).max()) if rv.shape == hv.shape else -1.0
            per_block_consistency[str(bi)] = {"same_input": same, "max_input_abs_diff": max_in_diff}
        input_consistency[str(n)] = {
            "first_block_consistent": bool(per_block_consistency.get(str(man["blocks"][0]), {}).get("same_input", False)),
            "per_block": per_block_consistency,
        }

    # Per-block parity: FP32 ref vs adapter output (relative limit 1e-3).
    # Valid only where the hybrid-run input matches the reference input; the
    # chained (input-differing) cases are evidenced by the fixed chain parity
    # report's per_block_error on identical chained inputs.
    chain_parity = json.loads((p15.REPO / "reports" / "multiblock_chain_parity.json").read_text("utf-8"))
    chain_blocks = {"1": [16], "2": [16, 17], "4": [15, 16, 17, 18]}
    per_block = {}
    for bi in REF_BLOCKS:
        bdir = BLOCK_PARITY_DIR / ("block_%d_reference" % bi)
        ref_vid = np.fromfile(str(bdir / "fp32_ref_output_vid.bin"), np.float32)
        ref_txt = np.fromfile(str(bdir / "fp32_ref_output_txt.bin"), np.float32)
        entry = {"input_hashes": ref_man["per_block"][str(bi)]["input_hashes"], "hybrid_runs": {}}
        for n in (1, 2, 4):
            if bi not in HYBRID_RANGES[n]:
                continue
            same_input = input_consistency[str(n)]["per_block"][str(bi)]["same_input"]
            if not same_input:
                entry["hybrid_runs"][str(n)] = {
                    "chained_upstream_hybrid_input": True,
                    "parity_note": "input differs by design (upstream ncnn); see chain parity per-block error",
                }
                continue
            workdir = VISUAL / ("run_hybrid_%d" % n) / ("block_%d" % bi)
            out_vid = np.fromfile(str(workdir / "output" / ("block_%d" % bi) / "output_vid.bin"), np.float32)
            out_txt = np.fromfile(str(workdir / "output" / ("block_%d" % bi) / "output_txt.bin"), np.float32)
            assert out_vid.shape == ref_vid.shape and out_txt.shape == ref_txt.shape
            ev = rel_error(np.concatenate([ref_vid, ref_txt]), np.concatenate([out_vid, out_txt]))
            entry["hybrid_runs"][str(n)] = {
                "final_vid_max_abs_error": float(np.abs(ref_vid - out_vid).max()),
                "final_txt_max_abs_error": float(np.abs(ref_txt - out_txt).max()),
                **ev, "rel_limit": PER_BLOCK_REL_LIMIT,
                "pass": bool(ev["relative_max_error"] < PER_BLOCK_REL_LIMIT
                             and ev["nan_count"] == 0 and ev["inf_count"] == 0),
            }
        per_block[str(bi)] = entry

    # Chain-parity per-block errors (identical chained inputs, FP32 torch vs ncnn).
    chain_per_block = {}
    for cn, cblocks in chain_blocks.items():
        chain_name = {"1": "chain_1", "2": "chain_2", "4": "chain_4"}[cn]
        per_rep = chain_parity["chains"][chain_name].get("per_block_error", {})
        rep0 = per_rep.get("0", []) if per_rep else []
        chain_per_block[str(cn)] = [
            {"block_index": int(e["block_index"]), "relative_max_error": e["relative_max_error"],
             "max_abs_error": e["max_abs_error"], "nan_count": e["nan_count"],
             "inf_count": e["inf_count"],
             "pass": bool(e["relative_max_error"] < PER_BLOCK_REL_LIMIT
                          and e["nan_count"] == 0 and e["inf_count"] == 0)}
            for e in rep0
        ]

    # Side-by-side and amplified diffs (all panels converted to uint8).
    h1_f = load_png_frames(VISUAL / hybrid_dirs[1])
    h2_f = load_png_frames(VISUAL / hybrid_dirs[2])
    h4_f = load_png_frames(VISUAL / hybrid_dirs[4])
    input_u8 = (np.clip(input_u8, 0, 1) * 255.0).round().astype(np.uint8)
    h1_u8 = (np.clip(h1_f, 0, 1) * 255.0).round().astype(np.uint8)
    h2_u8 = (np.clip(h2_f, 0, 1) * 255.0).round().astype(np.uint8)
    h4_u8 = (np.clip(h4_f, 0, 1) * 255.0).round().astype(np.uint8)
    ref_u8 = (np.clip(ref_frames, 0, 1) * 255.0).round().astype(np.uint8)
    factor = 25.0
    d4 = np.abs(ref_frames - h4_f)
    amp4 = (np.clip(d4 * factor, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    for n, d in ((1, "05_abs_diff_1block"), (2, "06_abs_diff_2block"), (4, "07_abs_diff_4block")):
        hyb = {1: h1_f, 2: h2_f, 4: h4_f}[n]
        diff = np.abs(ref_frames - hyb)
        amp = (np.clip(diff * factor, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        out_dir = VISUAL / d
        out_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for i in range(amp.shape[0]):
            Image.fromarray(amp[i], mode="RGB").save(out_dir / ("diff_amp_%04d.png" % i), compress_level=0)
    side_paths = make_six_panel_side_by_side(
        input_u8, ref_u8, h1_u8, h2_u8, h4_u8, amp4, VISUAL / "08_side_by_side")

    p4 = modes["4"]["pixels"]
    verdict = "PASS"
    reasons = []
    if not all(modes[str(n)]["adapter_ok"] for n in (1, 2, 4)):
        verdict = "PARTIAL"
        reasons.append("adapter evidence missing for one or more hybrid runs")
    if not all(input_consistency[str(n)]["first_block_consistent"] for n in (1, 2, 4)):
        verdict = "PARTIAL"
        reasons.append("first-block official-context input differs from reference")
    if not all(per_block[str(bi)]["hybrid_runs"][str(n)]["pass"]
               for bi in REF_BLOCKS for n in (1, 2, 4)
               if str(n) in per_block[str(bi)]["hybrid_runs"]
               and "pass" in per_block[str(bi)]["hybrid_runs"][str(n)]):
        verdict = "PARTIAL"
        reasons.append("per-block FP32-vs-ncnn parity exceeds 1e-3")
    if not all(e["pass"] for cn in chain_per_block for e in chain_per_block[cn]):
        verdict = "PARTIAL"
        reasons.append("chained per-block FP32-vs-ncnn parity exceeds 1e-3")
    if p4["psnr_db"] < p15.VISUAL_PSNR_MIN_4:
        verdict = "PARTIAL"
        reasons.append("4-block PSNR below 38 dB")
    if p4["ssim"] < p15.VISUAL_SSIM_MIN_4:
        verdict = "PARTIAL"
        reasons.append("4-block SSIM below 0.98")
    for n in (1, 2, 4):
        m = modes[str(n)]
        if m["pixels"]["nan_count"] or m["pixels"]["inf_count"] or m["latent"]["nan_count"] or m["latent"]["inf_count"]:
            verdict = "PARTIAL"
            reasons.append("NaN/Inf in %d-block outputs" % n)
        if not (m["pixels"]["non_constant_reference"] and m["pixels"]["non_constant_hybrid"]):
            verdict = "PARTIAL"
            reasons.append("constant output frames (%d-block)" % n)
    if not reasons:
        reasons.append("per-block parity, adapter evidence, finite non-constant outputs, 4-block PSNR/SSIM all pass")

    report = {
        "phase": 15,
        "milestone": "real-weight multi-block streaming hybrid visual (official SeedVR2 pipeline, 1/2/4 target blocks swapped to ncnn)",
        "verdict": verdict,
        "overall_pass": verdict == "PASS",
        "verdict_reasons": reasons,
        "checkpoint": {"sha256": p15.EXPECTED_MAIN_SHA256, "official_source_only": True},
        "run": {
            "seed": ref_man["seed"], "steps": ref_man["steps"],
            "cfg_scale": ref_man["cfg_scale"], "git_head": git_head(),
            "reference_elapsed_seconds": ref_man["elapsed_seconds"],
            "hybrid_elapsed_seconds": {n: modes[str(n)]["elapsed_seconds"] for n in (1, 2, 4)},
        },
        "ranges": {"range_1": HYBRID_RANGES[1], "range_2": HYBRID_RANGES[2], "range_4": HYBRID_RANGES[4]},
        "per_block_parity": per_block,
        "input_consistency": input_consistency,
        "chain_parity_per_block": chain_per_block,
        "modes": modes,
        "thresholds": {
            "per_block_final_rel_limit": PER_BLOCK_REL_LIMIT,
            "visual_psnr_min_4": p15.VISUAL_PSNR_MIN_4,
            "visual_ssim_min_4": p15.VISUAL_SSIM_MIN_4,
            "pixel_max_abs_sanity": 0.1,
            "latent_max_abs_sanity": 0.5,
            "latent_mean_abs_sanity": 0.05,
            "amplify_factor": factor,
        },
        "outputs": {
            "side_by_side_dir": "06b_placeholder",
            "side_by_side_frames": side_paths,
        },
    }
    report["outputs"]["side_by_side_dir"] = str((VISUAL / "08_side_by_side").relative_to(p15.REPO))
    (p15.REPO / "reports" / "multiblock_visual_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report



def run_fp32_eval(args) -> dict:
    """Separate-process true-FP32 reference evals for the four target blocks.

    Mirrors the chain-parity worker memory strategy: only the target blocks
    stay FP32 on GPU; every other block is moved to CPU so the transient
    MM-RoPE3D frequency grid (~7.9 GiB FP32) fits on the 16 GB GPU.
    """
    ensure_compat()
    dev = get_device()
    runner = build_runner(cfg_scale=args.cfg_scale, steps=args.steps, fp32_blocks=REF_BLOCKS)
    for i, blk in enumerate(runner.dit.blocks):
        if i not in REF_BLOCKS:
            blk.to("cpu")
    runner.dit.vid_in.to("cpu"); runner.dit.txt_in.to("cpu"); runner.dit.emb_in.to("cpu")
    if getattr(runner.dit, "vid_out_norm", None) is not None:
        runner.dit.vid_out_norm.to("cpu")
    if getattr(runner.dit, "vid_out", None) is not None:
        runner.dit.vid_out.to("cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from common.cache import Cache
    shared_cache = Cache()
    results = {}
    for bi in REF_BLOCKS:
        bdir = BLOCK_PARITY_DIR / ("block_%d_reference" % bi)
        inp = {
            "vid": torch.from_numpy(np.fromfile(str(bdir / "input_vid.bin"), np.float32)).reshape(-1, C),
            "txt": torch.from_numpy(np.fromfile(str(bdir / "input_txt.bin"), np.float32)).reshape(-1, C),
            "emb": torch.from_numpy(np.fromfile(str(bdir / "emb.bin"), np.float32)).reshape(-1, EMB),
            "vid_shape": json.loads((bdir / "vid_shapes.json").read_text("utf-8")),
            "txt_shape": json.loads((bdir / "txt_lengths.json").read_text("utf-8")),
        }
        fp32_vid, fp32_txt = block_fp32_eval(runner.dit.blocks[bi].forward, inp, shared_cache)
        fp32_vid.numpy().tofile(bdir / "fp32_ref_output_vid.bin")
        fp32_txt.numpy().tofile(bdir / "fp32_ref_output_txt.bin")
        results[str(bi)] = {
            "fp32_ref_vid_sha256": sha256_tensor(fp32_vid),
            "fp32_ref_txt_sha256": sha256_tensor(fp32_txt),
            "vid_shape": inp["vid_shape"], "txt_shape": inp["txt_shape"],
        }
        print("fp32-eval block %d done" % bi, flush=True)
    manifest = {"mode": "fp32-eval", "git_head": git_head(), "per_block": results}
    (VISUAL / "fp32_eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "hybrid", "fp32-eval", "metrics"], required=True)
    ap.add_argument("--blocks", type=int, default=None, help="hybrid block count (1/2/4)")
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
        if args.mode == "fp32-eval":
            run_fp32_eval(args)
            return 0
        if args.mode == "hybrid":
            if args.blocks not in HYBRID_RANGES:
                raise SystemExit("--blocks must be 1, 2 or 4 for hybrid mode")
            run_hybrid(args, HYBRID_RANGES[args.blocks])
            return 0
        compute_metrics(args)
        return 0
    except Exception as exc:  # noqa: BLE001
        err = {"phase": "phase15_visual", "error": str(exc), "traceback": traceback.format_exc()}
        VISUAL.mkdir(parents=True, exist_ok=True)
        (VISUAL / "pipeline_error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
        print(traceback.format_exc())
        return 3


if __name__ == "__main__":
    sys.exit(main())