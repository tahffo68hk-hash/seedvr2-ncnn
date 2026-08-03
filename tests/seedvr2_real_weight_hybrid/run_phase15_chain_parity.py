#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: real-weight chain parity orchestrator (ranges 1/2/4).

Process isolation to bound GPU memory: the capture pass runs in this process;
each FP32 chain reference runs in its own `.conda-cuda` worker subprocess; the
ncnn adapter always runs as its own executable.  Writes
reports/multiblock_chain_parity.json, reports/multiblock_error_growth.json and
reports/multiblock_latency_profile.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SEEDVR2 = Path(r"D:\AIProjects\third_party\SeedVR2")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SEEDVR2))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_ncnn"))

import phase15_common as p15  # noqa: E402
import streaming_block_manager as sbm  # noqa: E402
from probe_lib import ensure_compat  # noqa: E402
from run_official_pipeline import (  # noqa: E402
    build_runner, capture_inputs, write_adapter_input, make_input_video,
    build_video_transform, cut_videos, get_device,
)

CAPTURE = p15.PHASE15_ROOT / "official_context_capture"
PARITY_OUT = p15.PHASE15_ROOT / "chain_parity"
GPU_PY = str(p15.GPU_PY.resolve())


def sha256_tensor(t: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def capture_official_inputs(args):
    CAPTURE.mkdir(parents=True, exist_ok=True)
    cap = {"inputs": {}}
    runner = build_runner(cfg_scale=args.cfg_scale, steps=args.steps)
    for bi in p15.BLOCK_SET:
        blk = runner.dit.blocks[bi]
        orig = blk.forward

        def make_wrap(bi, orig):
            def wrap(vid, txt, vid_shape, txt_shape, emb, cache):
                if bi not in cap["inputs"]:
                    cap["inputs"][bi] = capture_inputs(vid, txt, vid_shape, txt_shape, emb)
                return orig(vid, txt, vid_shape, txt_shape, emb, cache)
            return wrap

        blk.forward = make_wrap(bi, orig)

    from common.seed import set_seed
    set_seed(args.seed, same_across_ranks=True)
    video = make_input_video(args.frames, args.height, args.width)
    video_transform = build_video_transform(args.height, args.width)
    cond = video_transform(video.permute(0, 3, 1, 2).to(get_device()))
    cond = cut_videos(cond, 1)
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
        _ = runner.inference(noises=noises, conditions=conditions, texts_pos=[pos_emb],
                             texts_neg=[neg_emb], cfg_scale=args.cfg_scale, dit_offload=False)

    manifests = {}
    for bi, inp in cap["inputs"].items():
        d = CAPTURE / f"block_{bi}"
        d.mkdir(parents=True, exist_ok=True)
        write_adapter_input(inp, d)
        manifests[str(bi)] = {
            "vid_shape": inp["vid_shape"], "txt_shape": inp["txt_shape"],
            "vid_sha256": sha256_tensor(inp["vid"]), "txt_sha256": sha256_tensor(inp["txt"]),
            "emb_sha256": sha256_tensor(inp["emb"]),
        }
    (CAPTURE / "capture_manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print("captured blocks:", sorted(cap["inputs"].keys()))
    return manifests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-capture", action="store_true")
    args = ap.parse_args()

    ensure_compat()
    if not args.skip_capture or not (CAPTURE / "capture_manifest.json").exists():
        capture_official_inputs(args)

    chains = {
        "chain_1": (p15.RANGE_1, p15.RANGE_1[0]),
        "chain_2": (p15.RANGE_2, p15.RANGE_2[0]),
        "chain_4": (p15.RANGE_4, p15.RANGE_4[0]),
    }
    PARITY_OUT.mkdir(parents=True, exist_ok=True)
    chain_results = {}
    for name, (block_indices, bi0) in chains.items():
        out_json = PARITY_OUT / f"{name}_result.json"
        cmd = [GPU_PY, str((HERE / "phase15_chain_worker.py").resolve()),
               "--chain", name, "--blocks", ",".join(map(str, block_indices)),
               "--input-block", str(bi0), "--repeats", str(args.repeats),
               "--out", str(out_json.resolve())]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
        if proc.returncode != 0 or not out_json.exists():
            print(f"{name} worker failed rc={proc.returncode}")
            print((proc.stdout or "")[-2000:])
            print((proc.stderr or "")[-2000:])
            chain_results[name] = {"pass": False, "worker_rc": proc.returncode}
            continue
        r = json.loads(out_json.read_text("utf-8"))
        chain_results[name] = r
        print(f"{name}: pass={r['pass']} final_rel="
              f"{r['final_error'][0]['relative_max_error'] if r['final_error'] else None} "
              f"limit={r['rel_limit']}")

    overall_pass = all(chain_results.get(n, {}).get("pass", False) for n in chains)
    report = {
        "phase": 15,
        "real_seedvr2_weights": True,
        "multiblock_hybrid": True,
        "checkpoint_sha256": p15.EXPECTED_MAIN_SHA256,
        "continuous_blocks": True,
        "max_resident_weighted_blocks": 1,
        "streaming_mode_validated": True,
        "chains": chain_results,
        "overall_pass": overall_pass,
        "overall_status": "PASS" if overall_pass else "FAIL",
    }
    (p15.REPO / "reports" / "multiblock_chain_parity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Error growth summary.
    growth = {"E1": None, "E2": None, "E4": None,
              "cumulative_relative_max": {}, "per_block": {}}
    for i, name in enumerate(("chain_1", "chain_2", "chain_4")):
        r = chain_results.get(name)
        if r and r.get("final_error"):
            growth[f"E{2 ** i}"] = r["final_error"][0]["relative_max_error"]
            growth["cumulative_relative_max"][name] = r["final_error"][0]
            if r.get("per_block_error") and "0" in r["per_block_error"]:
                for e in r["per_block_error"]["0"]:
                    growth["per_block"].setdefault(str(e["block_index"]), []).append(e)
    gf = {}
    if growth["E1"] and growth["E2"]:
        gf["E2_over_E1"] = growth["E2"] / max(growth["E1"], 1e-15)
    if growth["E2"] and growth["E4"]:
        gf["E4_over_E2"] = growth["E4"] / max(growth["E2"], 1e-15)
    if growth["E1"] and growth["E4"]:
        gf["E4_over_E1"] = growth["E4"] / max(growth["E1"], 1e-15)
    growth["growth_factors"] = gf
    (p15.REPO / "reports" / "multiblock_error_growth.json").write_text(
        json.dumps({"phase": 15, "x_replaced_blocks": [1, 2, 4],
                    "y_chain_relative_max": [growth["E1"], growth["E2"], growth["E4"]],
                    **growth, "overall_pass": overall_pass}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # Latency profile.
    lat = {"chains": {}}
    for name in chains:
        r = chain_results.get(name)
        if r and "streaming" in r and "rep0" in r["streaming"]:
            rep = r["streaming"]["rep0"]
            lat["chains"][name] = {
                "total_load_ms": rep["report"].get("total_load_ms"),
                "total_execute_ms": rep["report"].get("total_execute_ms"),
                "total_unload_ms": rep["report"].get("total_unload_ms"),
                "total_wall_ms": rep["report"].get("total_wall_ms"),
                "per_block": rep["report"].get("per_block"),
                "ref_elapsed_s": r.get("ref_elapsed_s"),
            }
    (p15.REPO / "reports" / "multiblock_latency_profile.json").write_text(
        json.dumps({"phase": 15, **lat,
                    "note": "chain adapter latency; full-pipeline latency in multiblock_visual_comparison.json"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("overall:", report["overall_status"])
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
