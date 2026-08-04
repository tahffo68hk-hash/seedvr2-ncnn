#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: capture REAL SeedVR2 VAE reference stage I/O for Profile A/B.

Builds the real VideoAutoencoderKLWrapper, strict-loads ema_vae.pth (fp32 CPU),
runs the deterministic whole-tensor encode/decode path (use_sample=False, the
official i2v runner path), and records per-stage inputs/outputs (raw fp32 +
stats) for the ncnn stage-parity contract.

Also verifies:
  * official sliced path (split_size=4 + memory_limit 0.5GiB) == whole-tensor
    for T=9 / T=17 (bit-close, fp rounding only);
  * T=8 raw (non-4k+1) input crashes the official sliced path (documented
    official short-clip limitation; Profile B uses the official T=9 padded
    convention so slicing is never triggered as a correctness requirement).

Heavy blobs live under the git-ignored .phase17_local_artifacts/reference_*.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402

ENC_STAGES = ["conv_in", "down0", "down1", "down2", "down3", "mid", "out"]
DEC_STAGES = ["conv_in", "mid", "up0", "up1", "up2", "up3", "out"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_blob(path: Path, t: torch.Tensor) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = t.detach().float().cpu().contiguous()
    t.numpy().tofile(str(path))
    return {
        "path": str(path),
        "shape": list(t.shape),
        "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
    }


def build_and_load():
    vae = pc.build_real_vae()
    sd, info = pc.load_vae_state_dict()
    missing, unexpected = vae.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    vae.eval().to(torch.float32)
    vae.disable_slicing()
    vae.set_memory_limit(None, None)
    return vae, info


def make_input_video(frames: int, height: int, width: int, seed: int):
    """Official-style deterministic moving clip [T,H,W,3] in 0..1."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(height, width, 3, generator=g)
    t_phase = torch.linspace(0.0, 2.0 * 3.141592653589793, frames)
    rows = torch.linspace(0.0, 1.0, height).reshape(-1, 1, 1)  # [H,1,1]
    clips = []
    for i in range(frames):
        shade = 0.16 * torch.sin(t_phase[i] + 2.5 * rows)  # [H,1,1]
        clips.append((base + shade).clamp(0.0, 1.0))
    return torch.stack(clips)


def cut_videos(video, sp_size: int = 1):
    """Official padding: repeat last frame to T = 4k+1."""
    t = video.size(1)
    if t <= 4 * sp_size:
        padding = [video[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
        return torch.cat([video] + padding, dim=1)
    if (t - 1) % (4 * sp_size) == 0:
        return video
    padding = [video[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
    return torch.cat([video] + padding, dim=1)


def capture_profile(vae, prof: dict, root: Path) -> dict:
    frames, height, width = prof["frames"], prof["height"], prof["width"]
    video = make_input_video(frames, height, width, prof["seed"])  # [T,H,W,3] 0..1
    cond = video.permute(0, 3, 1, 2)  # [T,3,H,W]
    cond = cond.permute(1, 0, 2, 3)  # [3,T,H,W]
    cond = cut_videos(cond, 1)  # -> T=4k+1
    x = ((cond.float() - 0.5) / 0.5).unsqueeze(0)  # [1,3,T,H,W] in [-1,1]

    caps: dict = {}

    def make_hooks(name):
        def pre(mod, inp):
            caps.setdefault(name, {})["in"] = inp[0].detach().float().cpu().contiguous()

        def post(mod, inp, out):
            caps.setdefault(name, {})["out"] = out.detach().float().cpu().contiguous()

        return pre, post

    enc = vae.encoder
    dec = vae.decoder
    enc_hooks = []
    dec_hooks = []
    for nm, mod in [
        ("conv_in", enc.conv_in),
        ("down0", enc.down_blocks[0]),
        ("down1", enc.down_blocks[1]),
        ("down2", enc.down_blocks[2]),
        ("down3", enc.down_blocks[3]),
        ("mid", enc.mid_block),
    ]:
        pre, post = make_hooks(f"enc_{nm}")
        enc_hooks.append(mod.register_forward_pre_hook(pre))
        enc_hooks.append(mod.register_forward_hook(post))
    # enc_out: stage input = conv_norm_out input, stage output = encoder output
    pre, post = make_hooks("enc_out")
    enc_hooks.append(enc.conv_norm_out.register_forward_pre_hook(pre))
    enc_hooks.append(enc.register_forward_hook(post))

    for nm, mod in [
        ("conv_in", dec.conv_in),
        ("mid", dec.mid_block),
        ("up0", dec.up_blocks[0]),
        ("up1", dec.up_blocks[1]),
        ("up2", dec.up_blocks[2]),
        ("up3", dec.up_blocks[3]),
    ]:
        pre, post = make_hooks(f"dec_{nm}")
        dec_hooks.append(mod.register_forward_pre_hook(pre))
        dec_hooks.append(mod.register_forward_hook(post))
    pre, post = make_hooks("dec_out")
    dec_hooks.append(dec.conv_norm_out.register_forward_pre_hook(pre))
    dec_hooks.append(dec.register_forward_hook(post))

    with torch.no_grad():
        t0 = time.time()
        latent = vae.encode(x).posterior.mode()  # [1,16,TL,h,w] deterministic
        enc_time = time.time() - t0
        t0 = time.time()
        decoded = vae.decode(latent).sample  # [1,3,T,H,W]
        dec_time = time.time() - t0

    for h in enc_hooks + dec_hooks:
        h.remove()

    # Scaled latent exactly as the official runner produces for the DiT:
    # rearrange "b c ... -> b ... c" then (latent - shift) * scale
    scale = pc.VAE_RUNNER_CFG["scaling_factor"]
    shift = pc.VAE_RUNNER_CFG["shifting_factor"]
    latent_bt = latent.permute(0, 2, 3, 4, 1)  # [B,TL,h,w,C]
    latent_scaled = (latent_bt - shift) * scale

    manifest = {
        "profile": prof["name"],
        "video_frames": frames,
        "video_shape": [frames, height, width, 3],
        "vae_input_shape": list(x.shape),
        "latent_shape": list(latent.shape),
        "latent_scaled_shape": list(latent_scaled.shape),
        "decoded_shape": list(decoded.shape),
        "scaling_factor": scale,
        "shifting_factor": shift,
        "use_sample": False,
        "slicing": "whole-tensor (memory_state=DISABLED)",
        "enc_seconds": round(enc_time, 3),
        "dec_seconds": round(dec_time, 3),
        "stage_files": {},
    }

    for key in caps:
        kind, stage = key.split("_", 1)
        if kind == "enc":
            entry = manifest["stage_files"].setdefault(f"enc_{stage}", {})
        else:
            entry = manifest["stage_files"].setdefault(f"dec_{stage}", {})
        entry["in"] = write_blob(root / key / "input.bin", caps[key]["in"])
        entry["out"] = write_blob(root / key / "output.bin", caps[key]["out"])

    write_blob(root / "input_video.bin", video)
    write_blob(root / "input_norm.bin", x)
    write_blob(root / "latent_raw.bin", latent)
    write_blob(root / "latent_scaled.bin", latent_scaled)
    write_blob(root / "decoded_raw.bin", decoded)

    for p in (root / "input_video.bin", root / "latent_scaled.bin"):
        manifest[p.name] = sha256_file(p)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(pc.ART))
    ap.add_argument("--profiles", default="A,B")
    ap.add_argument("--report", type=Path,
                    default=Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_vae_io_contract.json"))
    args = ap.parse_args()

    vae, load_info = build_and_load()
    print(f"vae loaded params={sum(p.numel() for p in vae.parameters())} strict_ok", flush=True)

    contract = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "checkpoint_bytes": os.path.getsize(pc.VAE_CKPT),
        "dtype": "float32 (CPU harness)",
        "load_info": load_info,
        "encoder_stages": ENC_STAGES,
        "decoder_stages": DEC_STAGES,
        "profiles": {},
        "slicing_equivalence": {},
        "t8_sliced_crash": {},
    }

    for pname in [p.strip() for p in args.profiles.split(",")]:
        prof = pc.PROFILES[pname]
        root = args.root / f"reference_{pname}"
        manifest = capture_profile(vae, prof, root)
        contract["profiles"][pname] = manifest
        pc.write_report(root / "manifest.json", manifest)
        print(f"profile {pname} captured: latent={manifest['latent_shape']} "
              f"decoded={manifest['decoded_shape']} enc={manifest['enc_seconds']}s "
              f"dec={manifest['dec_seconds']}s", flush=True)

    # Slicing equivalence (requires dist init for the official sliced path)
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            store = str(args.root.parent / ".phase17_diag" / "dist_store_p17")
            os.makedirs(os.path.dirname(store), exist_ok=True)
            dist.init_process_group(backend="gloo", rank=0, world_size=1,
                                    init_method=f"file:///{store}")
        for T, hw in [(9, (192, 256)), (17, (128, 128))]:
            xs = (torch.rand(1, 3, T, *hw, dtype=torch.float32) - 0.5) * 2
            sliced = build_and_load()[0]
            sliced.set_causal_slicing(split_size=4, memory_device="same")
            sliced.set_memory_limit(0.5, 0.5)
            with torch.no_grad():
                z_s = sliced.encode(xs).posterior.mode()
                z_w = vae.encode(xs).posterior.mode()
            contract["slicing_equivalence"][f"T{T}"] = {
                "whole_shape": list(z_w.shape),
                "sliced_shape": list(z_s.shape),
                "max_abs_diff": float((z_w - z_s).abs().max().item()),
                "equal": torch.equal(z_w, z_s),
            }
            print(f"  slicing T={T} max_diff="
                  f"{contract['slicing_equivalence'][f'T{T}']['max_abs_diff']:.3e}", flush=True)
        # T=8 raw crash documentation
        try:
            v8 = build_and_load()[0]
            v8.set_causal_slicing(split_size=4, memory_device="same")
            v8.set_memory_limit(0.5, 0.5)
            x8 = (torch.rand(1, 3, 8, 128, 128, dtype=torch.float32) - 0.5) * 2
            with torch.no_grad():
                v8.encode(x8)
            contract["t8_sliced_crash"] = {"crashed": False}
        except Exception as e:  # noqa: BLE001
            contract["t8_sliced_crash"] = {
                "crashed": True,
                "error_type": type(e).__name__,
                "error": str(e)[:300],
                "note": "official split_size=4 slicing on raw T=8 input; "
                        "Profile B uses the official T=9 padded convention.",
            }
    except Exception as e:  # noqa: BLE001
        contract["slicing_equivalence_error"] = str(e)[:500]

    pc.write_report(str(args.report), contract)
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
