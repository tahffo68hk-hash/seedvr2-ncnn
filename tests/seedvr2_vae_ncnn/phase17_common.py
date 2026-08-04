"""Phase 17 shared helpers: real SeedVR2 VAE construction, profiles, stats."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys

import torch

SEEDVR2_ROOT = r"D:\AIProjects\third_party\SeedVR2"
REPO_ROOT = r"D:\AIProjects\seedvr2-ncnn"
CKPT_DIR = os.path.join(REPO_ROOT, ".phase14_local_checkpoint")
VAE_CKPT = os.path.join(CKPT_DIR, "ema_vae.pth")
DIT_CKPT = os.path.join(CKPT_DIR, "seedvr2_ema_3b.pth")

VAE_CKPT_SHA256 = "c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac"
DIT_CKPT_SHA256 = "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b"

ART = os.path.join(REPO_ROOT, ".phase17_local_artifacts")
EXPORT_TMP = os.path.join(REPO_ROOT, ".phase17_export_tmp")
RUNTIME_TRACES = os.path.join(REPO_ROOT, ".phase17_runtime_traces")
VISUAL_OUT = os.path.join(REPO_ROOT, ".phase17_visual_outputs")
MEMORY_PROFILES = os.path.join(REPO_ROOT, ".phase17_memory_profiles")
DIAG = os.path.join(REPO_ROOT, ".phase17_diag")

# Profiles: A = Phase 16 visual baseline; B = dynamic asymmetric profile.
PROFILE_A = dict(name="A", batch=1, frames=4, height=128, width=128, seed=20260803)
PROFILE_B = dict(name="B", batch=1, frames=8, height=192, width=256, seed=20260803)
PROFILES = {"A": PROFILE_A, "B": PROFILE_B}

# Real VAE config from models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml
# + VideoAutoencoderKLWrapper kwargs (configs_3b/main.yaml vae.model).
VAE_CFG = dict(
    act_fn="silu",
    block_out_channels=[128, 256, 512, 512],
    down_block_types=["DownEncoderBlock3D"] * 4,
    in_channels=3,
    latent_channels=16,
    layers_per_block=2,
    norm_num_groups=32,
    out_channels=3,
    slicing_sample_min_size=4,
    temporal_scale_num=2,
    inflation_mode="pad",
    up_block_types=["UpDecoderBlock3D"] * 4,
    spatial_downsample_factor=8,
    temporal_downsample_factor=4,
    use_quant_conv=False,
    use_post_quant_conv=False,
    freeze_encoder=False,
)

# Official runner vae section (configs_3b/main.yaml).
VAE_RUNNER_CFG = dict(scaling_factor=0.9152, shifting_factor=0.0, grouping=False, use_sample=False)

DEFAULT_STAGE_THRESHOLDS = {
    "norm_activation": 1e-5,
    "conv": 5e-5,
    "residual_block": 1e-4,
    "downsample": 1e-4,
    "upsample_deconv": 2e-4,
    "encoder_final_latent": 2e-4,
    "decoder_pre_clamp": 3e-4,
    "decoder_final_float": 3e-4,
    "relative_max": 1e-3,
}


def install_seedvr2_paths() -> None:
    if SEEDVR2_ROOT not in sys.path:
        sys.path.insert(0, SEEDVR2_ROOT)


def build_real_vae():
    """Construct the REAL VideoAutoencoderKLWrapper with the official config."""
    install_seedvr2_paths()
    from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

    return VideoAutoencoderKLWrapper(**dict(VAE_CFG)).eval()


def load_vae_state_dict():
    """Load ema_vae.pth safely (weights_only, mmap, cpu). Returns (state_dict, info)."""
    t0 = torch.manual_seed(0) if False else None
    import time

    st = time.time()
    sd = torch.load(
        VAE_CKPT,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    return sd, {"load_seconds": round(time.time() - st, 3)}


def tensor_stats(t: torch.Tensor) -> dict:
    """Compact, JSON-safe statistics for a tensor."""
    n = t.numel()
    t = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": int(n),
        "min": None if n == 0 else float(t.min().item()),
        "max": None if n == 0 else float(t.max().item()),
        "mean": None if n == 0 else float(t.mean().item()),
        "std": None if n == 0 else float(t.std().item()),
        "nan_count": int(torch.isnan(t).sum().item()),
        "inf_count": int(torch.isinf(t).sum().item()),
    }


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float()
    b = b.detach().float()
    denom = a.abs().max().item()
    if denom == 0:
        return 0.0
    return float((a - b).abs().max().item() / denom)


def write_report(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"wrote {path}")


def ensure_clean_finite(payload: dict) -> bool:
    """Recursively check for NaN/Inf in a JSON payload."""
    bad = []

    def walk(o, key):
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                bad.append(key)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{key}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{key}[{i}]")

    walk(payload, "$")
    return len(bad) == 0, bad