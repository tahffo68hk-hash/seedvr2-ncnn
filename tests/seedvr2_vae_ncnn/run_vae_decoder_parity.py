# -*- coding: utf-8 -*-
"""Phase 17: VAE decoder end-to-end parity (official PyTorch vs ncnn FP32).

Generates openable PNG frames for both profiles, plus float/PNG metrics.
READY gates: decoded float relative max < 2e-3, decoded PNG PSNR >= 45 dB,
SSIM >= 0.99 (Profile A mandatory PNGs; Profile B at least one frame set).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402
from ncnn_vae_runtime import VaeDecoderRuntime  # noqa: E402

REL_LIMIT = 2e-3
PNG_PSNR_MIN = 45.0
PNG_SSIM_MIN = 0.99


def sha256_bytes(b: np.ndarray) -> str:
    return hashlib.sha256(b.astype("<f4").tobytes()).hexdigest()


def ssim_np(a: np.ndarray, b: np.ndarray, win: int = 11, data_range: float = 2.0) -> float:
    """Windowed SSIM (pure numpy), maps a,b in [-1,1] with data_range=2.0.

    Accepts 2D images or batched layouts (e.g. [B,C,T,H,W], [T,H,W,C]); the
    window always slides over the final two spatial dims.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    # Normalize [T,H,W,C] (HWC) to [T,C,H,W] so the window always slides
    # over the two spatial dims (H, W).
    if a.ndim == 4 and a.shape[-1] in (1, 3, 4):
        a = a.transpose(0, 3, 1, 2)
        b = b.transpose(0, 3, 1, 2)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    g = np.exp(-(np.arange(win) - (win - 1) / 2.0) ** 2 / (2 * 1.5 ** 2))
    g /= g.sum()
    win2d = np.outer(g, g)
    from numpy.lib.stride_tricks import sliding_window_view

    def local_stats(x):
        sw = sliding_window_view(x, (win, win), axis=(-2, -1))
        w = sw.reshape(*sw.shape[:-2], -1)
        m = w @ win2d.ravel()
        mu = np.full_like(x, np.nan)
        mu[..., win // 2:win // 2 + m.shape[-2], win // 2:win // 2 + m.shape[-1]] = m
        return mu

    mu_a = local_stats(a)
    mu_b = local_stats(b)
    mask = ~np.isnan(mu_a)
    mu_a = mu_a[mask]
    mu_b = mu_b[mask]
    a2 = a[mask]
    b2 = b[mask]
    va = local_stats(a * a)[mask] - mu_a * mu_a
    vb = local_stats(b * b)[mask] - mu_b * mu_b
    vab = local_stats(a * b)[mask] - mu_a * mu_b
    va = np.clip(va, 0, None)
    vb = np.clip(vb, 0, None)
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / \
               ((mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2))
    return float(np.nanmean(ssim_map))


def float_metrics(a: np.ndarray, b: np.ndarray, data_range: float = 2.0) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = np.abs(a - b)
    mse = float((d * d).mean()) if d.size else 0.0
    max_abs = float(d.max()) if d.size else 0.0
    denom = float(np.abs(a).max()) if a.size else 0.0
    rel = float(max_abs / denom) if denom > 0 else 0.0
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": float(d.mean()) if d.size else 0.0,
        "rmse": float(np.sqrt(mse)) if d.size else 0.0,
        "relative_max": rel,
        "psnr_db": float(10.0 * np.log10(data_range * data_range / mse)) if mse > 0 else None,
        "ref_max": denom,
        "nan": int(np.isnan(b).sum()),
        "inf": int(np.isinf(b).sum()),
    }


def save_pngs(frames: np.ndarray, out_dir: Path) -> list:
    """frames [T,H,W,C] (or [B,C,T,H,W]) in 0..1 -> 8-bit PNGs."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    if frames.ndim == 5:
        frames = frames[0].transpose(1, 2, 3, 0)
    elif frames.ndim == 4 and frames.shape[-1] not in (1, 3, 4):
        frames = frames.transpose(0, 2, 3, 1)
    paths = []
    u8 = (np.clip(frames, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    for i in range(frames.shape[0]):
        p = out_dir / f"frame_{i:03d}.png"
        Image.fromarray(u8[i], mode="RGB").save(p)
        paths.append(str(p.relative_to(Path(r"D:\AIProjects\seedvr2-ncnn"))))
    return paths


def run_profile(profile: str, root: Path, vis: Path) -> dict:
    ref_dir = root / f"reference_{profile}"
    manifest = json.loads((ref_dir / "manifest.json").read_text(encoding="utf-8"))
    latent = np.fromfile(ref_dir / "latent_raw.bin", dtype="<f4").reshape(manifest["latent_shape"])
    ref = np.fromfile(ref_dir / "decoded_raw.bin", dtype="<f4").reshape(manifest["decoded_shape"])

    rt = VaeDecoderRuntime(profile, root=str(root))
    t0 = time.perf_counter()
    got = rt.decode(latent).numpy().reshape(manifest["decoded_shape"])
    dec_s = round(time.perf_counter() - t0, 3)
    calls = rt.call_count
    rt.release()

    m = float_metrics(ref, got)
    m["decoded_shape_official"] = list(ref.shape)
    m["decoded_shape_ncnn"] = list(got.shape)
    m["shape_match"] = list(ref.shape) == list(got.shape)
    m["decoded_sha256_official"] = sha256_bytes(ref)
    m["decoded_sha256_ncnn"] = sha256_bytes(got)
    m["threshold_rel_max"] = REL_LIMIT

    # pixel-space metrics (float [-1,1] -> [0,1])
    ref_u = (np.clip(ref, -1, 1) + 1.0) / 2.0
    got_u = (np.clip(got, -1, 1) + 1.0) / 2.0
    pm = float_metrics(ref_u, got_u, data_range=1.0)
    pm["ssim"] = ssim_np(ref_u, got_u, data_range=1.0)
    pm["pixel_psnr_min_db"] = PNG_PSNR_MIN
    pm["pixel_ssim_min"] = PNG_SSIM_MIN
    pm["pass"] = bool(m["shape_match"] and m["relative_max"] < REL_LIMIT and m["nan"] == 0
                      and m["inf"] == 0 and pm["psnr_db"] is not None
                      and pm["psnr_db"] >= PNG_PSNR_MIN and pm["ssim"] >= PNG_SSIM_MIN)
    m["pixels"] = pm
    m["pass"] = pm["pass"]
    m["decoder_chain_seconds"] = dec_s
    m["decoder_call_count"] = calls
    m["latent_input_shape"] = manifest["latent_shape"]

    # openable PNGs
    base = vis / f"vae_decoder_parity_{profile}"
    m["png_official"] = save_pngs(ref_u, base / "official")
    m["png_ncnn"] = save_pngs(got_u, base / "ncnn")
    print(f"  {profile}: rel={m['relative_max']:.3e} float_psnr={m['psnr_db']} "
          f"pixel_psnr={pm['psnr_db']} ssim={pm['ssim']} pass={pm['pass']} {dec_s}s", flush=True)
    return m


def main() -> int:
    root = Path(pc.ART)
    vis = Path(pc.VISUAL_OUT)
    report = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "real_vae_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "scaling_factor": pc.VAE_RUNNER_CFG["scaling_factor"],
        "shifting_factor": pc.VAE_RUNNER_CFG["shifting_factor"],
        "use_sample": False,
        "mode": "decoder end-to-end (official PyTorch vs ncnn FP32 stage chain)",
        "profiles": {},
    }
    overall = True
    for prof in ("A", "B"):
        try:
            res = run_profile(prof, root, vis)
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}", "pass": False}
            overall = False
            print(f"  {prof} FAILED: {e}", flush=True)
        report["profiles"][prof] = res
        if not res.get("pass"):
            overall = False
    report["overall_pass"] = overall
    report["thresholds"] = {
        "decoded_float_relative_max": REL_LIMIT,
        "decoded_png_psnr_min_db": PNG_PSNR_MIN,
        "decoded_png_ssim_min": PNG_SSIM_MIN,
    }
    pc.write_report(str(Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_vae_decoder_parity.json")), report)
    print("overall:", overall)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
