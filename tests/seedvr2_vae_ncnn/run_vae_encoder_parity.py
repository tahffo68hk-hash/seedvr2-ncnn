# -*- coding: utf-8 -*-
"""Phase 17: VAE encoder end-to-end parity (official PyTorch vs ncnn FP32)."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402
from ncnn_vae_runtime import VaeEncoderRuntime  # noqa: E402

REL_LIMIT = 2e-3  # READY: Profile A/B latent relative max < 2e-3


def sha256_bytes(b: np.ndarray) -> str:
    return hashlib.sha256(b.astype("<f4").tobytes()).hexdigest()


def metrics(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> dict:
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
        "mse": mse,
        "psnr_db": float(10.0 * np.log10(data_range * data_range / mse)) if mse > 0 else None,
        "ref_max": denom,
        "nan": int(np.isnan(b).sum()),
        "inf": int(np.isinf(b).sum()),
    }


def run_profile(profile: str, root: Path, tmp: Path) -> dict:
    ref_dir = root / f"reference_{profile}"
    manifest = json.loads((ref_dir / "manifest.json").read_text(encoding="utf-8"))
    x = np.fromfile(ref_dir / "input_norm.bin", dtype="<f4").reshape(manifest["vae_input_shape"])
    ref = np.fromfile(ref_dir / "latent_raw.bin", dtype="<f4").reshape(manifest["latent_shape"])

    rt = VaeEncoderRuntime(profile, root=str(root))
    t0 = time.perf_counter()
    got = rt.encode(x).numpy().reshape(manifest["latent_shape"])
    enc_s = round(time.perf_counter() - t0, 3)
    rt.release()

    m = metrics(ref, got)
    m["latent_shape_official"] = list(ref.shape)
    m["latent_shape_ncnn"] = list(got.shape)
    m["shape_match"] = list(ref.shape) == list(got.shape)
    m["latent_sha256_official"] = sha256_bytes(ref)
    m["latent_sha256_ncnn"] = sha256_bytes(got)
    m["threshold_rel_max"] = REL_LIMIT
    m["pass"] = bool(m["shape_match"] and m["relative_max"] < REL_LIMIT
                     and m["nan"] == 0 and m["inf"] == 0)
    m["encoder_chain_seconds"] = enc_s
    m["encoder_call_count"] = rt.call_count
    m["input_shape"] = manifest["vae_input_shape"]
    m["note"] = "ncnn chain runs pre-scale latent; post-scale = (latent - shift) * scaling_factor"
    print(f"  {profile}: rel={m['relative_max']:.3e} max_abs={m['max_abs_error']:.3e} "
          f"psnr={m['psnr_db']} pass={m['pass']} {enc_s}s", flush=True)
    return m


def main() -> int:
    root = Path(pc.ART)
    report = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "real_vae_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "scaling_factor": pc.VAE_RUNNER_CFG["scaling_factor"],
        "shifting_factor": pc.VAE_RUNNER_CFG["shifting_factor"],
        "use_sample": False,
        "mode": "encoder end-to-end (official PyTorch vs ncnn FP32 stage chain)",
        "profiles": {},
    }
    overall = True
    for prof in ("A", "B"):
        try:
            res = run_profile(prof, root, Path(pc.EXPORT_TMP) / "enc_parity")
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}", "pass": False}
            overall = False
            print(f"  {prof} FAILED: {e}", flush=True)
        report["profiles"][prof] = res
        if not res.get("pass"):
            overall = False
    report["overall_pass"] = overall
    report["thresholds"] = {"encoder_latent_relative_max": REL_LIMIT}
    pc.write_report(str(Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_vae_encoder_parity.json")), report)
    print("overall:", overall)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
