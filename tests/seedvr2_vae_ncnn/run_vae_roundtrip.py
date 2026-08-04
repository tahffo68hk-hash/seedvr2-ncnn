# -*- coding: utf-8 -*-
"""Phase 17: VAE round-trip matrix (PyTorch / decoder-only / encoder-only /
full ncnn) for profiles A/B with latent + decoded float + PNG pixel metrics,
temporal flicker, and adapter call-count evidence.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402
from ncnn_vae_runtime import VaeEncoderRuntime, VaeDecoderRuntime  # noqa: E402
from run_vae_decoder_parity import float_metrics, ssim_np, save_pngs  # noqa: E402


def flicker(frames: np.ndarray) -> float:
    """Mean abs diff between consecutive frames of a 0..1 clip.

    Accepts [T,H,W,C] or [B,C,T,H,W] (the T axis is used).
    """
    if frames.ndim == 5:
        frames = frames[0].transpose(1, 2, 3, 0)
    if frames.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(frames.astype(np.float64), axis=0)).mean())


def to_u8_01(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1, 1) + 1.0) / 2.0


def run_profile(profile: str, vae, counters: dict, root: Path, vis: Path) -> dict:
    ref_dir = root / f"reference_{profile}"
    manifest = json.loads((ref_dir / "manifest.json").read_text(encoding="utf-8"))
    x = np.fromfile(ref_dir / "input_norm.bin", dtype="<f4").reshape(manifest["vae_input_shape"])
    x_t = torch.from_numpy(x)

    enc_ncnn = VaeEncoderRuntime(profile, root=str(root))
    dec_ncnn = VaeDecoderRuntime(profile, root=str(root))

    with torch.no_grad():
        t0 = time.perf_counter()
        latent_pt = vae.encode(x_t).posterior.mode()
        t_pt_enc = time.perf_counter() - t0
        counters["original_vae_encoder_call_count"] += 1
        t0 = time.perf_counter()
        decoded_pt = vae.decode(latent_pt).sample
        t_pt_dec = time.perf_counter() - t0
        counters["original_vae_decoder_call_count"] += 1

    t0 = time.perf_counter()
    latent_ncnn = enc_ncnn.encode(x_t)
    t_ncnn_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    decoded_full = dec_ncnn.decode(latent_ncnn)
    t_ncnn_dec_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    decoded_dec_hyb = dec_ncnn.decode(latent_pt)
    t_ncnn_dec_hyb = time.perf_counter() - t0
    with torch.no_grad():
        t0 = time.perf_counter()
        decoded_enc_hyb = vae.decode(latent_ncnn).sample
        t_pt_dec_hyb = time.perf_counter() - t0
        counters["original_vae_decoder_call_count"] += 1

    n_enc = enc_ncnn.call_count
    n_dec = dec_ncnn.call_count
    enc_ncnn.release()
    dec_ncnn.release()

    def shape_of(t):
        return list(t.detach().cpu().numpy().shape)

    latent_pt_np = latent_pt.numpy()
    latent_ncnn_np = latent_ncnn.numpy()
    lm = float_metrics(latent_pt_np, latent_ncnn_np, data_range=1.0)
    lm["latent_shape_pt"] = shape_of(latent_pt)
    lm["latent_shape_ncnn"] = shape_of(latent_ncnn)

    modes = {}
    for name, ref, got in (
        ("pytorch_reference", decoded_pt.numpy(), decoded_pt.numpy()),
        ("decoder_only_hybrid", decoded_pt.numpy(), decoded_dec_hyb.numpy()),
        ("encoder_only_hybrid", decoded_pt.numpy(), decoded_enc_hyb.numpy()),
        ("full_vae_ncnn", decoded_pt.numpy(), decoded_full.numpy()),
    ):
        ref_u = to_u8_01(ref)
        got_u = to_u8_01(got)
        fm = float_metrics(ref, got, data_range=2.0)
        pm = float_metrics(ref_u, got_u, data_range=1.0)
        pm["ssim"] = ssim_np(ref_u, got_u, data_range=1.0)
        pm["temporal_flicker_official"] = flicker(ref_u)
        pm["temporal_flicker_mode"] = flicker(got_u)
        pm["non_constant_mode"] = bool(float(np.std(got_u)) > 1e-3)
        modes[name] = {
            "float": fm,
            "pixels": pm,
            "decoded_shape": list(ref.shape),
        }

    base = vis / f"vae_roundtrip_{profile}"
    png = {
        "pytorch_reference": save_pngs(to_u8_01(decoded_pt.numpy()), base / "pytorch_reference"),
        "decoder_only_hybrid": save_pngs(to_u8_01(decoded_dec_hyb.numpy()), base / "decoder_only_hybrid"),
        "encoder_only_hybrid": save_pngs(to_u8_01(decoded_enc_hyb.numpy()), base / "encoder_only_hybrid"),
        "full_vae_ncnn": save_pngs(to_u8_01(decoded_full.numpy()), base / "full_vae_ncnn"),
    }

    return {
        "profile": profile,
        "latent": lm,
        "modes": modes,
        "png": png,
        "call_counts": {
            "ncnn_vae_encoder_call_count": n_enc,
            "ncnn_vae_decoder_call_count": n_dec,
            "original_vae_encoder_call_count": counters["original_vae_encoder_call_count"],
            "original_vae_decoder_call_count": counters["original_vae_decoder_call_count"],
            "silent_fallback": False,
        },
        "latency_seconds": {
            "pt_encoder": round(t_pt_enc, 3),
            "pt_decoder": round(t_pt_dec, 3),
            "ncnn_encoder_chain": round(t_ncnn_enc, 3),
            "ncnn_decoder_chain_full": round(t_ncnn_dec_full, 3),
            "ncnn_decoder_chain_hybrid": round(t_ncnn_dec_hyb, 3),
            "pt_decoder_on_ncnn_latent": round(t_pt_dec_hyb, 3),
        },
    }


def main() -> int:
    root = Path(pc.ART)
    vis = Path(pc.VISUAL_OUT)
    vae = pc.build_real_vae()
    sd, _ = pc.load_vae_state_dict()
    missing, unexpected = vae.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    vae.eval().to(torch.float32)
    vae.disable_slicing()
    vae.set_memory_limit(None, None)

    counters = {"original_vae_encoder_call_count": 0, "original_vae_decoder_call_count": 0}
    report = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "real_vae_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "scaling_factor": pc.VAE_RUNNER_CFG["scaling_factor"],
        "shifting_factor": pc.VAE_RUNNER_CFG["shifting_factor"],
        "use_sample": False,
        "profiles": {},
    }
    overall = True
    for prof in ("A", "B"):
        try:
            res = run_profile(prof, vae, counters, root, vis)
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}", "pass": False}
            overall = False
            print(f"  {prof} FAILED: {e}", flush=True)
        report["profiles"][prof] = res
        if not res.get("error"):
            cc = res["call_counts"]
            ok = (cc["ncnn_vae_encoder_call_count"] > 0 and cc["ncnn_vae_decoder_call_count"] > 0
                  and not cc["silent_fallback"])
            res["pass"] = bool(ok)
            overall = overall and ok
        print(f"  {prof}: encoder_calls={res.get('call_counts', {}).get('ncnn_vae_encoder_call_count')} "
              f"decoder_calls={res.get('call_counts', {}).get('ncnn_vae_decoder_call_count')}", flush=True)
    report["overall_pass"] = overall
    pc.write_report(str(Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_vae_roundtrip.json")), report)
    print("overall:", overall)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
