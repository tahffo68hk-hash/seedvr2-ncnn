#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: host VAE runtime that chains the exported FP32 ncnn stage
artifacts (VaeEncoderRuntime / VaeDecoderRuntime).

Each stage runs through the vae_stage_runner executable (single-threaded
FP32 CPU).  The runtime owns lifecycle semantics (initialize/run/clear_cache/
release), call counts and resident-byte estimates; it never falls back to
PyTorch.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNNER = Path(r"D:\AIProjects\seedvr2-ncnn\build\phase17-vae-smoke\vae_stage_runner.exe")
STAGES_ENC = ["enc_conv_in", "enc_down0", "enc_down1", "enc_down2", "enc_down3", "enc_mid", "enc_out"]
STAGES_DEC = ["dec_conv_in", "dec_mid", "dec_up0", "dec_up1", "dec_up2", "dec_up3", "dec_out"]


def ncnn_dims(shape):
    """torch BCTHW -> ncnn Mat dims (w,h,d,c) for batch=1."""
    _, c, t, h, w = shape
    return (w, h, t, c)


def _run_stage_exe(param: Path, bin_: Path, x: np.ndarray, in_dims, out_dims) -> np.ndarray:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vae_rt_") as td:
        inp = Path(td) / "in.bin"
        outp = Path(td) / "out.bin"
        x.astype("<f4").tofile(str(inp))
        cmd = [str(RUNNER), str(param), str(bin_), str(inp), str(outp)]
        cmd += [str(d) for d in in_dims] + ["--"] + [str(d) for d in out_dims]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"stage {param.parent.name} runner rc={r.returncode}: {r.stderr.strip()[:200]}")
        return np.fromfile(str(outp), dtype="<f4")


class VaeEncoderRuntime:
    """Streaming FP32 ncnn VAE encoder (stage chain)."""

    def __init__(self, profile: str, root: str = None):
        import phase17_common as pc  # local import to avoid circulars
        self.profile = profile
        self.root = Path(root or pc.ART)
        self.stages = STAGES_ENC
        self.call_count = 0
        self.loaded = False
        self.resident_bytes = 0
        self._nets = {}
        self.load()

    def load(self):
        for stage in self.stages:
            d = self.root / f"stages_{self.profile}" / stage
            p, b = d / "model.ncnn.param", d / "model.ncnn.bin"
            if not b.exists():
                raise FileNotFoundError(f"missing artifact {b}")
            self._nets[stage] = (p, b, b.stat().st_size)
        self.resident_bytes = sum(sz for _, _, sz in self._nets.values())
        self.loaded = True

    def run_stage(self, stage: str, x: np.ndarray, in_shape, out_shape):
        p, b, _ = self._nets[stage]
        out = _run_stage_exe(p, b, x, ncnn_dims(in_shape), ncnn_dims(out_shape))
        self.call_count += 1
        return out

    def encode(self, x_bcthw):
        """x_bcthw: numpy (B,C,T,H,W) -> latent raw (pre-scale) (B,16,T,h,w)."""
        import torch
        if not self.loaded:
            raise RuntimeError("encoder not loaded")
        h = x_bcthw.numpy() if hasattr(x_bcthw, "numpy") else np.asarray(x_bcthw, dtype=np.float32)
        in_shape = list(h.shape)
        for stage in self.stages:
            out_shape_i = out_shape(self.root, self.profile, stage, in_shape)
            out = self.run_stage(stage, h, in_shape, out_shape_i)
            h = out
            in_shape = list(out_shape_i)
        # first 16 channels are the deterministic mean
        return torch.from_numpy(h.reshape(in_shape))[:, :16].contiguous()

    def clear_cache(self):
        # runner processes exit after each call; nothing resident
        return True

    def release(self):
        self._nets.clear()
        self.resident_bytes = 0
        self.loaded = False
        return True


_SHAPES = None


def _load_shapes(root):
    """Stage in/out torch shapes from the captured reference manifest."""
    global _SHAPES
    if _SHAPES is None:
        import json
        _SHAPES = {}
        for prof in ("A", "B"):
            m = json.loads((Path(root) / f"reference_{prof}" / "manifest.json").read_text(encoding="utf-8"))
            _SHAPES[prof] = {st: {"in": info["in"]["shape"], "out": info["out"]["shape"]}
                             for st, info in m["stage_files"].items()}
    return _SHAPES


def out_shape(root, profile, stage, in_shape):
    shapes = _load_shapes(root)[profile][stage]
    return list(shapes["out"])


class VaeDecoderRuntime:
    """Streaming FP32 ncnn VAE decoder (stage chain)."""

    def __init__(self, profile: str, root: str = None):
        import phase17_common as pc
        self.profile = profile
        self.root = Path(root or pc.ART)
        self.stages = STAGES_DEC
        self.call_count = 0
        self.loaded = False
        self.resident_bytes = 0
        self._nets = {}
        self.load()

    def load(self):
        for stage in self.stages:
            d = self.root / f"stages_{self.profile}" / stage
            p, b = d / "model.ncnn.param", d / "model.ncnn.bin"
            if not b.exists():
                raise FileNotFoundError(f"missing artifact {b}")
            self._nets[stage] = (p, b, b.stat().st_size)
        self.resident_bytes = sum(sz for _, _, sz in self._nets.values())
        self.loaded = True

    def run_stage(self, stage, x, in_shape):
        out_shape_i = out_shape(self.root, self.profile, stage, in_shape)
        p, b, _ = self._nets[stage]
        out = _run_stage_exe(p, b, x, ncnn_dims(in_shape), ncnn_dims(out_shape_i))
        self.call_count += 1
        return out, out_shape_i

    def decode(self, latent_bcthw):
        import torch
        if not self.loaded:
            raise RuntimeError("decoder not loaded")
        h = latent_bcthw.numpy() if hasattr(latent_bcthw, "numpy") else np.asarray(latent_bcthw, dtype=np.float32)
        in_shape = list(h.shape)
        for stage in self.stages:
            h, in_shape = self.run_stage(stage, h, in_shape)
        return torch.from_numpy(h.reshape(in_shape))

    def clear_cache(self):
        return True

    def release(self):
        self._nets.clear()
        self.resident_bytes = 0
        self.loaded = False
        return True
