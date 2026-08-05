#!/usr/bin/env python3
"""Prepare the ignored Phase 18C device model bundle and tiny committed PNG fixture.

The deployment index maps each device-relative path to an existing local file;
it avoids duplicating the 14+ GiB Phase 16/17 artifacts.  The device manifest
contains only relative paths, byte counts and SHA256 values.  No model is
placed in the APK or Git.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MAIN = Path(r"D:\AIProjects\seedvr2-ncnn")
OUT = REPO / ".phase18c_device_models"
ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "phase18c_demo"
CHECKPOINT = MAIN / ".phase14_local_checkpoint" / "seedvr2_ema_3b.pth"
PNNX = MAIN / "build" / "pnnx-cpu" / "src" / "pnnx.exe"
LIBTORCH = Path(r"V:\AI\LibTorch\2.7.0+cpu\lib")
VAE = MAIN / ".phase17_local_artifacts" / "stages_A"
BLOCKS = MAIN / ".phase16_local_artifacts"
CHAIN_INPUT = MAIN / ".phase17_visual_outputs" / "run_full_ncnn_core" / "chain_input"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_frames() -> None:
    """Four deterministic 128x128 moving checker/gradient/texture frames."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:128, 0:128]
    rng = np.random.default_rng(1803)
    texture = rng.integers(0, 28, size=(128, 128), dtype=np.uint8)
    for frame in range(4):
        checker = (((x + frame * 7) // 12 + (y + frame * 5) // 12) & 1) * 78
        edge = ((x - (22 + frame * 18)) >= 0).astype(np.uint8) * 62
        red = (x * 2 + checker + texture + frame * 11) % 256
        green = (y * 2 + np.roll(texture, frame * 3, axis=1) + edge) % 256
        blue = ((x + y) + np.roll(checker, frame * 4, axis=0) + frame * 29) % 256
        rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
        Image.fromarray(rgb, "RGB").save(ASSETS / f"frame_{frame:03d}.png", optimize=True)


class Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.fc = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(weight)
            self.fc.bias.copy_(bias)

    def forward(self, value):
        return self.fc(value)


def export_linear(name: str, weight: torch.Tensor, bias: torch.Tensor, rows: int) -> Path:
    directory = OUT / "dit" / name
    directory.mkdir(parents=True, exist_ok=True)
    model = Linear(weight, bias).eval()
    traced = torch.jit.trace(model, torch.zeros(rows, weight.shape[1], dtype=torch.float32))
    traced.save(str(directory / "model.pt"))
    env = os.environ.copy()
    env["PATH"] = str(LIBTORCH) + os.pathsep + env.get("PATH", "")
    subprocess.run([str(PNNX), "model.pt", f"inputshape=[{rows},{weight.shape[1]}]", "fp16=0"],
                   cwd=directory, env=env, check=True)
    for path in directory.glob("model.*"):
        if path.name not in ("model.ncnn.param", "model.ncnn.bin"):
            path.unlink()
    return directory


def add(index: list[dict], relative: str, source: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    index.append({"relative_path": relative.replace("\\", "/"), "source_path": str(source)})


def main() -> int:
    make_frames()
    OUT.mkdir(parents=True, exist_ok=True)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    vid_in = export_linear("vid_in", state["vid_in.proj.weight"], state["vid_in.proj.bias"], 128)
    vid_out = export_linear("vid_out", state["vid_out.proj.weight"], state["vid_out.proj.bias"], 128)
    post = OUT / "dit" / "post"
    post.mkdir(parents=True, exist_ok=True)
    for key, filename in (("vid_out_norm.weight", "norm_weight.f32"),
                          ("vid_out_ada.out_shift", "out_shift.f32"),
                          ("vid_out_ada.out_scale", "out_scale.f32")):
        state[key].detach().contiguous().numpy().astype("<f4").tofile(post / filename)
    del state

    torch.manual_seed(20260803)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the verified Phase 17 bfloat16 noise")
    noise = torch.randn((2, 16, 16, 16), device="cuda", dtype=torch.bfloat16).float().cpu()
    constants = OUT / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    noise.numpy().astype("<f4").tofile(constants / "noise.f32")
    shutil.copyfile(CHAIN_INPUT / "input_txt.bin", constants / "input_txt.f32")
    shutil.copyfile(CHAIN_INPUT / "emb.bin", constants / "emb.f32")

    index: list[dict] = []
    encoder = ["enc_conv_in", "enc_down0", "enc_down1", "enc_down2", "enc_down3", "enc_mid", "enc_out"]
    decoder = ["dec_conv_in", "dec_mid", "dec_up0", "dec_up1", "dec_up2", "dec_up3", "dec_out"]
    for kind, names in (("encoder", encoder), ("decoder", decoder)):
        for name in names:
            for filename in ("model.ncnn.param", "model.ncnn.bin"):
                add(index, f"vae/{kind}/{name}/{filename}", VAE / name / filename)
    for block in range(32):
        variant = f"real_seedvr2_block{block}_c2560_h20_hd128"
        root = BLOCKS / variant
        add(index, f"dit/blocks/{variant}/metadata.json", root / "metadata.json")
        for source in sorted((root / "params").glob("*.bin")):
            add(index, f"dit/blocks/{variant}/params/{source.name}", source)
        for source in sorted((root / "models").glob("*/model.ncnn.*")):
            add(index, f"dit/blocks/{variant}/models/{source.parent.name}/{source.name}", source)
    for root, prefix in ((vid_in, "dit/vid_in"), (vid_out, "dit/vid_out"),
                         (post, "dit/post"), (constants, "constants")):
        for source in sorted(root.glob("*")):
            if source.is_file():
                add(index, f"{prefix}/{source.name}", source)

    files = []
    total = 0
    for position, item in enumerate(index, 1):
        source = Path(item["source_path"])
        size = source.stat().st_size
        files.append({"relative_path": item["relative_path"], "size_bytes": size, "sha256": sha256(source)})
        total += size
        if position % 50 == 0:
            print(f"hashed {position}/{len(index)} files", flush=True)
    manifest = {"schema_version": 1, "model_set": "phase18c_minimal_video_demo", "files": files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    index.append({"relative_path": "manifest.json", "source_path": str(OUT / "manifest.json")})
    (OUT / "deploy_index.json").write_text(json.dumps({"files": index}, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 1, "model_set": manifest["model_set"], "file_count": len(files),
        "total_size_bytes": total, "manifest_sha256": sha256(OUT / "manifest.json"),
        "input_frames": [{"name": p.name, "sha256": sha256(p), "size_bytes": p.stat().st_size}
                         for p in sorted(ASSETS.glob("*.png"))],
        "fixed_only": True, "precomputed_final_latent": False, "precomputed_output_frames": False,
    }
    (OUT / "bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
