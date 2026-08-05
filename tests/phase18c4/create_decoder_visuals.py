#!/usr/bin/env python3
"""Create non-committed visual evidence from a Pad decoder output and input tensor."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(rgb: np.ndarray) -> dict:
    data = rgb.astype(np.float64)
    return {
        "min": int(rgb.min()), "max": int(rgb.max()),
        "mean": float(data.mean()), "std": float(data.std()),
        "all_zero": bool(np.all(rgb == 0)),
        "constant": bool(rgb.min() == rgb.max()),
        "finite": bool(np.isfinite(data).all()), "nan_count": 0, "inf_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-f32", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor = np.fromfile(args.input_f32, dtype="<f4")
    expected = 3 * 5 * 128 * 128
    if tensor.size != expected or not np.isfinite(tensor).all():
        raise SystemExit(f"invalid input tensor: elements={tensor.size}, expected={expected}")
    tensor = tensor.reshape(3, 5, 128, 128)
    input_rgb = np.clip(np.rint((tensor[:, 0].transpose(1, 2, 0) * .5 + .5) * 255), 0, 255).astype(np.uint8)
    output_rgb = np.asarray(Image.open(args.output_png).convert("RGB"), dtype=np.uint8)
    if output_rgb.shape != input_rgb.shape:
        raise SystemExit(f"shape mismatch: input={input_rgb.shape}, output={output_rgb.shape}")
    input_path, output_path = args.output_dir / "input.png", args.output_dir / "output.png"
    side_path, diff_path = args.output_dir / "side_by_side.png", args.output_dir / "diff.png"
    Image.fromarray(input_rgb, "RGB").save(input_path)
    Image.fromarray(output_rgb, "RGB").save(output_path)
    canvas = Image.new("RGB", (256, 152), "white")
    canvas.paste(Image.fromarray(input_rgb, "RGB"), (0, 24)); canvas.paste(Image.fromarray(output_rgb, "RGB"), (128, 24))
    draw = ImageDraw.Draw(canvas); draw.text((6, 6), "INPUT frame 0", fill="black"); draw.text((134, 6), "PAD DECODER OUTPUT", fill="black")
    canvas.save(side_path)
    difference = np.abs(output_rgb.astype(np.int16) - input_rgb.astype(np.int16)).astype(np.uint8)
    Image.fromarray(np.clip(difference.astype(np.uint16) * 4, 0, 255).astype(np.uint8), "RGB").save(diff_path)
    delta = output_rgb.astype(np.float64) - input_rgb.astype(np.float64)
    report = {
        "schema_version": 1,
        "input_source": "Pad app internal phase18c_input/input_norm.f32, frame index 0",
        "output_source": "Pad VAE decoder frame_000.png restored from dec_up1 checkpoint",
        "decoder_only_checkpoint_recovery": True, "general_gallery_enhancement_flow": False,
        "shape_hwc": list(output_rgb.shape),
        "input": {**stats(input_rgb), "path": str(input_path), "sha256": sha256(input_path)},
        "output": {**stats(output_rgb), "path": str(output_path), "sha256": sha256(output_path)},
        "identical_to_input": bool(np.array_equal(input_rgb, output_rgb)),
        "difference": {"max_abs": float(np.abs(delta).max()), "mean_abs": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta * delta))), "visual_scale": 4, "path": str(diff_path)},
        "side_by_side_path": str(side_path),
    }
    (args.output_dir / "visual_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
