"""Convert the RMS hybrid artifact with FP32 ncnn weights.

pnnx defaults to fp16=1, which is unsuitable for this FP32 numerical fixture:
the QKV Gemm alone then exceeds the 1e-4 acceptance bound.  Keep fp16=0 on
every staged head and the output projection.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from export_attention_heads_rms import ROOT


def convert(pnnx: Path, directory: Path, shape: str) -> None:
    subprocess.run([str(pnnx), "model.pt", f"inputshape={shape}", "fp16=0"], cwd=directory, check=True)


def convert_hd128(pnnx: Path) -> None:
    """Convert the shared Phase 10 synthetic head_dim=128 head/projection models.

    The S profile owns the ncnn files; the M and P profiles hardlink them.
    Every head takes [64,512] windows, [64,64] additive mask and [64,64]
    RoPE cos/sin; the projection takes [64,512] (ncnn Gemm adapts M).
    """
    import os as _os
    grids = ["1x1x1", "1x2x2", "2x3x3"]
    source = ROOT / "artifacts" / "attention_core_fidelity" / "hybrid_synthetic_hd128_rope444_grid_1x1x1"
    for head in range(4):
        convert(pnnx, source / "attention_heads" / f"head{head}", "[64,512],[64,64],[64,64],[64,64]")
    convert(pnnx, source / "output_projection", "[64,512]")
    for grid in grids[1:]:
        target = ROOT / "artifacts" / "attention_core_fidelity" / f"hybrid_synthetic_hd128_rope444_grid_{grid}"
        for head in range(4):
            src_dir = source / "attention_heads" / f"head{head}"
            dst_dir = target / "attention_heads" / f"head{head}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in ("model.ncnn.param", "model.ncnn.bin"):
                dst = dst_dir / name
                if dst.exists():
                    _os.remove(dst)
                _os.link(src_dir / name, dst)
        src_dir = source / "output_projection"
        dst_dir = target / "output_projection"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.ncnn.param", "model.ncnn.bin"):
            dst = dst_dir / name
            if dst.exists():
                _os.remove(dst)
            _os.link(src_dir / name, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=Path, default=Path(os.environ.get("PNNX_EXE", ROOT.parent.parent / "build" / "pnnx-cpu" / "src" / "pnnx.exe")))
    parser.add_argument("--hd128", action="store_true", help="convert Phase 10 synthetic head_dim=128 heads/projection")
    args = parser.parse_args()
    if args.hd128:
        convert_hd128(args.pnnx)
        return
    artifact = ROOT / "artifacts" / "attention_core_fidelity" / "hybrid_fidelity_rms_rope444_grid_1x2x2"
    for head in range(4):
        convert(args.pnnx, artifact / "attention_heads" / f"head{head}", "[64,32],[64,64],[64,4],[64,4]")
    convert(args.pnnx, artifact / "output_projection", "[256,32]")


if __name__ == "__main__":
    main()
