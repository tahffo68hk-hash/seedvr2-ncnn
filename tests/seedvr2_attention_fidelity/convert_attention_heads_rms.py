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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=Path, default=Path(os.environ.get("PNNX_EXE", ROOT.parent.parent / "build" / "pnnx-cpu" / "src" / "pnnx.exe")))
    args = parser.parse_args()
    artifact = ROOT / "artifacts" / "attention_core_fidelity" / "hybrid_fidelity_rms_rope444_grid_1x2x2"
    for head in range(4):
        convert(args.pnnx, artifact / "attention_heads" / f"head{head}", "[64,32],[64,64],[64,4],[64,4]")
    convert(args.pnnx, artifact / "output_projection", "[256,32]")


if __name__ == "__main__":
    main()
