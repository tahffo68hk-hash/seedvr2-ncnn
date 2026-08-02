"""Apply the Case-C host-side dynamic reverse/crop to an ncnn core output."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from dynamic_mask_attention import DynamicMaskAttention
from dynamic_window import dynamic_window_reverse


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: postprocess.py <ncnn_core_output.bin> <bthwc_output.bin>")
    root = Path(__file__).resolve().parent / "artifacts"
    x = torch.load(root / "reference_input.pt", map_location="cpu", weights_only=True)
    model = DynamicMaskAttention().eval()
    _, _, _, metadata = model.partition_with_mask(x)
    raw = np.fromfile(sys.argv[1], dtype=np.float32).reshape(25, 64, 32)
    restored = dynamic_window_reverse(torch.from_numpy(raw), metadata)
    restored.numpy().tofile(sys.argv[2])


if __name__ == "__main__":
    main()
