"""Compare same-shape float32 raw tensor files and emit ncnn accuracy JSON."""

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_tensor.py <reference.bin> <actual.bin> <report.json>")
    reference = np.fromfile(sys.argv[1], dtype=np.float32)
    actual = np.fromfile(sys.argv[2], dtype=np.float32)
    if reference.shape != actual.shape:
        raise SystemExit(f"shape mismatch: {reference.shape} != {actual.shape}")
    abs_error = np.abs(reference - actual)
    denominator = np.maximum(np.abs(reference), 1e-12)
    report = {
        "max_abs_error": float(abs_error.max()),
        "mean_abs_error": float(abs_error.mean()),
        "max_relative_error": float((abs_error / denominator).max()),
        "nan_count": int(np.isnan(actual).sum()),
        "inf_count": int(np.isinf(actual).sum()),
    }
    Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
