"""Compare the BTHWC float32 reference with ncnn runner output."""

from __future__ import annotations

import json
import sys

import numpy as np


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare.py <reference.bin> <actual.bin>")
    reference = np.fromfile(sys.argv[1], dtype=np.float32)
    actual = np.fromfile(sys.argv[2], dtype=np.float32)
    if reference.shape != actual.shape:
        raise SystemExit(f"shape mismatch {reference.shape} != {actual.shape}")
    error = np.abs(reference - actual)
    print(json.dumps({"max_abs_error": float(error.max()), "mean_abs_error": float(error.mean())}, indent=2))


if __name__ == "__main__":
    main()
