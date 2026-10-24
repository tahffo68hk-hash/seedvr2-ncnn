"""Strict FP32 and SHA256 guards for Phase 8.2 pnnx artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence


def validate_pnnx_command(command: Sequence[str]) -> None:
    fp16_options = [argument for argument in command if argument.startswith("fp16=")]
    if fp16_options != ["fp16=0"]:
        raise ValueError("pnnx command must contain exactly one explicit fp16=0 option")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact_integrity(
    artifact_root: Path,
    metadata: Mapping[str, object],
    expected_hashes: Mapping[str, str],
) -> None:
    if metadata.get("dtype") != "fp32":
        raise ValueError("dtype must be fp32")
    features = metadata.get("features")
    if not isinstance(features, dict):
        raise ValueError("features must be an object")
    if features.get("pnnx_fp16") is not False:
        raise ValueError("features.pnnx_fp16 must be false")
    for field in ("weight_dtype", "activation_dtype"):
        if features.get(field) != "fp32":
            raise ValueError(f"features.{field} must be fp32")

    for relative, expected in expected_hashes.items():
        path = artifact_root / relative
        if not path.is_file():
            raise ValueError(f"{relative}: artifact file missing")
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"{relative}: SHA256 mismatch expected={expected.lower()} actual={actual}"
            )
