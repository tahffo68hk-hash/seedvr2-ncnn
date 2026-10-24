import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phase8_2_conversion_guard import (
    validate_artifact_integrity,
    validate_pnnx_command,
)


class Phase82ConversionGuardTests(unittest.TestCase):
    def test_rejects_missing_or_enabled_fp16(self):
        with self.assertRaisesRegex(ValueError, "fp16=0"):
            validate_pnnx_command(["pnnx", "model.pt", "inputshape=[64,32]"])
        with self.assertRaisesRegex(ValueError, "fp16=0"):
            validate_pnnx_command(["pnnx", "model.pt", "fp16=1"])

    def test_rejects_non_fp32_metadata(self):
        metadata = {
            "dtype": "fp32",
            "features": {
                "pnnx_fp16": False,
                "weight_dtype": "fp16",
                "activation_dtype": "fp32",
            },
        }
        with self.assertRaisesRegex(ValueError, "weight_dtype"):
            validate_artifact_integrity(Path("unused"), metadata, {})

    def test_detects_sha256_mismatch_with_path_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.ncnn.param"
            model.write_bytes(b"phase8.2")
            metadata = {
                "dtype": "fp32",
                "features": {
                    "pnnx_fp16": False,
                    "weight_dtype": "fp32",
                    "activation_dtype": "fp32",
                },
            }
            with self.assertRaisesRegex(ValueError, "model.ncnn.param.*SHA256 mismatch"):
                validate_artifact_integrity(root, metadata, {"model.ncnn.param": "0" * 64})

    def test_accepts_explicit_fp32_command_metadata_and_hash(self):
        validate_pnnx_command(["pnnx", "model.pt", "inputshape=[64,32]", "fp16=0"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.ncnn.bin"
            model.write_bytes(b"full-fp32-weights")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            metadata = {
                "dtype": "fp32",
                "features": {
                    "pnnx_fp16": False,
                    "weight_dtype": "fp32",
                    "activation_dtype": "fp32",
                },
            }
            validate_artifact_integrity(root, metadata, {"model.ncnn.bin": digest})


if __name__ == "__main__":
    unittest.main()
