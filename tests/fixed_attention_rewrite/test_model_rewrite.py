"""Behavioral tests for the full fixed-window QKV Gemm rewrite."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import export_fixture
from model_rewrite import FixedWindowAttentionRewrite


class FixedWindowAttentionRewriteTest(unittest.TestCase):
    def test_fixed_window_rewrite_preserves_bthwc_shape_and_matches_reference(self) -> None:
        """A fused 2D-QKV projection must reproduce the direct reference result."""
        torch.manual_seed(20260802)
        model = FixedWindowAttentionRewrite().eval()
        x = torch.randn((1, 2, 8, 8, 32), dtype=torch.float32)

        with torch.no_grad():
            output = model(x)
            reference = model.reference_forward(x)

        self.assertEqual(output.shape, (1, 2, 8, 8, 32))
        self.assertTrue(torch.allclose(output, reference, atol=1e-6, rtol=1e-5))


    def test_qkv_projection_is_flattened_to_a_2d_gemm_boundary(self) -> None:
        """The exported projection boundary is [windows*tokens, C] -> [windows*tokens, 3C]."""
        torch.manual_seed(20260802)
        model = FixedWindowAttentionRewrite().eval()
        x = torch.randn((1, 2, 8, 8, 32), dtype=torch.float32)

        with torch.no_grad():
            qkv, windows = model.project_qkv(x)

        self.assertEqual(windows.shape, (4, 32, 32))
        self.assertEqual(qkv.shape, (4, 32, 96))

    def test_export_writes_traceable_core_and_reference_tensors(self) -> None:
        """The migration unit must produce a core TorchScript artifact and BTHWC references."""
        artifacts = export_fixture()
        self.assertTrue((artifacts / "fixed_attention_rewrite_core.pt").is_file())
        self.assertTrue((artifacts / "reference_input.bin").is_file())
        self.assertTrue((artifacts / "reference_output.bin").is_file())


if __name__ == "__main__":
    unittest.main()
