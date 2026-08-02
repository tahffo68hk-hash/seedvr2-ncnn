"""Behavioral contract for masking padded dynamic-window tokens."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import export_fixture
from dynamic_mask_attention import DynamicMaskAttention


class DynamicMaskAttentionTest(unittest.TestCase):
    def test_padded_key_values_do_not_change_valid_token_outputs(self) -> None:
        """Changing only padded tokens must not change any valid query output."""
        torch.manual_seed(20260803)
        model = DynamicMaskAttention().eval()
        x = torch.randn((1, 4, 17, 17, 32), dtype=torch.float32)

        with torch.no_grad():
            windows, mask, valid, metadata = model.partition_with_mask(x)
            baseline = model.core(windows, mask)
            dirty = windows.clone()
            dirty[~valid] = torch.randn_like(dirty[~valid]) * 1000.0
            changed = model.core(dirty, mask)

        self.assertEqual(mask.shape, (25, 64, 64))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(valid.shape, (25, 64))
        self.assertEqual(metadata["padding"], (0, 3, 3))
        self.assertTrue(torch.allclose(baseline[valid], changed[valid], atol=1e-5, rtol=1e-5))

    def test_masked_dynamic_attention_reverse_crops_to_original_bthwc_shape(self) -> None:
        """The full masked path must return the original non-divisible BTHWC shape."""
        torch.manual_seed(20260803)
        model = DynamicMaskAttention().eval()
        x = torch.randn((1, 4, 17, 17, 32), dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, x.shape)

    def test_export_writes_two_input_core_and_mask_artifacts(self) -> None:
        """The ncnn unit must expose both partitioned windows and the additive mask."""
        artifacts = export_fixture()
        self.assertTrue((artifacts / "dynamic_mask_core.pt").is_file())
        self.assertTrue((artifacts / "mask.bin").is_file())
        self.assertTrue((artifacts / "reference_output.bin").is_file())


if __name__ == "__main__":
    unittest.main()
