"""Contract tests for dynamic local-window metadata and lossless reverse mapping."""

from __future__ import annotations

import unittest

import torch

from dynamic_window import dynamic_window_partition, dynamic_window_reverse


class DynamicWindowFixtureTest(unittest.TestCase):
    def _round_trip(self, shape: tuple[int, int, int, int, int]):
        torch.manual_seed(sum(shape))
        x = torch.randn(shape, dtype=torch.float32)
        windows, metadata = dynamic_window_partition(x, base_window=(4, 4, 4))
        restored = dynamic_window_reverse(windows, metadata)
        self.assertTrue(torch.equal(restored, x))
        return windows, metadata

    def test_case_a_divisible_fixed_window_has_no_padding(self) -> None:
        windows, metadata = self._round_trip((1, 4, 16, 16, 32))
        self.assertEqual(metadata["window"], (4, 4, 4))
        self.assertEqual(metadata["padding"], (0, 0, 0))
        self.assertEqual(metadata["num_windows"], 16)
        self.assertEqual(windows.shape, (16, 64, 32))

    def test_case_b_short_time_shrinks_temporal_window(self) -> None:
        windows, metadata = self._round_trip((1, 3, 16, 16, 32))
        self.assertEqual(metadata["window"], (3, 4, 4))
        self.assertEqual(metadata["padding"], (0, 0, 0))
        self.assertEqual(metadata["num_windows"], 16)
        self.assertEqual(windows.shape, (16, 48, 32))

    def test_case_c_nondivisible_spatial_shape_pads_and_crops_losslessly(self) -> None:
        windows, metadata = self._round_trip((1, 4, 17, 17, 32))
        self.assertEqual(metadata["window"], (4, 4, 4))
        self.assertEqual(metadata["padding"], (0, 3, 3))
        self.assertEqual(metadata["padded_shape"], (1, 4, 20, 20, 32))
        self.assertEqual(metadata["num_windows"], 25)
        self.assertEqual(windows.shape, (25, 64, 32))


if __name__ == "__main__":
    unittest.main()
