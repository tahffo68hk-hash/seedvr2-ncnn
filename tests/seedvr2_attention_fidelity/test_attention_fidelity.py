"""Phase 7 fidelity tests for the SeedVR2 QK-norm and mmrope3d path."""

from __future__ import annotations

import unittest

import torch

from attention_fidelity import FidelityAttention, partition_bthwc, reverse_bthwc
from rope3d_reference import apply_mmrope3d, mmrope3d_frequencies


class AttentionFidelityTests(unittest.TestCase):
    def test_independent_and_source_equivalent_references_match(self) -> None:
        torch.manual_seed(20260802)
        model = FidelityAttention().eval()
        x = torch.randn(1, 4, 8, 8, 32)
        with torch.no_grad():
            independent = model.forward_independent(x)
            source_equivalent = model.forward_source_equivalent(x)
        self.assertEqual(independent.shape, x.shape)
        self.assertLess(float((independent - source_equivalent).abs().max()), 1e-6)

    def test_mmrope3d_changes_all_three_axes_and_preserves_zero_position(self) -> None:
        q = torch.randn(4 * 8 * 8, 4, 8)
        frequencies = mmrope3d_frequencies((4, 8, 8), head_dim=8, text_length=0)
        rotated = apply_mmrope3d(q, frequencies)
        self.assertTrue(torch.equal(q[0], rotated[0]))
        self.assertFalse(torch.equal(rotated[0], rotated[1]))
        self.assertFalse(torch.equal(rotated[0], rotated[8]))
        self.assertFalse(torch.equal(rotated[0], rotated[64]))

    def test_partition_reverse_handles_padding_and_preserves_bthwc(self) -> None:
        x = torch.randn(1, 5, 9, 10, 32)
        windows, metadata = partition_bthwc(x, (4, 4, 4))
        restored = reverse_bthwc(windows, metadata)
        self.assertEqual(tuple(windows.shape), (18, 64, 32))
        self.assertTrue(torch.equal(restored, x))


if __name__ == "__main__":
    unittest.main()
