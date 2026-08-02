"""Break caught: flatten/restore changes QKV projection values or layout."""

import unittest

import torch

from model_original import OriginalQkvProjection
from model_rewrite import RewriteQkvProjection


class RewriteEquivalenceTests(unittest.TestCase):
    def test_flattened_gemm_projection_matches_original_3d_projection(self) -> None:
        torch.manual_seed(20250804)
        original = OriginalQkvProjection().eval()
        rewrite = RewriteQkvProjection().eval()
        rewrite.load_state_dict(original.state_dict())
        x = torch.randn((4, 32, 32), dtype=torch.float32)
        self.assertEqual(tuple(rewrite(x).shape), (4, 32, 96))
        self.assertTrue(torch.equal(original(x), rewrite(x)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
