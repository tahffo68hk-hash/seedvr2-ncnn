"""Mathematical RMSNorm contract tests for the Phase 8 fixture."""
import unittest

import torch

from rmsnorm_reference import RMSNormReference
from rmsnorm_rewrite import RMSNormRewriteA, RMSNormRewriteB, RMSNormRewriteC, RMSNormRewriteD


class RMSNormFixtureTests(unittest.TestCase):
    def test_all_rewrites_match_reference_on_fixed_edge_cases(self):
        torch.manual_seed(20260804)
        weight = torch.linspace(0.5, 1.5, 8)
        cases = [
            torch.randn(4, 64, 8),
            torch.zeros(4, 64, 8),
            torch.full((4, 64, 8), 1.0e-20),
            torch.tensor([[[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]]]).repeat(4, 64, 1),
            torch.linspace(-1.0e6, 1.0e6, 4 * 64 * 8).reshape(4, 64, 8),
        ]
        reference = RMSNormReference(8, eps=1.0e-5, weight=weight)
        rewrites = [RMSNormRewriteA(8, 1.0e-5, weight), RMSNormRewriteB(8, 1.0e-5, weight),
                    RMSNormRewriteC(8, 1.0e-5, weight), RMSNormRewriteD(8, 1.0e-5, weight)]
        for values in cases:
            expected = reference(values)
            self.assertTrue(torch.isfinite(expected).all())
            for rewrite in rewrites:
                actual = rewrite(values)
                self.assertLess(float((actual - expected).abs().max()), 1.0e-6)

    def test_normalization_is_per_token_and_has_no_bias(self):
        torch.manual_seed(17)
        values = torch.randn(4, 64, 8)
        model = RMSNormReference(8, eps=1.0e-5, weight=torch.ones(8))
        changed = values.clone()
        changed[2, 19] += 10.0
        output = model(values)
        changed_output = model(changed)
        self.assertTrue(torch.equal(output[:2], changed_output[:2]))
        self.assertTrue(torch.equal(output[3:], changed_output[3:]))
        self.assertTrue(torch.allclose(output[2, 19].square().mean(), torch.tensor(1.0), atol=1.0e-5))
        self.assertEqual(model.bias, None)

    def test_single_channel_and_window_amplitudes_do_not_mix_statistics(self):
        model = RMSNormReference(8, eps=1.0e-5, weight=torch.ones(8))
        values = torch.zeros(3, 4, 8)
        values[0, 0, 0] = 3.0
        values[0, 1, 0] = 6.0
        values[1, 0, 0] = -9.0
        output = model(values)
        for window, token, amplitude in ((0, 0, 3.0), (0, 1, 6.0), (1, 0, -9.0)):
            expected = amplitude / torch.sqrt(torch.tensor(amplitude * amplitude / 8.0 + 1.0e-5))
            self.assertTrue(torch.allclose(output[window, token, 0], expected, atol=1.0e-6))
            self.assertTrue(torch.equal(output[window, token, 1:], torch.zeros(7)))
        self.assertTrue(torch.equal(output[2], torch.zeros_like(output[2])))
        changed = values.clone(); changed[0, 0, 0] = 30.0
        changed_output = model(changed)
        self.assertTrue(torch.equal(output[0, 1], changed_output[0, 1]))
        self.assertTrue(torch.equal(output[1], changed_output[1]))


if __name__ == "__main__":
    unittest.main()
