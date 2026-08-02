"""Behavior tests for the fixed-shape migration fixture.

Break caught: replacing explicit QKV attention with identity/incorrect head
reshaping must change the hand-derived two-token reference output below.
"""

import unittest

import torch

from attention import ExplicitMultiHeadAttention
from model import AdaptiveWindowAttentionFixture, ExportAttentionCore, WindowAttentionCore


class ExplicitAttentionTests(unittest.TestCase):
    def test_identity_qkv_matches_hand_derived_two_token_attention(self) -> None:
        attention = ExplicitMultiHeadAttention(channels=8, num_heads=1, head_dim=8, bias=False)
        with torch.no_grad():
            eye = torch.eye(8)
            attention.wq.weight.copy_(eye)
            attention.wk.weight.copy_(eye)
            attention.wv.weight.copy_(eye)
            attention.wo.weight.copy_(eye)

        x = torch.zeros((1, 2, 8), dtype=torch.float32)
        x[0, 0, 0] = 1.0
        x[0, 1, 1] = 1.0
        output, details = attention(x, return_details=True)

        expected_weight = torch.tensor(0.587479, dtype=torch.float32)
        self.assertEqual(tuple(details["q"].shape), (1, 1, 2, 8))
        self.assertEqual(tuple(details["attention_flat"].shape), (1, 2, 2))
        self.assertEqual(tuple(details["attention"].shape), (1, 1, 2, 2))
        self.assertTrue(torch.allclose(details["attention"][0, 0, 0], torch.tensor([expected_weight, 1 - expected_weight]), atol=1e-5))
        self.assertTrue(torch.allclose(output[0, 0, :2], torch.tensor([expected_weight, 1 - expected_weight]), atol=1e-5))


class FixtureModelTests(unittest.TestCase):
    def test_core_preserves_window_tensor_shape(self) -> None:
        torch.manual_seed(11)
        core = WindowAttentionCore(channels=32, num_heads=4, head_dim=8)
        windows = torch.randn((4, 32, 32), dtype=torch.float32)
        output = core(windows)
        self.assertEqual(tuple(output.shape), (4, 32, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_unrolled_export_core_matches_reference_attention(self) -> None:
        torch.manual_seed(13)
        reference = WindowAttentionCore(channels=32, num_heads=4, head_dim=8).eval()
        export_core = ExportAttentionCore(channels=32, num_heads=4, head_dim=8).eval()
        export_core.load_state_dict(reference.state_dict())
        windows = torch.randn((4, 32, 32), dtype=torch.float32)
        self.assertTrue(torch.allclose(export_core(windows), reference(windows), atol=1e-6, rtol=1e-5))

    def test_model_partitions_attends_and_reverses_required_shape(self) -> None:
        torch.manual_seed(7)
        model = AdaptiveWindowAttentionFixture(channels=32, num_heads=4, head_dim=8, window_size=4)
        x = torch.randn((1, 2, 8, 8, 32), dtype=torch.float32)
        output, details = model(x, return_details=True)

        self.assertEqual(tuple(details["windows"].shape), (4, 32, 32))
        self.assertEqual(tuple(details["q"].shape), (4, 4, 32, 8))
        self.assertEqual(tuple(details["attention"].shape), (4, 4, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 2, 8, 8, 32))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
