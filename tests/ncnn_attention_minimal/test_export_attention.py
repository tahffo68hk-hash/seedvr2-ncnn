"""Break caught: incorrect scale, transpose or softmax axis changes attention."""

import unittest

import torch

from export_attention import Attention, MatMulOnly, MatMulSoftmax


class MinimalAttentionTests(unittest.TestCase):
    def test_attention_matches_hand_derived_two_token_case(self) -> None:
        x = torch.zeros((1, 2, 8), dtype=torch.float32)
        x[0, 0, 0] = 1.0
        x[0, 1, 1] = 1.0
        expected_weight = torch.tensor(0.587479, dtype=torch.float32)
        scores = MatMulOnly()(x)
        probabilities = MatMulSoftmax()(x)
        output = Attention()(x)
        self.assertTrue(torch.allclose(scores[0, 0], torch.tensor([0.353553, 0.0]), atol=1e-5))
        self.assertTrue(torch.allclose(probabilities[0, 0], torch.tensor([expected_weight, 1 - expected_weight]), atol=1e-5))
        self.assertTrue(torch.allclose(output[0, 0, :2], torch.tensor([expected_weight, 1 - expected_weight]), atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
