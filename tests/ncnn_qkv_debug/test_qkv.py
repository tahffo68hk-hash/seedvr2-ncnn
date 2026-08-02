"""Break caught: QKV projection or head layout creates a wrong tensor shape."""

import unittest

import torch

from export_qkv import Linear2dOnly, LinearOnly, LinearReshape, LinearReshapeTranspose, LinearSplit


class QkvShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20250803)
        self.x = torch.randn((4, 32, 32), dtype=torch.float32)

    def test_linear_projects_32_channels_to_96(self) -> None:
        self.assertEqual(tuple(LinearOnly()(self.x).shape), (4, 32, 96))

    def test_two_dimensional_linear_projects_32_channels_to_96(self) -> None:
        self.assertEqual(tuple(Linear2dOnly()(self.x[0]).shape), (32, 96))

    def test_split_returns_three_32_channel_tensors(self) -> None:
        q, k, v = LinearSplit()(self.x)
        self.assertEqual((tuple(q.shape), tuple(k.shape), tuple(v.shape)), ((4, 32, 32),) * 3)

    def test_reshape_exposes_four_heads_of_eight_channels(self) -> None:
        q, k, v = LinearReshape()(self.x)
        self.assertEqual((tuple(q.shape), tuple(k.shape), tuple(v.shape)), ((4, 32, 4, 8),) * 3)

    def test_transpose_places_heads_before_tokens(self) -> None:
        q, k, v = LinearReshapeTranspose()(self.x)
        self.assertEqual((tuple(q.shape), tuple(k.shape), tuple(v.shape)), ((4, 4, 32, 8),) * 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
