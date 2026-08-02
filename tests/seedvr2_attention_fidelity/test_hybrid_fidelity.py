"""Tests for staged hybrid host merge semantics."""

from __future__ import annotations

import unittest

import torch

from host_head_merge import merge_heads


class HybridFidelityTests(unittest.TestCase):
    def test_single_attention_head_core_matches_selected_reference_head(self) -> None:
        from attention_heads_core import SingleAttentionHeadCore
        from full_attention_reference import FullAttentionReference, make_padding_mask
        from rope3d_reference import host_rope_cos_sin

        torch.manual_seed(7)
        reference = FullAttentionReference().eval()
        core = SingleAttentionHeadCore(reference, 2).eval()
        windows = torch.randn(64, 32)
        mask = make_padding_mask(1, 64, valid_tokens=60)[0]
        cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
        with torch.no_grad():
            actual = core(windows, mask, cos, sin)
            from full_attention_per_head_rewrite import PerHeadBranchRewrite
            rewrite = PerHeadBranchRewrite().eval()
            rewrite.load_state_dict(reference.state_dict())
            qkv = reference.qkv(windows.reshape(-1, 32)).reshape(1, 64, 96)
            q, k, v = qkv.split(32, dim=-1)
            expected = rewrite._branch(q[..., 16:24], k[..., 16:24], v[..., 16:24], mask, cos, sin)[4]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_host_merge_uses_head_major_channel_order(self) -> None:
        heads = [torch.full((2, 3, 2), 1000.0 * (index + 1)) for index in range(4)]
        merged = merge_heads(heads)
        self.assertEqual(tuple(merged.shape), (2, 3, 8))
        self.assertTrue(torch.equal(merged[0, 0], torch.tensor([1000., 1000., 2000., 2000., 3000., 3000., 4000., 4000.])))

    def test_host_merge_rejects_inconsistent_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            merge_heads([torch.zeros(1, 2, 8), torch.zeros(1, 3, 8), torch.zeros(1, 2, 8), torch.zeros(1, 2, 8)])


if __name__ == "__main__":
    unittest.main()
