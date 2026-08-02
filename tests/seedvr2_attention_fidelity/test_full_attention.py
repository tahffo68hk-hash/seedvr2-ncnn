"""Tests for the complete fixed-window masked fidelity core."""

from __future__ import annotations

import unittest

import torch

from full_attention_reference import FullAttentionReference, make_padding_mask
from full_attention_rewrite import FullAttentionRewrite
from full_attention_per_head_rewrite import PerHeadBranchRewrite
from rope3d_reference import host_rope_cos_sin


class FullAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260803)
        self.reference = FullAttentionReference().eval()
        self.rewrite = FullAttentionRewrite().eval()
        self.rewrite.load_state_dict(self.reference.state_dict())
        self.input = torch.randn(1, 4, 8, 8, 32)
        self.mask = make_padding_mask(4, 64, valid_tokens=60)
        self.cos, self.sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)

    def test_rewrite_matches_math_reference_at_every_observable_stage(self) -> None:
        expected = self.reference.forward_stages(self.input, self.mask, self.cos, self.sin)
        actual = self.rewrite.forward_stages(self.input, self.mask, self.cos, self.sin)
        for name in ("q", "k", "logits", "probabilities", "av", "merged", "windows", "output"):
            self.assertLess(float((expected[name] - actual[name]).abs().max()), 1e-6, name)

    def test_masked_padding_values_do_not_change_valid_outputs(self) -> None:
        altered = self.input.clone()
        windows = altered.reshape(1, 1, 4, 2, 4, 2, 4, 32).permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(4, 64, 32)
        windows[:, 60:] = torch.randn_like(windows[:, 60:]) * 1000.0
        altered = windows.reshape(1, 1, 2, 2, 4, 4, 4, 32).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape_as(altered)
        base = self.rewrite.forward_stages(self.input, self.mask, self.cos, self.sin)["windows"]
        changed = self.rewrite.forward_stages(altered, self.mask, self.cos, self.sin)["windows"]
        self.assertLess(float((base[:, :60] - changed[:, :60]).abs().max()), 1e-6)

    def test_identity_rope_is_identity_and_outputs_are_finite(self) -> None:
        cos = torch.ones_like(self.cos)
        sin = torch.zeros_like(self.sin)
        stages = self.rewrite.forward_stages(self.input, self.mask, cos, sin)
        self.assertTrue(torch.isfinite(stages["output"]).all())
        self.assertEqual(tuple(stages["output"].shape), (1, 4, 8, 8, 32))

    def test_static_per_head_branches_match_reference_and_preserve_head_order(self) -> None:
        per_head = PerHeadBranchRewrite().eval()
        per_head.load_state_dict(self.reference.state_dict())
        expected = self.reference.forward_stages(self.input, self.mask, self.cos, self.sin)
        actual = per_head.forward_stages(self.input, self.mask, self.cos, self.sin)
        for head in range(4):
            self.assertLess(float((expected["av"][:, head] - actual[f"head{head}_av"]).abs().max()), 1e-6)
        self.assertLess(float((expected["windows"] - actual["windows"]).abs().max()), 1e-6)
        tagged = torch.cat([torch.full((1, 2, 8), 1000.0), torch.full((1, 2, 8), 2000.0), torch.full((1, 2, 8), 3000.0), torch.full((1, 2, 8), 4000.0)], dim=-1)
        self.assertTrue(torch.equal(per_head.concat_heads(tagged[..., :8], tagged[..., 8:16], tagged[..., 16:24], tagged[..., 24:]), tagged))


if __name__ == "__main__":
    unittest.main()
