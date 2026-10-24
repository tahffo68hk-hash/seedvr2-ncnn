import unittest

import torch

from full_attention_reference import make_padding_mask
from rms_attention import RMSPerHeadBranchRewrite
from rope3d_reference import host_rope_cos_sin


class Phase82ReferenceStageTests(unittest.TestCase):
    def test_exports_complete_attention_stage_contract(self):
        torch.manual_seed(20260805)
        model = RMSPerHeadBranchRewrite().eval()
        x = torch.randn(1, 4, 8, 8, 32)
        mask = make_padding_mask(4, 64, valid_tokens=60)
        cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
        with torch.no_grad():
            stages = model.forward_stages(x, mask, cos, sin)

        expected_shapes = {
            "qkv_raw": (4, 64, 96),
            "head0_q_raw": (4, 64, 8),
            "head0_k_raw": (4, 64, 8),
            "head0_v_raw": (4, 64, 8),
            "head0_q_rms": (4, 64, 8),
            "head0_k_rms": (4, 64, 8),
            "head0_q_rope": (4, 64, 8),
            "head0_k_rope": (4, 64, 8),
            "head0_logits": (4, 64, 64),
            "head0_masked_logits": (4, 64, 64),
            "head0_probabilities": (4, 64, 64),
            "head0_av": (4, 64, 8),
            "merged": (4, 64, 32),
            "projection_input": (256, 32),
            "projection_output": (4, 64, 32),
            "output": (1, 4, 8, 8, 32),
        }
        for name, shape in expected_shapes.items():
            self.assertIn(name, stages)
            self.assertEqual(tuple(stages[name].shape), shape)
            self.assertEqual(stages[name].dtype, torch.float32)
            self.assertTrue(torch.isfinite(stages[name]).all())


if __name__ == "__main__":
    unittest.main()
