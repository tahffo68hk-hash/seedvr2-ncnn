"""Export four 2-D-input static ncnn attention-head cores for the hybrid runtime."""

from pathlib import Path

import torch

from attention_heads_core import SingleAttentionHeadCore
from full_attention_reference import FullAttentionReference, make_padding_mask
from rope3d_reference import host_rope_cos_sin


def main() -> Path:
    root = Path(__file__).resolve().parent / "artifacts" / "attention_core_fidelity" / "hybrid_fidelity_ln_rope444_grid_1x2x2" / "attention_heads"
    torch.manual_seed(20260803)
    reference = FullAttentionReference().eval()
    windows = torch.randn(64, 32)
    mask = make_padding_mask(1, 64, valid_tokens=60)[0]
    cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
    for head_index in range(4):
        directory = root / f"head{head_index}"
        directory.mkdir(parents=True, exist_ok=True)
        core = SingleAttentionHeadCore(reference, head_index).eval()
        torch.jit.trace(core, (windows, mask, cos, sin), check_trace=True).save(str(directory / "model.pt"))
    return root


if __name__ == "__main__":
    print(main())
