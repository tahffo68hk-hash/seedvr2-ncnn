"""Export the static per-head full fidelity attention core."""

from __future__ import annotations

from pathlib import Path

import torch

from attention_fidelity import partition_bthwc
from full_attention_reference import FullAttentionReference, make_padding_mask
from full_attention_per_head_rewrite import PerHeadBranchRewrite
from rope3d_reference import host_rope_cos_sin


ROOT = Path(__file__).resolve().parent


def write_f32(value: torch.Tensor, path: Path) -> None:
    value.detach().cpu().contiguous().float().numpy().tofile(path)


def export_per_head_attention() -> Path:
    root = ROOT / "artifacts" / "attention_core_fidelity" / "core_fidelity_ln_rope444_grid_1x2x2_perhead"
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260803)
    reference, core = FullAttentionReference().eval(), PerHeadBranchRewrite().eval()
    core.load_state_dict(reference.state_dict())
    x = torch.randn(1, 4, 8, 8, 32)
    windows, _ = partition_bthwc(x, (4, 4, 4))
    mask = make_padding_mask(4, 64, valid_tokens=60)
    cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
    with torch.no_grad():
        expected = reference.forward_stages(x, mask, cos, sin)
        actual = core(windows, mask, cos, sin)
    if not torch.allclose(expected["windows"], actual, atol=1e-6, rtol=1e-5):
        raise RuntimeError("per-head core differs from reference")
    torch.jit.trace(core, (windows, mask, cos, sin), check_trace=True).save(str(root / "model.pt"))
    for name, value in (("reference_input", x), ("windows", windows), ("reference_mask", mask),
                        ("reference_rope_cos", cos), ("reference_rope_sin", sin),
                        ("reference_windows", expected["windows"]), ("reference_output", expected["output"])):
        write_f32(value, root / f"{name}.bin")
    (root / "metadata.json").write_text(
        '{\n  "name": "core_fidelity_ln_rope444_grid_1x2x2_perhead",\n  "window": [4, 4, 4],\n  "grid": [1, 2, 2],\n  "padding_rule": "right-edge-to-window-multiple",\n  "tokens_per_window": 64,\n  "input_shape": [1, 4, 8, 8, 32],\n  "dtype": "fp32",\n  "ncnn_param": "model.ncnn.param",\n  "ncnn_bin": "model.ncnn.bin",\n  "variant": "fidelity_ln_rope3d_perhead",\n  "features": {"qk_norm": "layernorm", "rope3d": "host_tensor", "head_execution": "static_per_head", "masked_attention": true, "output_projection": true},\n  "inputs": ["input", "mask", "rope_cos", "rope_sin"]\n}\n', encoding="utf-8")
    return root


if __name__ == "__main__":
    print(export_per_head_attention())
