"""Export deterministic complete masked attention core and reference tensors."""

from __future__ import annotations

from pathlib import Path

import torch

from attention_fidelity import partition_bthwc
from full_attention_reference import FullAttentionReference, make_padding_mask
from full_attention_rewrite import FullAttentionRewrite
from rope3d_reference import host_rope_cos_sin


ROOT = Path(__file__).resolve().parent


def write_f32(value: torch.Tensor, path: Path) -> None:
    value.detach().cpu().contiguous().float().numpy().tofile(path)


def export_full_attention() -> Path:
    artifacts = ROOT / "artifacts" / "attention_core_fidelity" / "core_fidelity_ln_rope444_grid_1x2x2"
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260803)
    reference = FullAttentionReference().eval()
    rewrite = FullAttentionRewrite().eval()
    rewrite.load_state_dict(reference.state_dict())
    x = torch.randn(1, 4, 8, 8, 32)
    windows, _ = partition_bthwc(x, (4, 4, 4))
    mask = make_padding_mask(4, 64, valid_tokens=60)
    cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
    with torch.no_grad():
        expected = reference.forward_stages(x, mask, cos, sin)
        core_output = rewrite(windows, mask, cos, sin)
    if not torch.allclose(expected["windows"], core_output, atol=1e-6, rtol=1e-5):
        raise RuntimeError("core rewrite differs from full reference windows")
    torch.jit.trace(rewrite, (windows, mask, cos, sin), check_trace=True).save(str(artifacts / "model.pt"))
    write_f32(x, artifacts / "reference_input.bin")
    write_f32(windows, artifacts / "windows.bin")
    write_f32(mask, artifacts / "reference_mask.bin")
    write_f32(cos, artifacts / "reference_rope_cos.bin")
    write_f32(sin, artifacts / "reference_rope_sin.bin")
    write_f32(expected["windows"], artifacts / "reference_windows.bin")
    write_f32(expected["output"], artifacts / "reference_output.bin")
    (artifacts / "metadata.json").write_text(
        "{\n"
        "  \"name\": \"core_fidelity_ln_rope444_grid_1x2x2\",\n"
        "  \"window\": [4, 4, 4],\n"
        "  \"grid\": [1, 2, 2],\n"
        "  \"padding_rule\": \"right-edge-to-window-multiple\",\n"
        "  \"tokens_per_window\": 64,\n"
        "  \"input_shape\": [1, 4, 8, 8, 32],\n"
        "  \"dtype\": \"fp32\",\n"
        "  \"ncnn_param\": \"model.ncnn.param\",\n"
        "  \"ncnn_bin\": \"model.ncnn.bin\",\n"
        "  \"features\": {\"qk_norm\": \"layernorm\", \"rope3d\": \"host_tensor\", \"masked_attention\": true, \"output_projection\": true},\n"
        "  \"inputs\": [\"input\", \"mask\", \"rope_cos\", \"rope_sin\"]\n"
        "}\n", encoding="utf-8")
    return artifacts


if __name__ == "__main__":
    print(export_full_attention())
