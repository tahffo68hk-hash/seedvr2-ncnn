"""Export two independent ncnn stages sharing one deterministic reference model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from attention_fidelity import partition_bthwc
from attention_heads_core import AttentionHeadsCore
from full_attention_reference import FullAttentionReference, make_padding_mask
from host_head_merge import merge_heads
from output_projection_core import OutputProjectionCore
from rope3d_reference import host_rope_cos_sin


ROOT = Path(__file__).resolve().parent


def write(value: torch.Tensor, path: Path) -> None:
    value.detach().cpu().contiguous().float().numpy().tofile(path)


def main() -> Path:
    root = ROOT / "artifacts" / "attention_core_fidelity" / "hybrid_fidelity_ln_rope444_grid_1x2x2"
    heads_dir, projection_dir, reference_dir = root / "attention_heads", root / "output_projection", root / "reference"
    for directory in (heads_dir, projection_dir, reference_dir): directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260803)
    reference = FullAttentionReference().eval()
    heads = AttentionHeadsCore().eval(); heads.load_state_dict(reference.state_dict())
    projection = OutputProjectionCore(reference.output).eval()
    x = torch.randn(1, 4, 8, 8, 32); windows, _ = partition_bthwc(x, (4, 4, 4))
    mask = make_padding_mask(4, 64, valid_tokens=60); cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
    with torch.no_grad():
        expected = reference.forward_stages(x, mask, cos, sin)
        head_values = heads(windows, mask, cos, sin)
        merged = merge_heads(head_values)
        flat_output = projection(merged.reshape(-1, 32))
    if not torch.allclose(flat_output.reshape_as(windows), expected["windows"], atol=1e-6, rtol=1e-5): raise RuntimeError("hybrid reference mismatch")
    torch.jit.trace(heads, (windows, mask, cos, sin), check_trace=True).save(str(heads_dir / "model.pt"))
    torch.jit.trace(projection, merged.reshape(-1, 32), check_trace=True).save(str(projection_dir / "model.pt"))
    for name, value in (("input", x), ("windows", windows), ("mask", mask), ("rope_cos", cos), ("rope_sin", sin), ("merged", merged), ("projection_2d", flat_output), ("windows_output", expected["windows"]), ("bthwc_output", expected["output"])): write(value, reference_dir / f"{name}.bin")
    for index, value in enumerate(head_values): write(value, reference_dir / f"head{index}.bin")
    (root / "metadata.json").write_text(json.dumps({"name":"hybrid_fidelity_ln_rope444_grid_1x2x2","variant":"fidelity_ln_rope3d_hybrid","execution":"staged_host_head_merge","window":[4,4,4],"grid":[1,2,2],"features":{"qk_norm":"layernorm","rope3d":"host_tensor","masked_attention":True,"head_merge":"host","output_projection":"ncnn_2d_gemm"},"models":{"attention_heads":{"ncnn_param":"attention_heads/model.ncnn.param","ncnn_bin":"attention_heads/model.ncnn.bin"},"output_projection":{"ncnn_param":"output_projection/model.ncnn.param","ncnn_bin":"output_projection/model.ncnn.bin"}},"inputs":["input","mask","rope_cos","rope_sin"]}, indent=2)+"\n", encoding="utf-8")
    return root


if __name__ == "__main__": print(main())
