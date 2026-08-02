"""Export the Phase 8 RMSNorm staged hybrid artifact."""
from pathlib import Path
import json
import torch
from attention_fidelity import partition_bthwc
from full_attention_reference import make_padding_mask
from output_projection_core import OutputProjectionCore
from rms_attention import RMSPerHeadBranchRewrite, SingleRMSAttentionHead
from rope3d_reference import host_rope_cos_sin

ROOT = Path(__file__).resolve().parent

def write(value, path): value.detach().cpu().contiguous().float().numpy().tofile(path)

def main():
    torch.manual_seed(20260805)
    root = ROOT / "artifacts" / "attention_core_fidelity" / "hybrid_fidelity_rms_rope444_grid_1x2x2"
    heads_root, projection_root, reference_root = root / "attention_heads", root / "output_projection", root / "reference"
    for p in (heads_root, projection_root, reference_root): p.mkdir(parents=True, exist_ok=True)
    reference = RMSPerHeadBranchRewrite().eval()
    x = torch.randn(1, 4, 8, 8, 32)
    windows, _ = partition_bthwc(x, (4, 4, 4))
    mask = make_padding_mask(4, 64, valid_tokens=60)
    cos, sin = host_rope_cos_sin(4, 4, 4, head_dim=8, rotary_dim=6)
    with torch.no_grad(): stages = reference.forward_stages(x, mask, cos, sin)
    for i in range(4):
        head = SingleRMSAttentionHead(reference, i).eval()
        directory = heads_root / f"head{i}"; directory.mkdir(exist_ok=True)
        torch.jit.trace(head, (windows[0], mask[0], cos, sin), check_trace=True).save(str(directory / "model.pt"))
                # Host merge is the concatenation of per-head AV tensors.
    merged = torch.cat(tuple(stages[f"head{i}_av"] for i in range(4)), dim=-1)
    torch.jit.trace(OutputProjectionCore(reference.output).eval(), (merged.reshape(-1, 32),), check_trace=True).save(str(projection_root / "model.pt"))
    for name, value in (("input", x), ("windows", windows), ("mask", mask), ("rope_cos", cos), ("rope_sin", sin), ("head0", stages["head0_av"]), ("head1", stages["head1_av"]), ("head2", stages["head2_av"]), ("head3", stages["head3_av"]), ("merged", merged), ("projection_2d", reference.output(merged.reshape(-1, 32))), ("windows_output", stages["windows"]), ("bthwc_output", stages["output"])): write(value, reference_root / f"{name}.bin")
    (root / "metadata.json").write_text(json.dumps({"name":"hybrid_fidelity_rms_rope444_grid_1x2x2","variant":"fidelity_rms_rope3d_hybrid","execution":"staged_host_head_merge","window":[4,4,4],"grid":[1,2,2],"padding_rule":"right-edge-to-window-multiple","tokens_per_window":64,"input_shape":[1,4,8,8,32],"dtype":"fp32","features":{"qk_norm":"rmsnorm","qk_norm_reference":"mathematical_unfused","rmsnorm_rewrite":"explicit_sqrt_reciprocal","pnnx_fp16":False,"rope3d":"artifact_reference","masked_attention":True,"head_merge":"host","output_projection":"ncnn_2d_gemm"},"models":{"heads":[{"name":f"head{i}","ncnn_param":f"attention_heads/head{i}/model.ncnn.param","ncnn_bin":f"attention_heads/head{i}/model.ncnn.bin"} for i in range(4)],"output_projection":{"name":"output_projection","ncnn_param":"output_projection/model.ncnn.param","ncnn_bin":"output_projection/model.ncnn.bin"}}}, indent=2)+"\n", encoding="utf-8")
    return root
if __name__ == "__main__": print(main())
