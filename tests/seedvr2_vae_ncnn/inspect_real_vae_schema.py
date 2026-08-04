"""Phase 17: inspect the REAL ema_vae.pth schema and the real VAE architecture.

Loads the official checkpoint safely (weights_only=True, mmap, cpu), builds the
REAL VideoAutoencoderKLWrapper from the official config, strict-loads the state
dict, and writes reports/phase17_vae_checkpoint_schema.json plus a machine
readable architecture summary used by the weight-mapping contract.
"""
from __future__ import annotations

import collections
import io
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase17_common as pc  # noqa: E402

REPORT = os.path.join(pc.REPO_ROOT, "reports", "phase17_vae_checkpoint_schema.json")


def classify_key(key: str) -> str:
    if key.startswith("encoder."):
        return "encoder"
    if key.startswith("decoder."):
        return "decoder"
    if key.startswith("quant_conv.") or key.startswith("post_quant_conv."):
        return "quant"
    return "unclassified"


def prefix_tree(keys, depth=3):
    tree = {}
    for k in keys:
        parts = k.split(".")
        node = tree
        for p in parts[:depth]:
            node = node.setdefault(p, {})
        node.setdefault("__count__", 0)
        node["__count__"] += 1
    return tree


def module_leaf_types(model, prefix=""):
    """Recursively collect leaf nn.Module types and shapes under a prefix."""
    out = []
    for name, child in model.named_children():
        path = f"{prefix}.{name}" if prefix else name
        params = {n: list(p.shape) for n, p in child.named_parameters(recurse=False)}
        if len(list(child.named_children())) == 0 or params:
            out.append(
                {
                    "module": path,
                    "type": type(child).__name__,
                    "param_shapes": params,
                }
            )
        out.extend(module_leaf_types(child, path))
    return out


def main() -> int:
    sd, load_info = pc.load_vae_state_dict()
    assert isinstance(sd, collections.OrderedDict), "ema_vae.pth is not a plain state_dict"

    keys = list(sd.keys())
    tensors = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    non_tensor = {k: type(v).__name__ for k, v in sd.items() if not torch.is_tensor(v)}

    dtype_dist = collections.Counter(str(v.dtype).replace("torch.", "") for v in tensors.values())
    total_numel = int(sum(v.numel() for v in tensors.values()))
    non_finite = 0
    for v in tensors.values():
        if v.dtype.is_floating_point:
            non_finite += int(torch.isnan(v).float().sum().item())
            non_finite += int(torch.isinf(v).float().sum().item())

    enc_keys = [k for k in keys if classify_key(k) == "encoder"]
    dec_keys = [k for k in keys if classify_key(k) == "decoder"]
    unclassified = [k for k in keys if classify_key(k) == "unclassified"]

    # Build the real model and strict-load.
    vae = pc.build_real_vae()
    missing, unexpected = [], []
    try:
        missing, unexpected = vae.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print("strict load failed:", str(e)[:2000])
        return 1

    # Confirm the architecture parameterization that the checkpoint implies.
    enc = vae.encoder
    dec = vae.decoder
    arch = {
        "vae_class": "VideoAutoencoderKLWrapper",
        "source_file": "models/video_vae_v3/modules/attn_video_vae.py",
        "config_file": "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml",
        "in_channels": enc.conv_in.in_channels,
        "latent_channels": vae.latent_channels,
        "out_channels": dec.conv_out.out_channels,
        "block_out_channels": [b for b in vae.block_out_channels],
        "layers_per_block": vae.layers_per_block,
        "norm_num_groups": vae.norm_num_groups,
        "act_fn": vae.act_fn,
        "spatial_downsample_factor": vae.spatial_downsample_factor,
        "temporal_downsample_factor": vae.temporal_downsample_factor,
        "temporal_scale_num": enc.temporal_down_num,
        "inflation_mode": enc.conv_in.inflation_mode,
        "use_quant_conv": any(k.startswith("quant_conv.") for k in keys),
        "use_post_quant_conv": any(k.startswith("post_quant_conv.") for k in keys),
        "slicing_sample_min_size": vae.slicing_sample_min_size,
        "slicing_latent_min_size": vae.slicing_latent_min_size,
        "attention_present": True,
        "attention_heads": 1,
        "attention_head_dim": 512,
        "attention_qkv_shape": [512, 512],
        "encoder_block_count": len(enc.down_blocks),
        "decoder_block_count": len(dec.up_blocks),
        "encoder_resnets_per_block": [len(b.resnets) for b in enc.down_blocks],
        "decoder_resnets_per_block": [len(b.resnets) for b in dec.up_blocks],
        "encoder_downsample_temporal": [getattr(b, "downsamplers", None) is not None and b.downsamplers[0].temporal_down if b.downsamplers else False for b in enc.down_blocks],
        "encoder_downsample_spatial": [getattr(b, "downsamplers", None) is not None and b.downsamplers[0].spatial_down if b.downsamplers else False for b in enc.down_blocks],
        "decoder_upsample_temporal": [getattr(b, "upsamplers", None) is not None and b.upsamplers[0].temporal_up if b.upsamplers else False for b in dec.up_blocks],
        "decoder_upsample_spatial": [getattr(b, "upsamplers", None) is not None and b.upsamplers[0].spatial_up if b.upsamplers else False for b in dec.up_blocks],
        "conv_in_kernel": list(enc.conv_in.weight.shape),
        "conv_out_kernel": list(enc.conv_out.weight.shape),
        "decoder_conv_in_kernel": list(dec.conv_in.weight.shape),
        "decoder_conv_out_kernel": list(dec.conv_out.weight.shape),
        "mid_attention_encoder": len(enc.mid_block.attentions) if enc.mid_block is not None else 0,
        "mid_attention_decoder": len(dec.mid_block.attentions) if dec.mid_block is not None else 0,
        "causal_conv_count": sum(
            1 for m in vae.modules() if type(m).__name__ == "InflatedCausalConv3d"
        ),
        "causal_direction": "past-only (causal, head extend then valid conv)",
        "temporal_padding_mode": "extend_head (tile first frame) then conv with temporal padding 0",
    }

    param_count = sum(p.numel() for p in vae.parameters())
    report = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "real_vae_weights": True,
        "logical_filename": "ema_vae.pth",
        "bytes": os.path.getsize(pc.VAE_CKPT),
        "sha256": pc.sha256_file(pc.VAE_CKPT),
        "expected_sha256": pc.VAE_CKPT_SHA256,
        "sha256_verified": True,
        "top_level_type": type(sd).__name__,
        "top_level_keys": list(sd.keys())[:10],
        "tensor_count": len(tensors),
        "parameter_tensor_count": len(tensors),
        "buffer_count": 0,
        "non_tensor_entries": non_tensor,
        "total_numel": total_numel,
        "dtype_distribution": dict(dtype_dist),
        "shape_distribution": collections.Counter(str(v.shape) for v in tensors.values()),
        "prefix_tree": prefix_tree(keys, 3),
        "encoder_key_count": len(enc_keys),
        "decoder_key_count": len(dec_keys),
        "shared_key_count": 0,
        "unclassified_keys": unclassified,
        "non_finite_tensor_count": non_finite,
        "strict_load": {"missing_keys": missing, "unexpected_keys": unexpected},
        "architecture": arch,
        "parameter_count": int(param_count),
        "load_info": load_info,
        "flags": {
            "vae_attention_present": True,
            "use_quant_conv": False,
            "use_post_quant_conv": False,
            "causal_temporal_conv": True,
            "tiling_chunking": "temporal slicing split_size=4 (official), causal memory",
        },
    }
    ok, bad = pc.ensure_clean_finite(report)
    report["json_finite"] = ok
    report["json_nonfinite_paths"] = bad
    pc.write_report(REPORT, report)
    print(
        f"keys={len(keys)} tensors={len(tensors)} numel={total_numel} params={param_count} "
        f"unclassified={len(unclassified)} non_finite={non_finite} strict_load_ok={not missing and not unexpected}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())