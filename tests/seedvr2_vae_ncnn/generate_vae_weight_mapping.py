#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: VAE state-dict -> stage weight mapping contract (generate).

Every ema_vae.pth key is assigned to exactly one FP32 ncnn export stage.
All weights are consumed through the graph-equivalent export wrappers and
baked into the stage ncnn bins by pnnx (fp16=0); GroupNorm weights are
repeated over T by the GNAlt2 fold and upsampler weights are re-arranged
into the combined ConvTranspose3d layout, so those keys are marked with the
corresponding conversion.  Runtime-only params (latent scale/shift) are
recorded separately, they are not state-dict keys.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402
import vae_export_model as vem  # noqa: E402

STAGE_PREFIXES = {
    "enc_conv_in": ("encoder.conv_in",),
    "enc_down0": ("encoder.down_blocks.0",),
    "enc_down1": ("encoder.down_blocks.1",),
    "enc_down2": ("encoder.down_blocks.2",),
    "enc_down3": ("encoder.down_blocks.3",),
    "enc_mid": ("encoder.mid_block",),
    "enc_out": ("encoder.conv_norm_out", "encoder.conv_out"),
    "dec_conv_in": ("decoder.conv_in",),
    "dec_mid": ("decoder.mid_block",),
    "dec_up0": ("decoder.up_blocks.0",),
    "dec_up1": ("decoder.up_blocks.1",),
    "dec_up2": ("decoder.up_blocks.2",),
    "dec_up3": ("decoder.up_blocks.3",),
    "dec_out": ("decoder.conv_norm_out", "decoder.conv_out"),
}


def semantic_role(key: str) -> str:
    parts = key.split(".")
    tail = parts[-1]  # weight | bias
    kind = "bias" if tail == "bias" else "weight"
    # find the parameter container name
    stem = [p for p in parts if p not in ("encoder", "decoder", "weight", "bias")]
    container = stem[-1] if stem else "?"
    if container.startswith("to_q"):
        role = "attention_q"
    elif container.startswith("to_k"):
        role = "attention_k"
    elif container.startswith("to_v"):
        role = "attention_v"
    elif container.startswith("to_out"):
        role = "attention_out"
    elif container in ("norm1", "norm2", "conv_norm_out", "group_norm"):
        role = "groupnorm"
    elif container.startswith("conv") or container == "conv":
        role = "conv3d"
    elif container == "upscale_conv":
        role = "upsample_shuffle_conv"
    else:
        role = container
    return f"{role}_{kind}"


def conversion_of(key: str) -> str:
    parts = key.split(".")
    if "upsamplers" in parts and "upscale_conv" in parts:
        # upscale 1x1 conv weight/bias folded into the combined ConvTranspose3d
        return "deconv_weight_layout"
    if parts[-1] in ("norm1", "norm2", "conv_norm_out") or "group_norm" in parts:
        # GNAlt2: weight/bias repeated over the time axis inside the wrapper
        return "reshape"
    return "pnnx_managed"


def main() -> int:
    vae = pc.build_real_vae()
    sd, load_info = pc.load_vae_state_dict()
    missing, unexpected = vae.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    enc = vae.encoder
    dec = vae.decoder
    enc_stages = vem.build_encoder_stages(enc)
    dec_stages = vem.build_decoder_stages(dec)
    all_stages = {**enc_stages, **dec_stages}

    stage_of = {}
    for stage, prefixes in STAGE_PREFIXES.items():
        for pfx in prefixes:
            for key in sd.keys():
                if key.startswith(pfx + "."):
                    assert stage not in stage_of.get(key, {}), ("duplicate", key)
                    stage_of[key] = stage

    # shape expectations: identity copies keep the wrapper parameter shape;
    # for the deconv-folded upscale weights we compare numel only.
    expected_shape = {}
    for key, stage in stage_of.items():
        mod = all_stages[stage]
        parts = key.split(".")
        tail = parts[-1]
        stem = [p for p in parts if p not in ("encoder", "decoder", "weight", "bias")]
        container = stem[-1] if stem else "?"
        if "upsamplers" in parts and "upscale_conv" in parts:
            expected_shape[key] = ("numel",)
            continue
        target = None
        if container.startswith("to_q"):
            target = mod.attns[0].to_q
        elif container.startswith("to_k"):
            target = mod.attns[0].to_k
        elif container.startswith("to_v"):
            target = mod.attns[0].to_v
        elif container.startswith("to_out"):
            target = mod.attns[0].to_out0
        elif container in ("norm1", "norm2"):
            target = mod.resnets[int(stem[-2])].norm1.gn if container == "norm1" else mod.resnets[int(stem[-2])].norm2.gn
        elif container == "group_norm":
            target = mod.attns[0].group_norm.gn
        elif container in ("conv1", "conv2", "conv_shortcut"):
            r = mod.resnets[int(stem[-2])]
            target = getattr(r, {"conv1": "conv1", "conv2": "conv2", "conv_shortcut": "shortcut"}[container], None)
        elif container == "conv_in":
            # enc_conv_in / dec_conv_in stages ARE the ExportCausalConv3d
            target = mod
        elif container == "conv_out":
            target = mod.conv_out
        elif container == "conv_norm_out":
            target = mod.norm.gn
        elif container == "conv" and "upsamplers" in parts:
            target = mod.up.conv
        if target is not None:
            p = {"weight": "weight", "bias": "bias"}[tail]
            expected_shape[key] = tuple(int(s) for s in getattr(target, p).shape)
        else:
            expected_shape[key] = ("numel",)

    entries = {}
    total_numel = 0
    for key, t in sd.items():
        stage = stage_of.get(key)
        if stage is None:
            continue
        shape = tuple(int(s) for s in t.shape)
        numel = int(t.numel())
        total_numel += numel
        entries[key] = {
            "state_dict_key": key,
            "source_module": ".".join(key.split(".")[:-1]),
            "semantic_role": semantic_role(key),
            "encoder_or_decoder": "encoder" if key.startswith("encoder") else "decoder",
            "block_index": -1,
            "stage_index": -1,
            "shape": list(shape),
            "numel": numel,
            "dtype": str(t.dtype).replace("torch.", ""),
            "kind": "weight" if key.endswith(".weight") else "bias",
            "target_stage": stage,
            "target_artifact": f".phase17_local_artifacts/stages_<A|B>/{stage}/model.ncnn.bin",
            "target_blob": "pnnx-managed",
            "conversion": conversion_of(key),
            "transpose": "none",
            "reshape": "none",
            "split": "none",
            "concat": "none",
            "padding_semantics": "causal (extend_head first-frame repeat)" if "conv" in key and not key.endswith(".bias") else "none",
            "expected_target_shape": list(expected_shape[key]) if isinstance(expected_shape[key], tuple) and expected_shape[key] != ("numel",) else "numel-compare",
            "shared_or_independent": "independent",
            "runtime_only": False,
        }

    unmapped = [k for k in sd if k not in entries]
    dup = sorted({k for k in entries if sum(1 for s in STAGE_PREFIXES for p in STAGE_PREFIXES[s] if k.startswith(p + ".")) > 1})

    report = {
        "phase": 17,
        "real_seedvr2_weights": True,
        "checkpoint_sha256": pc.VAE_CKPT_SHA256,
        "total_state_dict_keys": len(sd),
        "mapped_keys": len(entries),
        "mapped_numel": total_numel,
        "source_numel": sum(int(t.numel()) for t in sd.values()),
        "unmapped_keys": unmapped,
        "duplicate_keys": dup,
        "conversion_summary": {},
        "stages": {},
        "runtime_only": {
            "latent_scaling_factor": pc.VAE_RUNNER_CFG["scaling_factor"],
            "latent_shift": pc.VAE_RUNNER_CFG["shifting_factor"],
            "use_sample": pc.VAE_RUNNER_CFG["use_sample"],
            "note": "handled by the host runtime, not state-dict keys",
        },
        "load_info": load_info,
    }
    for conv in sorted({e["conversion"] for e in entries.values()}):
        report["conversion_summary"][conv] = sum(1 for e in entries.values() if e["conversion"] == conv)
    for stage in all_stages:
        keys = [k for k, e in entries.items() if e["target_stage"] == stage]
        n = sum(e["numel"] for e in entries.values() if e["target_stage"] == stage)
        report["stages"][stage] = {"key_count": len(keys), "source_numel": n,
                                   "keys": sorted(keys)}
    report["entries"] = entries
    report["mapping_pass"] = (len(entries) == len(sd) and not unmapped and not dup
                              and report["mapped_numel"] == report["source_numel"])
    pc.write_report(str(HERE.parent.parent / "reports" / "phase17_vae_weight_mapping.json"), report)
    print("mapped", len(entries), "/", len(sd), "pass", report["mapping_pass"])
    return 0 if report["mapping_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
