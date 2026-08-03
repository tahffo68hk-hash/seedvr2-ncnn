#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14: generate the REAL-weight (C=2560/H=20/HD=128) RW-A staged reference.

Runs the REAL `NaMMSRTransformerBlock` (shared_weights=True) with the REAL
checkpoint `blocks.16.*` weights (FP32 CPU eval/no_grad) for the minimal RW-A
profile (batch=1, vid [1,4,4], text length 2).  Captures the full Phase 12/13
stage set via the shared-weight capture, derives window/packing/merge
intermediates, and writes FP32 .bin references + manifest + block_state.pt +
AdaSingle real params to the git-ignored `.phase14_local_artifacts/` directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_ncnn"))

import phase14_common as p14  # noqa: E402
import probe_lib  # noqa: E402
from phase14_capture import install_phase14_capture  # noqa: E402
from generate_synthetic_dit_block_reference import (  # noqa: E402
    window_metadata, per_window_txt_len, stage_stats,
    STAGE_ALIAS, STAGE_ORDER, PRODUCERS, write_bins, _git_head,
)

RW_A = [(1, 4, 4, 2)]
C, HEADS, HD = p14.C, p14.HEADS, p14.HD


def derive_stages(cap, win, txt_shape):
    """q/k/v splits, window outputs and residuals (real-width slices)."""
    out = {}
    qkv_vid = cap.kept("qkv_vid").detach().cpu().contiguous().float()
    qkv_txt = cap.kept("qkv_txt").detach().cpu().contiguous().float()
    out["q_vid"] = qkv_vid[:, 0:C]
    out["k_vid"] = qkv_vid[:, C:2 * C]
    out["v_vid"] = qkv_vid[:, 2 * C:3 * C]
    out["q_txt"] = qkv_txt[:, 0:C]
    out["k_txt"] = qkv_txt[:, C:2 * C]
    out["v_txt"] = qkv_txt[:, 2 * C:3 * C]

    packed = cap.kept("attention_packed_output").detach().cpu().contiguous().float()
    sizes = win["window_sizes"]
    txt_lens = per_window_txt_len(win, txt_shape)
    seg_lens = [s + l for s, l in zip(sizes, txt_lens)]
    vid_wins, txt_wins = [], []
    off = 0
    for sl, vl in zip(seg_lens, sizes):
        vid_wins.append(packed[off:off + vl])
        txt_wins.append(packed[off + vl:off + sl])
        off += sl
    out["attention_vid_windows"] = torch.cat(vid_wins) if vid_wins else packed.new_zeros(0, HEADS, HD)
    out["attention_txt_windows"] = torch.cat(txt_wins) if txt_wins else packed.new_zeros(0, HEADS, HD)
    out["attention_txt_mean_merged"] = cap.kept("attn_merge_txt").detach().cpu().contiguous().float()
    out["attention_vid_merged"] = cap.kept("attn_merge_vid").detach().cpu().contiguous().float()

    in_vid = cap.kept("input_vid").detach().cpu().contiguous().float()
    in_txt = cap.kept("input_txt").detach().cpu().contiguous().float()
    gated_vid = cap.kept("ada_vid_attn_out_out").detach().cpu().contiguous().float()
    gated_txt = cap.kept("ada_txt_attn_out_out").detach().cpu().contiguous().float()
    out["attention_residual_vid"] = gated_vid + in_vid
    out["attention_residual_txt"] = gated_txt + in_txt
    return out


def run_profile(block, cap, name):
    inputs = probe_lib.make_scaled_inputs(RW_A, vid_dim=C, seed=p14.EXPECTED_MAIN_SHA256 and 20260803)
    probe_lib.clear_rope_cache(block)
    with torch.no_grad():
        vid_out, txt_out, vs, ts = probe_lib.run_forward(block, cap, name, inputs, keep_mode=True)
    win = window_metadata(inputs["vid_shape"])
    txt_len_win = per_window_txt_len(win, inputs["txt_shape"])
    seg_lens = [s + l for s, l in zip(win["window_sizes"], txt_len_win)]
    cu_q = cap._kept["cu_seqlens_q"][-1]
    return {
        "inputs": inputs, "vid_out": vid_out, "txt_out": txt_out, "win": win,
        "seg_lens": seg_lens, "cu_seqlens_q": cu_q.tolist(),
        "cu_seqlens_k": cap._kept["cu_seqlens_k"][-1].tolist(),
        "max_segment_q": int(cap._kept["max_seqlen_q"][-1]),
        "total_packed": int(cu_q[-1].item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=p14.LOCAL_ROOT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    probe_lib.ensure_compat()
    torch.Tensor.bfloat16 = lambda self: self  # FP32 attention mode (documented)

    from models.dit_v2.nablocks.mmsr_block import NaMMSRTransformerBlock
    import inspect

    provenance = {
        "phase": "phase14",
        "seedvr2_commit": p14.SEEDVR2_HEAD,
        "checkpoint_sha256": p14.EXPECTED_MAIN_SHA256,
        "real_seedvr2_weights": True,
        "block_index": 16,
        "shared_weights": True,
        "block_class": NaMMSRTransformerBlock.__name__,
        "block_module": NaMMSRTransformerBlock.__module__,
        "block_source": inspect.getsourcefile(NaMMSRTransformerBlock),
        "torch_version": torch.__version__,
        "real_flash_attn_cuda_kernel": False,
        "real_apex_cuda_kernel": False,
        "attention_dtype_mode": "fp32 (torch.Tensor.bfloat16 patched to identity in test harness)",
        "flash_attn": "cpu_compat (math-equivalent FP32 varlen)",
        "apex": "cpu_compat (FusedRMSNorm FP32)",
        "pnnx": "fp16=0 (see export_real_weight_block_stages.py)",
    }

    block = probe_lib.build_block(dict(p14.REAL_CFG), seed=0).eval()
    sd_all = torch.load(p14.MAIN_CKPT, map_location="cpu", weights_only=True, mmap=True)
    prefix = p14.BLOCK_PREFIX
    sd = {k[len(prefix):]: v for k, v in sd_all.items() if k.startswith(prefix)}
    missing, unexpected = block.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    block.eval()

    cap = probe_lib.StageCapture()
    install_phase14_capture(block, cap, capture_txt=True)

    info = run_profile(block, cap, "A")
    derived = derive_stages(cap, info["win"], info["inputs"]["txt_shape"])
    manifest = write_bins(cap, derived, out / "reference" / "A", "A")
    manifest["windows"] = {
        "window_shape": info["win"]["window_shape"],
        "window_count": info["win"]["window_count"],
        "window_sizes": info["win"]["window_sizes"],
        "partition_index": info["win"]["partition_index"],
        "reverse_index": info["win"]["reverse_index"],
        "segment_lengths": info["seg_lens"],
        "cu_seqlens_q": info["cu_seqlens_q"],
        "cu_seqlens_k": info["cu_seqlens_k"],
        "max_segment": info["max_segment_q"],
        "total_packed": info["total_packed"],
    }
    manifest["inputs"] = {
        "vid_shape": info["inputs"]["vid_shape"].tolist(),
        "txt_shape": info["inputs"]["txt_shape"].tolist(),
        "vid_tokens": int(info["inputs"]["vid_shape"].prod(-1).sum().item()),
        "txt_tokens": int(info["inputs"]["txt_shape"].sum().item()),
        "batch": len(info["inputs"]["vid_shape"]),
    }
    (out / "reference" / "A" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"RW-A: windows={info['win']['window_sizes']} seg_lens={info['seg_lens']} max_seg={info['max_segment_q']}")

    # determinism RW-A -> RW-A
    first_a = (info["vid_out"], info["txt_out"])
    info_a2 = run_profile(block, cap, "A")
    det_ok = torch.equal(first_a[0], info_a2["vid_out"]) and torch.equal(first_a[1], info_a2["txt_out"])
    provenance["rwa_deterministic_bit_identical"] = bool(det_ok)

    torch.save({k: v.detach().cpu().clone() for k, v in block.state_dict().items()}, out / "block_state.pt")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # AdaSingle real params (runtime params dir, 12 bins from 6 .all. values).
    params = out / "params"
    params.mkdir(parents=True, exist_ok=True)
    bsd = dict(block.state_dict())
    for layer in ("attn", "mlp"):
        for part in ("shift", "scale", "gate"):
            t = bsd[f"ada.all.{layer}_{part}"].detach().cpu().contiguous().float()
            for side in ("vid", "txt"):
                t.numpy().tofile(params / f"ada_{side}_{layer}_{part}.bin")
    print("references written to", out)
    print("deterministic:", det_ok)
    return 0 if det_ok else 1


if __name__ == "__main__":
    sys.exit(main())
