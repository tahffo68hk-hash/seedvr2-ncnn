"""Phase 12: generate staged PyTorch references for the synthetic SeedVR2 DiT block.

Uses the REAL `NaMMSRTransformerBlock` (third_party/SeedVR2) at the scaled
C=512 / H=4 / HD=128 config with deterministic random weights (seed 20260802).
Only the fused-CUDA libraries (flash_attn / apex) are replaced by the Phase 11
CPU compat, and the attention dtype mode is pinned to FP32 by making
`torch.Tensor.bfloat16` a no-op (documented test-harness numerical-mode change;
the real flash-attn kernel is bf16 and its fidelity is NOT claimed).  The block
forward itself is never copied or rewritten.

For every profile (A/B/C/D) the script captures the full stage list required by
the Phase 12 task (section 8), derives window/packing/merge intermediates from
the captured tensors, and writes float32 `.bin` references plus a JSON manifest
with logical shape / dtype / numel / finite / mean / std / min / max / file /
producer / consumer for each stage.  The real block state dict and provenance
are also persisted for the ncnn exporter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))

import probe_lib  # noqa: E402
from phase12_capture import install_phase12_capture  # noqa: E402

SEED = 20260802
VARIANT = "synthetic_seedvr2_dit_block_c512_h4_hd128"
VARIANT_LL = VARIANT + "_last_layer"

SCALED_CFG = dict(
    vid_dim=512, txt_dim=512, emb_dim=3072, heads=4, head_dim=128,
    expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
    qk_bias=False, qk_norm="fusedrms", mlp_type="swiglu",
    shared_weights=False, rope_type="mmrope3d", rope_dim=128,
    is_last_layer=False, window=(4, 3, 3), window_method="720pwin_by_size_bysize",
)

PROFILE_SHAPES = {
    "A": [(1, 4, 4, 2)],
    "B": [(10, 4, 6, 8)],
    "C": [(4, 6, 10, 11)],
    "D": [(2, 8, 8, 2), (3, 10, 12, 5)],
}


def window_metadata(vid_shape):
    """Real na.window_idx over the real window op for the full batch."""
    from models.dit_v2 import na
    from models.dit_v2.window import get_window_op

    window = SCALED_CFG["window"]
    method = SCALED_CFG["window_method"]

    def make_window(x):
        t, h, w, _ = x.shape
        slices = get_window_op(method)((t, h, w), window)
        return [x[st, sh, sw] for (st, sh, sw) in slices]

    partition, reverse, window_shape, window_count = na.window_idx(vid_shape, make_window)
    idx = torch.arange(window_shape.prod(-1).sum(), device=vid_shape.device)
    tgt = partition(idx.unsqueeze(-1)).squeeze(-1)
    src = torch.argsort(tgt)
    return {
        "partition_index": tgt.tolist(),
        "reverse_index": src.tolist(),
        "window_shape": window_shape.tolist(),
        "window_count": window_count.tolist(),
        "window_sizes": window_shape.prod(-1).tolist(),
    }


def per_window_txt_len(win, txt_shape):
    """txt length of each window (sample i repeated window_count_i times)."""
    result = []
    counts = win["window_count"]
    txt = [int(x[0]) for x in txt_shape.tolist()]
    for l, n in zip(txt, counts):
        result.extend([l] * n)
    return result


def stage_stats(value):
    v = value.detach().cpu().float()
    finite = bool(torch.isfinite(v).all().item())
    return {
        "logical_shape": list(v.shape),
        "dtype": str(v.dtype).replace("torch.", ""),
        "numel": int(v.numel()),
        "finite": finite,
        "mean": None if not finite else float(v.mean().item()),
        "std": None if not finite else float(v.std().item()),
        "min": None if not finite else float(v.min().item()),
        "max": None if not finite else float(v.max().item()),
    }


def derive_stages(cap, win, txt_shape):
    """Derive q/k/v splits, window outputs and residuals from captured tensors."""
    out = {}
    qkv_vid = cap.kept("qkv_vid").detach().cpu().contiguous().float()
    qkv_txt = cap.kept("qkv_txt").detach().cpu().contiguous().float()
    out["q_vid"] = qkv_vid[:, 0:512]
    out["k_vid"] = qkv_vid[:, 512:1024]
    out["v_vid"] = qkv_vid[:, 1024:1536]
    out["q_txt"] = qkv_txt[:, 0:512]
    out["k_txt"] = qkv_txt[:, 512:1024]
    out["v_txt"] = qkv_txt[:, 1024:1536]

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
    out["attention_vid_windows"] = torch.cat(vid_wins) if vid_wins else packed.new_zeros(0, 4, 128)
    out["attention_txt_windows"] = torch.cat(txt_wins) if txt_wins else packed.new_zeros(0, 4, 128)
    out["attention_txt_mean_merged"] = cap.kept("attn_merge_txt").detach().cpu().contiguous().float()
    out["attention_vid_merged"] = cap.kept("attn_merge_vid").detach().cpu().contiguous().float()

    in_vid = cap.kept("input_vid").detach().cpu().contiguous().float()
    in_txt = cap.kept("input_txt").detach().cpu().contiguous().float()
    gated_vid = cap.kept("ada_vid_attn_out_out").detach().cpu().contiguous().float()
    if "ada_txt_attn_out_out" in cap._kept:
        gated_txt = cap.kept("ada_txt_attn_out_out").detach().cpu().contiguous().float()
    else:
        # last-layer vid_only ada: txt attention output is not gated
        gated_txt = cap.kept("attn_proj_txt").detach().cpu().contiguous().float()
    out["attention_residual_vid"] = gated_vid + in_vid
    out["attention_residual_txt"] = gated_txt + in_txt
    return out


def _txt_mean_merge(txt_windows, win, txt_shape):
    """Mean-merge the window-repeated text attention outputs (real unconcat_coalesce)."""
    counts = win["window_count"]
    txt = [int(x[0]) for x in txt_shape.tolist()]
    parts, off = [], 0
    for l, n in zip(txt, counts):
        block = txt_windows[off:off + l * n].reshape(l, n, 4, 128).mean(1)
        parts.append(block)
        off += l * n
    return torch.cat(parts)


STAGE_ALIAS = {
    "input_vid": "input_vid", "input_txt": "input_txt", "emb": "emb",
    "vid_shape": "vid_shape", "txt_shape": "txt_shape",
    "attn_norm_vid": "attn_norm_vid", "attn_norm_txt": "attn_norm_txt",
    "ada_attn_in_vid": "ada_vid_attn_in_out", "ada_attn_in_txt": "ada_txt_attn_in_out",
    "qkv_vid": "qkv_vid", "qkv_txt": "qkv_txt",
    "q_vid": "q_vid", "k_vid": "k_vid", "v_vid": "v_vid",
    "q_txt": "q_txt", "k_txt": "k_txt", "v_txt": "v_txt",
    "q_norm_vid": "q_norm_vid", "k_norm_vid": "k_norm_vid",
    "q_norm_txt": "q_norm_txt", "k_norm_txt": "k_norm_txt",
    "rope_before_q_vid": "rope_before_q_vid", "rope_before_k_vid": "rope_before_k_vid",
    "rope_after_q_vid": "rope_after_q_vid", "rope_after_k_vid": "rope_after_k_vid",
    "rope_after_q_txt": "rope_after_q_txt", "rope_after_k_txt": "rope_after_k_txt",
    "rope_freqs_vid": "rope_freqs_vid", "rope_freqs_txt": "rope_freqs_txt",
    "packed_q": "packed_q", "packed_k": "packed_k", "packed_v": "packed_v",
    "attention_packed_output": "attention_packed_output",
    "attention_vid_windows": "attention_vid_windows", "attention_txt_windows": "attention_txt_windows",
    "attention_vid_merged": "attention_vid_merged", "attention_txt_mean_merged": "attention_txt_mean_merged",
    "attention_proj_vid": "attn_proj_vid", "attention_proj_txt": "attn_proj_txt",
    "ada_attn_gate_vid": "ada_vid_attn_out_out", "ada_attn_gate_txt": "ada_txt_attn_out_out",
    "attention_residual_vid": "attention_residual_vid", "attention_residual_txt": "attention_residual_txt",
    "mlp_norm_vid": "mlp_norm_vid", "mlp_norm_txt": "mlp_norm_txt",
    "ada_mlp_in_vid": "ada_vid_mlp_in_out", "ada_mlp_in_txt": "ada_txt_mlp_in_out",
    "swiglu_gate_projection_vid": "swiglu_gate_projection_vid", "swiglu_gate_projection_txt": "swiglu_gate_projection_txt",
    "swiglu_up_projection_vid": "swiglu_up_projection_vid", "swiglu_up_projection_txt": "swiglu_up_projection_txt",
    "swiglu_product_vid": "swiglu_product_vid", "swiglu_product_txt": "swiglu_product_txt",
    "swiglu_output_vid": "swiglu_output_vid", "swiglu_output_txt": "swiglu_output_txt",
    "ada_mlp_gate_vid": "ada_vid_mlp_out_out", "ada_mlp_gate_txt": "ada_txt_mlp_out_out",
    "final_vid": "final_vid", "final_txt": "final_txt",
}

STAGE_ORDER = [
    "input_vid", "input_txt", "vid_shape", "txt_shape", "emb",
    "attn_norm_vid", "attn_norm_txt", "ada_attn_in_vid", "ada_attn_in_txt",
    "qkv_vid", "qkv_txt", "q_vid", "k_vid", "v_vid", "q_txt", "k_txt", "v_txt",
    "q_norm_vid", "k_norm_vid", "q_norm_txt", "k_norm_txt",
    "rope_before_q_vid", "rope_before_k_vid", "rope_after_q_vid", "rope_after_k_vid",
    "rope_after_q_txt", "rope_after_k_txt", "rope_freqs_vid", "rope_freqs_txt",
    "attention_vid_merged", "attention_txt_mean_merged",
    "packed_q", "packed_k", "packed_v",
    "attention_packed_output", "attention_vid_windows", "attention_txt_windows",
    "attention_proj_vid", "attention_proj_txt",
    "ada_attn_gate_vid", "ada_attn_gate_txt",
    "attention_residual_vid", "attention_residual_txt",
    "mlp_norm_vid", "mlp_norm_txt", "ada_mlp_in_vid", "ada_mlp_in_txt",
    "swiglu_gate_projection_vid", "swiglu_gate_projection_txt",
    "swiglu_up_projection_vid", "swiglu_up_projection_txt",
    "swiglu_product_vid", "swiglu_product_txt",
    "swiglu_output_vid", "swiglu_output_txt",
    "ada_mlp_gate_vid", "ada_mlp_gate_txt",
    "final_vid", "final_txt",
]

PRODUCERS = {
    "input_vid": "external", "input_txt": "external", "vid_shape": "external", "txt_shape": "external",
    "emb": "external",
    "attn_norm_vid": "attn_norm.vid", "attn_norm_txt": "attn_norm.txt",
    "ada_attn_in_vid": "AdaSingle.vid attn in (modulated)", "ada_attn_in_txt": "AdaSingle.txt attn in (modulated)",
    "qkv_vid": "attn.proj_qkv.vid", "qkv_txt": "attn.proj_qkv.txt",
    "q_vid": "derive qkv_vid[:,0:512]", "k_vid": "derive qkv_vid[:,512:1024]", "v_vid": "derive qkv_vid[:,1024:1536]",
    "q_txt": "derive qkv_txt[:,0:512]", "k_txt": "derive qkv_txt[:,512:1024]", "v_txt": "derive qkv_txt[:,1024:1536]",
    "q_norm_vid": "attn.norm_q.vid", "k_norm_vid": "attn.norm_k.vid",
    "q_norm_txt": "attn.norm_q.txt", "k_norm_txt": "attn.norm_k.txt",
    "rope_before_q_vid": "rope input vid_q", "rope_before_k_vid": "rope input vid_k",
    "rope_after_q_vid": "NaMMRotaryEmbedding3d vid_q", "rope_after_k_vid": "NaMMRotaryEmbedding3d vid_k",
    "rope_after_q_txt": "NaMMRotaryEmbedding3d txt_q", "rope_after_k_txt": "NaMMRotaryEmbedding3d txt_k",
    "rope_freqs_vid": "get_freqs vid_freqs", "rope_freqs_txt": "get_freqs txt_freqs",
    "packed_q": "repeat_concat_idx concat (flash input q)", "packed_k": "repeat_concat_idx concat (flash input k)",
    "packed_v": "repeat_concat_idx concat (flash input v)",
    "attention_packed_output": "FlashAttentionVarlen (FP32 compat)",
    "attention_vid_windows": "derive packed per-window vid rows",
    "attention_txt_windows": "derive packed per-window txt rows",
    "attention_vid_merged": "window_reverse + head merge", "attention_txt_mean_merged": "txt mean coalesce",
    "attention_proj_vid": "attn.proj_out.vid", "attention_proj_txt": "attn.proj_out.txt",
    "ada_attn_gate_vid": "AdaSingle.vid attn out (gated)", "ada_attn_gate_txt": "AdaSingle.txt attn out (gated)",
    "attention_residual_vid": "derive gate+input_vid", "attention_residual_txt": "derive gate+input_txt",
    "mlp_norm_vid": "mlp_norm.vid", "mlp_norm_txt": "mlp_norm.txt",
    "ada_mlp_in_vid": "AdaSingle.vid mlp in (modulated)", "ada_mlp_in_txt": "AdaSingle.txt mlp in (modulated)",
    "swiglu_gate_projection_vid": "mlp.vid.proj_in_gate", "swiglu_gate_projection_txt": "mlp.txt.proj_in_gate",
    "swiglu_up_projection_vid": "mlp.vid.proj_in", "swiglu_up_projection_txt": "mlp.txt.proj_in",
    "swiglu_product_vid": "mlp.vid.proj_out input (silu(gate)*up)", "swiglu_product_txt": "mlp.txt.proj_out input",
    "swiglu_output_vid": "mlp.vid.proj_out", "swiglu_output_txt": "mlp.txt.proj_out",
    "ada_mlp_gate_vid": "AdaSingle.vid mlp out (gated)", "ada_mlp_gate_txt": "AdaSingle.txt mlp out (gated)",
    "final_vid": "block output vid", "final_txt": "block output txt",
}


def write_bins(cap, derived, target, profile, stage_order=None):
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"profile": profile, "stages": {}, "windows": None, "inputs": None}
    for name in (stage_order or STAGE_ORDER):
        if name in derived:
            value = derived[name]
        elif STAGE_ALIAS.get(name) in cap._kept:
            value = cap.kept(STAGE_ALIAS[name])
        else:
            continue
        value = value.detach().cpu().contiguous().float()
        file = target / f"{name}.bin"
        value.numpy().tofile(file)
        stats = stage_stats(value)
        stats["file"] = f"{name}.bin"
        stats["producer"] = PRODUCERS[name]
        stats["consumer"] = "ncnn stage parity" if name not in ("input_vid", "input_txt", "emb", "vid_shape", "txt_shape") else "runtime input"
        manifest["stages"][name] = stats
    return manifest


def run_profile(block, cap, name):
    inputs = probe_lib.make_scaled_inputs(PROFILE_SHAPES[name], vid_dim=512, seed=SEED)
    probe_lib.clear_rope_cache(block)
    with torch.no_grad():
        vid_out, txt_out, vs, ts = probe_lib.run_forward(block, cap, name, inputs, keep_mode=True)
    win = window_metadata(inputs["vid_shape"])
    txt_len_win = per_window_txt_len(win, inputs["txt_shape"])
    seg_lens = [s + l for s, l in zip(win["window_sizes"], txt_len_win)]
    cu_q = cap._kept["cu_seqlens_q"][-1]
    cu_k = cap._kept["cu_seqlens_k"][-1]
    max_q = cap._kept["max_seqlen_q"][-1]
    return {
        "inputs": inputs,
        "vid_out": vid_out, "txt_out": txt_out,
        "win": win,
        "seg_lens": seg_lens,
        "cu_seqlens_q": cu_q.tolist(),
        "cu_seqlens_k": cu_k.tolist(),
        "max_segment_q": int(max_q),
        "total_packed": int(cu_q[-1].item()),
        "deterministic": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "artifacts" / VARIANT)
    parser.add_argument("--last-layer-out", type=Path, default=HERE / "artifacts" / VARIANT_LL)
    parser.add_argument("--profiles", default="A,B,C,D")
    args = parser.parse_args()

    probe_lib.ensure_compat()
    torch.Tensor.bfloat16 = lambda self: self  # FP32 attention mode (documented)

    from models.dit_v2.nablocks.mmsr_block import NaMMSRTransformerBlock
    import inspect

    provenance = {
        "seedvr2_commit": probe_lib.SEEDVR2_ROOT,
        "seedvr2_head": _git_head(probe_lib.SEEDVR2_ROOT),
        "block_class": NaMMSRTransformerBlock.__name__,
        "block_module": NaMMSRTransformerBlock.__module__,
        "block_source": inspect.getsourcefile(NaMMSRTransformerBlock),
        "torch_version": torch.__version__,
        "seed": SEED,
        "real_seedvr2_weights": False,
        "real_flash_attn_cuda_kernel": False,
        "real_apex_cuda_kernel": False,
        "attention_dtype_mode": "fp32 (torch.Tensor.bfloat16 patched to identity in test harness)",
        "flash_attn": "cpu_compat (math-equivalent FP32 varlen)",
        "apex": "cpu_compat (FusedRMSNorm FP32)",
        "pnnx": "fp16=0 (see export_synthetic_dit_block.py)",
    }

    # ---- normal block ----
    block = probe_lib.build_block(dict(SCALED_CFG), seed=SEED).eval()
    cap = probe_lib.StageCapture()
    install_phase12_capture(block, cap, capture_txt=True)

    profiles = [p for p in args.profiles.split(",") if p]
    runs = {}
    for name in profiles:
        info = run_profile(block, cap, name)
        derived = derive_stages(cap, info["win"], info["inputs"]["txt_shape"])
        manifest = write_bins(cap, derived, args.out / "reference" / name, name)
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
        runs[name] = info
        (args.out / "reference" / name / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"profile {name}: windows={info['win']['window_sizes']} seg_lens={info['seg_lens']} max_seg={info['max_segment_q']}")

    # determinism: A -> B -> C -> D -> A bit-identical
    first_a = (runs["A"]["vid_out"], runs["A"]["txt_out"])
    info_a2 = run_profile(block, cap, "A")
    second_a = (info_a2["vid_out"], info_a2["txt_out"])
    det_ok = torch.equal(first_a[0], second_a[0]) and torch.equal(first_a[1], second_a[1])
    provenance["a_deterministic_bit_identical"] = bool(det_ok)

    torch.save({k: v.detach().cpu().clone() for k, v in block.state_dict().items()}, args.out / "block_state.pt")
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("normal block references written to", args.out)
    if not det_ok:
        print("ERROR: A->A not bit-identical")
        return 1

    # ---- last-layer block ----
    ll_cfg = dict(SCALED_CFG, is_last_layer=True)
    ll_block = probe_lib.build_block(ll_cfg, seed=SEED).eval()
    ll_cap = probe_lib.StageCapture()
    install_phase12_capture(ll_block, ll_cap, capture_txt=False)
    ll_stage_order = [s for s in STAGE_ORDER if "txt" not in s or s in (
        "input_txt", "qkv_txt", "q_txt", "k_txt", "v_txt", "q_norm_txt", "k_norm_txt",
        "rope_after_q_txt", "rope_after_k_txt", "rope_freqs_txt", "packed_q", "packed_k",
        "packed_v", "attention_packed_output", "attention_txt_windows", "attention_txt_mean_merged",
        "attention_proj_txt", "ada_attn_gate_txt", "attention_residual_txt", "final_txt") or "mlp" in s]
    ll_stage_order = [s for s in STAGE_ORDER if s not in (
        "mlp_norm_txt", "ada_mlp_in_txt", "swiglu_gate_projection_txt", "swiglu_up_projection_txt",
        "swiglu_product_txt", "swiglu_output_txt", "ada_mlp_gate_txt")]
    for name in profiles:
        inputs = probe_lib.make_scaled_inputs(PROFILE_SHAPES[name], vid_dim=512, seed=SEED)
        probe_lib.clear_rope_cache(ll_block)
        with torch.no_grad():
            vid_out, txt_out, vs, ts = probe_lib.run_forward(ll_block, ll_cap, name, inputs, keep_mode=True)
        win = window_metadata(inputs["vid_shape"])
        derived = derive_stages(ll_cap, win, inputs["txt_shape"])
        manifest = write_bins(ll_cap, derived, args.last_layer_out / "reference" / name, name, stage_order=ll_stage_order)
        manifest["windows"] = {
            "window_shape": win["window_shape"],
            "window_count": win["window_count"],
            "window_sizes": win["window_sizes"],
            "partition_index": win["partition_index"],
            "reverse_index": win["reverse_index"],
            "segment_lengths": [s + l for s, l in zip(win["window_sizes"], per_window_txt_len(win, inputs["txt_shape"]))],
            "cu_seqlens_q": ll_cap._kept["cu_seqlens_q"][-1].tolist(),
            "cu_seqlens_k": ll_cap._kept["cu_seqlens_k"][-1].tolist(),
            "max_segment": int(ll_cap._kept["max_seqlen_q"][-1]),
            "total_packed": int(ll_cap._kept["cu_seqlens_q"][-1][-1].item()),
        }
        manifest["inputs"] = {
            "vid_shape": inputs["vid_shape"].tolist(),
            "txt_shape": inputs["txt_shape"].tolist(),
            "vid_tokens": int(inputs["vid_shape"].prod(-1).sum().item()),
            "txt_tokens": int(inputs["txt_shape"].sum().item()),
            "batch": len(inputs["vid_shape"]),
        }
        (args.last_layer_out / "reference" / name / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save({k: v.detach().cpu().clone() for k, v in ll_block.state_dict().items()}, args.last_layer_out / "block_state.pt")
    (args.last_layer_out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("last-layer references written to", args.last_layer_out)
    return 0


def _git_head(root: str) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    sys.exit(main())
