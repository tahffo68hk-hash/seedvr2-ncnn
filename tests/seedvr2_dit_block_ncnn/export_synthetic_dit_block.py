"""Phase 12: export the scaled SeedVR2 DiT block stages to FP32 ncnn artifacts.

Rebuilds the REAL `NaMMSRTransformerBlock` at the scaled C=512 config with the
same seed, loads the reference state dict (strict), then traces small wrapper
modules over the REAL parameter tensors and converts each with
`pnnx ... fp16=0`.  Attention-dtype mode is FP32 throughout (the reference
already ran with the documented bf16->fp32 test-harness patch, and ncnn is a
FP32 CPU runtime; real flash-attn bf16 kernel fidelity is NOT claimed).

Linear stages are exported with a small trace token count and the trace-baked
Gemm M parameter is then rewritten to dynamic (M=0) so any token count is
accepted.  The attention head / rope models are exported at a fixed padded
capacity (Q_MAX=128 rows) which the host runtime always pads to.

The last-layer variant reuses every attention-stage artifact of the base
variant (the constructor seeds the attention modules identically for
is_last_layer=False/True) and adds its own vid-only MLP/SwiGLU artifacts plus
its own AdaSingle params.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))

import probe_lib  # noqa: E402

SEED = 20260802
VARIANT = "synthetic_seedvr2_dit_block_c512_h4_hd128"
VARIANT_LL = VARIANT + "_last_layer"
C, HEADS, HD = 512, 4, 128
QKV = 3 * C
MLP_HIDDEN = 1536
Q_MAX = 128
PAIRS = HD // 2  # 64
SCALE = HD ** -0.5

SCALED_CFG = dict(
    vid_dim=C, txt_dim=C, emb_dim=6 * C, heads=HEADS, head_dim=HD,
    expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
    qk_bias=False, qk_norm="fusedrms", mlp_type="swiglu",
    shared_weights=False, rope_type="mmrope3d", rope_dim=128,
    is_last_layer=False, window=(4, 3, 3), window_method="720pwin_by_size_bysize",
)


class LinearUnit(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RMSNormUnit(nn.Module):
    def __init__(self, dim, eps, weight=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(weight.clone()) if weight is not None else None

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out


class RopeApplyUnit(nn.Module):
    def forward(self, x, cos, sin):
        m = x.shape[0]
        v = x.reshape(m, PAIRS, 2)
        even, odd = v[..., 0], v[..., 1]
        # cos/sin are [m, pairs]; even/odd are [m, pairs]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).reshape(m, HD)


class HeadUnit(nn.Module):
    """Attention-only head over pre-rotated q/k: scores=scale*(q@k^T)+mask, softmax, @v."""

    def forward(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) * SCALE + mask
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


class SwigluActUnit(nn.Module):
    def forward(self, gate, up):
        return F.silu(gate) * up


def write_raw(value: torch.Tensor, path: Path) -> None:
    value.detach().cpu().contiguous().float().numpy().tofile(path)


def trace_and_save(model, example, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.jit.trace(model, example, check_trace=True).save(str(target / "model.pt"))


def run_pnnx(pnnx: Path, directory: Path, shape: str, libtorch: Path) -> None:
    env = dict(os.environ)
    env["PATH"] = str(libtorch) + os.pathsep + env.get("PATH", "")
    subprocess.run([str(pnnx), "model.pt", f"inputshape={shape}", "fp16=0"],
                   cwd=directory, check=True, env=env)


def make_gemm_dynamic(param_path: Path) -> None:
    """Rewrite trace-baked Gemm M (param 7) to dynamic (0) for Linear stages."""
    lines = param_path.read_text(encoding="utf-8").splitlines()
    changed = 0
    out = []
    for line in lines:
        if line.startswith("Gemm"):
            parts = line.split()
            for i, part in enumerate(parts):
                if part.startswith("7=") and part != "7=0":
                    parts[i] = "7=0"
                    changed += 1
            line = " ".join(parts)
        out.append(line)
    if changed:
        param_path.write_text("\n".join(out), encoding="utf-8")


def export_linear(tag, weight, bias, trace_rows, shape, target, pnnx, libtorch):
    unit = LinearUnit(weight, bias)
    x = torch.randn(trace_rows, weight.shape[1])
    with torch.no_grad():
        expected = unit(x)
    trace_and_save(unit, x, target)
    run_pnnx(pnnx, target, shape, libtorch)
    make_gemm_dynamic(target / "model.ncnn.param")
    write_raw(expected, target / "reference_output.bin")
    return target


def export_rms(tag, dim, eps, weight, trace_rows, shape, target, pnnx, libtorch):
    unit = RMSNormUnit(dim, eps, weight)
    x = torch.randn(trace_rows, dim)
    with torch.no_grad():
        expected = unit(x)
    trace_and_save(unit, x, target)
    run_pnnx(pnnx, target, shape, libtorch)
    write_raw(expected, target / "reference_output.bin")
    return target


def load_blocks(root: Path, ll_root: Path):
    probe_lib.ensure_compat()
    torch.Tensor.bfloat16 = lambda self: self
    # Block weights are deterministically reproducible from SEED; the optional
    # block_state.pt is loaded when present (reference generator writes it),
    # otherwise the block is built directly from the seed (same weights).
    block = probe_lib.build_block(dict(SCALED_CFG), seed=SEED).eval()
    sd_path = root / "block_state.pt"
    if sd_path.exists():
        sd = torch.load(sd_path, map_location="cpu")
        block.load_state_dict({k: v for k, v in sd.items()}, strict=True)

    ll_block = probe_lib.build_block(dict(SCALED_CFG, is_last_layer=True), seed=SEED).eval()
    sd_ll_path = ll_root / "block_state.pt"
    if sd_ll_path.exists():
        sd_ll = torch.load(sd_ll_path, map_location="cpu")
        ll_block.load_state_dict({k: v for k, v in sd_ll.items()}, strict=True)
    return block, ll_block


def build_metadata(root: Path, ll_root: Path, pnnx: Path) -> dict:
    pnnx_version = "unknown"
    try:
        env = dict(os.environ)
        env["PATH"] = str(Path(r"V:\AI\LibTorch\2.7.0+cpu\lib")) + os.pathsep + env.get("PATH", "")
        r = subprocess.run([str(pnnx), "-V"], capture_output=True, text=True, env=env)
        pnnx_version = (r.stdout or r.stderr).strip()
    except Exception:
        pass
    base = {
        "name": VARIANT, "variant": VARIANT, "execution": "staged_dit_block",
        "window": [4, 3, 3], "grid": [1, 1, 1], "dtype": "fp32",
        "block": {
            "hidden_dim": C, "txt_dim": C, "heads": HEADS, "head_dim": HD,
            "configured_rope_dim": 128, "effective_rotary_dim": 126, "identity_tail_dim": 2,
            "norm_eps": 1e-5, "mlp_hidden_dim": MLP_HIDDEN, "mlp_type": "swiglu",
            "is_last_layer": False, "weight_seed": SEED, "real_seedvr2_weights": False,
            "q_max": Q_MAX, "head_pairs": PAIRS,
        },
        "params_dir": "params",
        "models": {
            "attn_norm": {"name": "attn_norm", "ncnn_param": "models/attn_norm/model.ncnn.param", "ncnn_bin": "models/attn_norm/model.ncnn.bin"},
            "qkv_vid": {"name": "qkv_vid", "ncnn_param": "models/qkv_vid/model.ncnn.param", "ncnn_bin": "models/qkv_vid/model.ncnn.bin"},
            "qkv_txt": {"name": "qkv_txt", "ncnn_param": "models/qkv_txt/model.ncnn.param", "ncnn_bin": "models/qkv_txt/model.ncnn.bin"},
            "q_norm_vid": {"name": "q_norm_vid", "ncnn_param": "models/q_norm_vid/model.ncnn.param", "ncnn_bin": "models/q_norm_vid/model.ncnn.bin"},
            "q_norm_txt": {"name": "q_norm_txt", "ncnn_param": "models/q_norm_txt/model.ncnn.param", "ncnn_bin": "models/q_norm_txt/model.ncnn.bin"},
            "k_norm_vid": {"name": "k_norm_vid", "ncnn_param": "models/k_norm_vid/model.ncnn.param", "ncnn_bin": "models/k_norm_vid/model.ncnn.bin"},
            "k_norm_txt": {"name": "k_norm_txt", "ncnn_param": "models/k_norm_txt/model.ncnn.param", "ncnn_bin": "models/k_norm_txt/model.ncnn.bin"},
            "rope_apply": {"name": "rope_apply", "ncnn_param": "models/rope_apply/model.ncnn.param", "ncnn_bin": "models/rope_apply/model.ncnn.bin"},
            "heads": [{"name": f"head{i}", "ncnn_param": "models/head/model.ncnn.param", "ncnn_bin": "models/head/model.ncnn.bin"} for i in range(HEADS)],
            "attn_proj_vid": {"name": "attn_proj_vid", "ncnn_param": "models/attn_proj_vid/model.ncnn.param", "ncnn_bin": "models/attn_proj_vid/model.ncnn.bin"},
            "attn_proj_txt": {"name": "attn_proj_txt", "ncnn_param": "models/attn_proj_txt/model.ncnn.param", "ncnn_bin": "models/attn_proj_txt/model.ncnn.bin"},
            "mlp_norm": {"name": "mlp_norm", "ncnn_param": "models/mlp_norm/model.ncnn.param", "ncnn_bin": "models/mlp_norm/model.ncnn.bin"},
            "swiglu_gate_vid": {"name": "swiglu_gate_vid", "ncnn_param": "models/swiglu_gate_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_gate_vid/model.ncnn.bin"},
            "swiglu_gate_txt": {"name": "swiglu_gate_txt", "ncnn_param": "models/swiglu_gate_txt/model.ncnn.param", "ncnn_bin": "models/swiglu_gate_txt/model.ncnn.bin"},
            "swiglu_up_vid": {"name": "swiglu_up_vid", "ncnn_param": "models/swiglu_up_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_up_vid/model.ncnn.bin"},
            "swiglu_up_txt": {"name": "swiglu_up_txt", "ncnn_param": "models/swiglu_up_txt/model.ncnn.param", "ncnn_bin": "models/swiglu_up_txt/model.ncnn.bin"},
            "swiglu_act": {"name": "swiglu_act", "ncnn_param": "models/swiglu_act/model.ncnn.param", "ncnn_bin": "models/swiglu_act/model.ncnn.bin"},
            "swiglu_out_vid": {"name": "swiglu_out_vid", "ncnn_param": "models/swiglu_out_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_out_vid/model.ncnn.bin"},
            "swiglu_out_txt": {"name": "swiglu_out_txt", "ncnn_param": "models/swiglu_out_txt/model.ncnn.param", "ncnn_bin": "models/swiglu_out_txt/model.ncnn.bin"},
        },
        "pnnx": {"command": "pnnx model.pt inputshape=[...] fp16=0", "version": pnnx_version, "fp16": 0},
    }
    ll = {
        "name": VARIANT_LL, "variant": VARIANT_LL, "execution": "staged_dit_block",
        "window": [4, 3, 3], "grid": [1, 1, 1], "dtype": "fp32",
        "block": {
            "hidden_dim": C, "txt_dim": C, "heads": HEADS, "head_dim": HD,
            "configured_rope_dim": 128, "effective_rotary_dim": 126, "identity_tail_dim": 2,
            "norm_eps": 1e-5, "mlp_hidden_dim": MLP_HIDDEN, "mlp_type": "swiglu",
            "is_last_layer": True, "weight_seed": SEED, "real_seedvr2_weights": False,
            "q_max": Q_MAX, "head_pairs": PAIRS,
        },
        "params_dir": "params",
        "shared_base": VARIANT,
        "models": {
            "attn_norm": {"name": "attn_norm", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/attn_norm/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/attn_norm/model.ncnn.bin"},
            "qkv_vid": {"name": "qkv_vid", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/qkv_vid/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/qkv_vid/model.ncnn.bin"},
            "qkv_txt": {"name": "qkv_txt", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/qkv_txt/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/qkv_txt/model.ncnn.bin"},
            "q_norm_vid": {"name": "q_norm_vid", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/q_norm_vid/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/q_norm_vid/model.ncnn.bin"},
            "q_norm_txt": {"name": "q_norm_txt", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/q_norm_txt/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/q_norm_txt/model.ncnn.bin"},
            "k_norm_vid": {"name": "k_norm_vid", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/k_norm_vid/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/k_norm_vid/model.ncnn.bin"},
            "k_norm_txt": {"name": "k_norm_txt", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/k_norm_txt/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/k_norm_txt/model.ncnn.bin"},
            "rope_apply": {"name": "rope_apply", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/rope_apply/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/rope_apply/model.ncnn.bin"},
            "heads": [{"name": f"head{i}", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/head/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/head/model.ncnn.bin"} for i in range(HEADS)],
            "attn_proj_vid": {"name": "attn_proj_vid", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/attn_proj_vid/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/attn_proj_vid/model.ncnn.bin"},
            "attn_proj_txt": {"name": "attn_proj_txt", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/attn_proj_txt/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/attn_proj_txt/model.ncnn.bin"},
            "mlp_norm": {"name": "mlp_norm", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/mlp_norm/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/mlp_norm/model.ncnn.bin"},
            "swiglu_gate_vid": {"name": "swiglu_gate_vid", "ncnn_param": "models/swiglu_gate_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_gate_vid/model.ncnn.bin"},
            "swiglu_up_vid": {"name": "swiglu_up_vid", "ncnn_param": "models/swiglu_up_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_up_vid/model.ncnn.bin"},
            "swiglu_act": {"name": "swiglu_act", "ncnn_param": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/swiglu_act/model.ncnn.param", "ncnn_bin": "../synthetic_seedvr2_dit_block_c512_h4_hd128/models/swiglu_act/model.ncnn.bin"},
            "swiglu_out_vid": {"name": "swiglu_out_vid", "ncnn_param": "models/swiglu_out_vid/model.ncnn.param", "ncnn_bin": "models/swiglu_out_vid/model.ncnn.bin"},
        },
        "pnnx": {"command": "pnnx model.pt inputshape=[...] fp16=0", "version": pnnx_version, "fp16": 0},
    }
    return base, ll


def export_all(root: Path, ll_root: Path, pnnx: Path, libtorch: Path, skip_shared: bool) -> None:
    block, ll_block = load_blocks(root, ll_root)
    sd = {k: v.detach() for k, v in block.state_dict().items()}
    sd_ll = {k: v.detach() for k, v in ll_block.state_dict().items()}

    # verify the last-layer attention modules share weights with the normal block
    for key in ("attn.proj_qkv.vid.weight", "attn.proj_qkv.txt.weight",
                "attn.norm_q.vid.weight", "attn.norm_q.txt.weight",
                "attn.norm_k.vid.weight", "attn.norm_k.txt.weight",
                "attn.proj_out.vid.weight", "attn.proj_out.txt.weight",
                "attn.proj_out.vid.bias", "attn.proj_out.txt.bias"):
        if not torch.equal(sd[key], sd_ll[key]):
            raise RuntimeError(f"last-layer shared attention weight mismatch: {key}")

    models = root / "models"
    ll_models = ll_root / "models"

    if not skip_shared:
        # branch norms (no affine)
        export_rms("attn_norm", C, 1e-5, None, 8, "[8,512]", models / "attn_norm", pnnx, libtorch)
        export_rms("mlp_norm", C, 1e-5, None, 8, "[8,512]", models / "mlp_norm", pnnx, libtorch)
        # qkv projections (bias=False)
        export_linear("qkv_vid", sd["attn.proj_qkv.vid.weight"], None, 8, "[8,512]", models / "qkv_vid", pnnx, libtorch)
        export_linear("qkv_txt", sd["attn.proj_qkv.txt.weight"], None, 8, "[8,512]", models / "qkv_txt", pnnx, libtorch)
        # q/k RMSNorm (affine, per-head shared weight)
        for tag, key in (("q_norm_vid", "attn.norm_q.vid.weight"), ("q_norm_txt", "attn.norm_q.txt.weight"),
                         ("k_norm_vid", "attn.norm_k.vid.weight"), ("k_norm_txt", "attn.norm_k.txt.weight")):
            export_rms(tag, HD, 1e-5, sd[key], 8, "[8,128]", models / tag, pnnx, libtorch)
        # rope apply (fixed padded capacity Q_MAX)
        rope = RopeApplyUnit()
        x = torch.randn(Q_MAX, HD)
        cos = torch.randn(Q_MAX, PAIRS)
        sin = torch.randn(Q_MAX, PAIRS)
        with torch.no_grad():
            expected = rope(x, cos, sin)
        trace_and_save(rope, (x, cos, sin), models / "rope_apply")
        run_pnnx(pnnx, models / "rope_apply", f"[{Q_MAX},{HD}],[{Q_MAX},{PAIRS}],[{Q_MAX},{PAIRS}]", libtorch)
        write_raw(expected, models / "rope_apply" / "reference_output.bin")
        # attention head (attention only, fixed capacity, no weights)
        head = HeadUnit()
        q = torch.randn(Q_MAX, HD); k = torch.randn(Q_MAX, HD); v = torch.randn(Q_MAX, HD)
        mask = torch.zeros(Q_MAX, Q_MAX)
        with torch.no_grad():
            expected = head(q, k, v, mask)
        trace_and_save(head, (q, k, v, mask), models / "head")
        run_pnnx(pnnx, models / "head", f"[{Q_MAX},{HD}],[{Q_MAX},{HD}],[{Q_MAX},{HD}],[{Q_MAX},{Q_MAX}]", libtorch)
        write_raw(expected, models / "head" / "reference_output.bin")
        # attention output projections (bias=True)
        export_linear("attn_proj_vid", sd["attn.proj_out.vid.weight"], sd["attn.proj_out.vid.bias"], 8, "[8,512]", models / "attn_proj_vid", pnnx, libtorch)
        export_linear("attn_proj_txt", sd["attn.proj_out.txt.weight"], sd["attn.proj_out.txt.bias"], 8, "[8,512]", models / "attn_proj_txt", pnnx, libtorch)
        # swiglu gate/up/out for vid and txt (bias=False)
        export_linear("swiglu_gate_vid", sd["mlp.vid.proj_in_gate.weight"], None, 8, "[8,512]", models / "swiglu_gate_vid", pnnx, libtorch)
        export_linear("swiglu_gate_txt", sd["mlp.txt.proj_in_gate.weight"], None, 8, "[8,512]", models / "swiglu_gate_txt", pnnx, libtorch)
        export_linear("swiglu_up_vid", sd["mlp.vid.proj_in.weight"], None, 8, "[8,512]", models / "swiglu_up_vid", pnnx, libtorch)
        export_linear("swiglu_up_txt", sd["mlp.txt.proj_in.weight"], None, 8, "[8,512]", models / "swiglu_up_txt", pnnx, libtorch)
        export_linear("swiglu_out_vid", sd["mlp.vid.proj_out.weight"], None, 8, "[8,1536]", models / "swiglu_out_vid", pnnx, libtorch)
        export_linear("swiglu_out_txt", sd["mlp.txt.proj_out.weight"], None, 8, "[8,1536]", models / "swiglu_out_txt", pnnx, libtorch)
        # silu(gate)*up
        act = SwigluActUnit()
        g = torch.randn(8, MLP_HIDDEN); u = torch.randn(8, MLP_HIDDEN)
        with torch.no_grad():
            expected = act(g, u)
        trace_and_save(act, (g, u), models / "swiglu_act")
        run_pnnx(pnnx, models / "swiglu_act", f"[8,{MLP_HIDDEN}],[8,{MLP_HIDDEN}]", libtorch)
        write_raw(expected, models / "swiglu_act" / "reference_output.bin")

    # last-layer vid-only MLP weights
    if not skip_shared:
        export_linear("swiglu_gate_vid", sd_ll["mlp.vid.proj_in_gate.weight"], None, 8, "[8,512]", ll_models / "swiglu_gate_vid", pnnx, libtorch)
        export_linear("swiglu_up_vid", sd_ll["mlp.vid.proj_in.weight"], None, 8, "[8,512]", ll_models / "swiglu_up_vid", pnnx, libtorch)
        export_linear("swiglu_out_vid", sd_ll["mlp.vid.proj_out.weight"], None, 8, "[8,1536]", ll_models / "swiglu_out_vid", pnnx, libtorch)

    # AdaSingle params (host-side modulation data)
    params = root / "params"
    params.mkdir(parents=True, exist_ok=True)
    ll_params = ll_root / "params"
    ll_params.mkdir(parents=True, exist_ok=True)
    for side in ("vid", "txt"):
        for layer in ("attn", "mlp"):
            for p in ("shift", "scale", "gate"):
                write_raw(sd[f"ada.{side}.{layer}_{p}"], params / f"ada_{side}_{layer}_{p}.bin")
    for layer in ("attn", "mlp"):
        for p in ("shift", "scale", "gate"):
            write_raw(sd_ll[f"ada.vid.{layer}_{p}"], ll_params / f"ada_vid_{layer}_{p}.bin")

    base_meta, ll_meta = build_metadata(root, ll_root, pnnx)
    (root / "metadata.json").write_text(json.dumps(base_meta, indent=2), encoding="utf-8")
    (ll_root / "metadata.json").write_text(json.dumps(ll_meta, indent=2), encoding="utf-8")
    print("export complete:", root)
    print("export complete:", ll_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=Path, default=Path(os.environ.get("PNNX_EXE", HERE.parent.parent / "build" / "pnnx-cpu" / "src" / "pnnx.exe")))
    parser.add_argument("--libtorch", type=Path, default=Path(r"V:\AI\LibTorch\2.7.0+cpu\lib"))
    parser.add_argument("--root", type=Path, default=HERE / "artifacts" / VARIANT)
    parser.add_argument("--last-layer-root", type=Path, default=HERE / "artifacts" / VARIANT_LL)
    parser.add_argument("--skip-shared", action="store_true", help="reuse existing exported models")
    args = parser.parse_args()
    export_all(args.root, args.last_layer_root, args.pnnx, args.libtorch, args.skip_shared)
    return 0


if __name__ == "__main__":
    sys.exit(main())
