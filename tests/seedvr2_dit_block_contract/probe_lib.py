"""Shared harness for the Phase 11.1 real SeedVR2 block forward probe.

The REAL `NaMMSRTransformerBlock` class from `third_party/SeedVR2` is
imported and executed unmodified (real forward source).  Only the fused
CUDA kernel libraries (flash_attn / apex) are replaced in the test harness
by math-equivalent CPU implementations (see `compat/`).  This module
provides block construction, input synthesis, stage hooks/capture, profile
runs, structural sentinel runs and tensor stats used by the three test
entry points.
"""
from __future__ import annotations

import gc
import inspect
import os

import torch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDVR2_ROOT = r"D:\AIProjects\third_party\SeedVR2"

# Real SeedVR2 3B config (configs_3b/main.yaml) for a single block.
REAL_CFG = dict(
    vid_dim=2560, txt_dim=2560, emb_dim=15360, heads=20, head_dim=128,
    expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
    qk_bias=False, qk_norm="fusedrms", mlp_type="swiglu",
    shared_weights=False, rope_type="mmrope3d", rope_dim=128,
    is_last_layer=False, window=(4, 3, 3), window_method="720pwin_by_size_bysize",
)

# (T, H, W, text_len) per sample per profile.
PROFILE_SHAPES = {
    "A_min": [(1, 4, 4, 2)],
    "B_multi_window": [(5, 32, 32, 8)],
    "C_asymmetric": [(7, 45, 64, 11)],
    "D_batch2": [(2, 8, 8, 2), (3, 10, 12, 5)],
}

NEG_CFG = dict(
    vid_dim=256, txt_dim=256, emb_dim=1536, heads=4, head_dim=64,
    rope_dim=64, is_last_layer=False,
)


def ensure_compat() -> dict:
    """Install compat (flash_attn/apex) and SeedVR2 paths; return info dict."""
    from compat.dependency_injection import install_compat, install_probe_paths
    info = install_compat()
    install_probe_paths()
    return info


def build_block(cfg=None, seed=0):
    """Construct the REAL block with factory-converted norm/ada/qk_norm.

    cfg is a dict of REAL_CFG overrides.  Scaled configs are clearly marked
    by the caller as `scaled_random_weight_probe`; the forward is never
    touched or copied.  Enforces the probe-level contract invariant
    heads*head_dim == vid_dim.
    """
    from models.dit_v2.nablocks.mmsr_block import NaMMSRTransformerBlock
    from models.dit_v2.normalization import get_norm_layer
    from models.dit_v2.modulation import get_ada_layer

    c = dict(REAL_CFG)
    if cfg:
        c.update(cfg)
    if c["heads"] * c["head_dim"] != c["vid_dim"]:
        raise AssertionError("probe contract violation: heads*head_dim != vid_dim")
    torch.manual_seed(seed)
    kwargs = dict(c)
    kwargs["norm"] = get_norm_layer(c["norm"])
    kwargs["qk_norm"] = get_norm_layer(c["qk_norm"])
    kwargs["ada"] = get_ada_layer(c["ada"])
    return NaMMSRTransformerBlock(**kwargs).eval()


def make_inputs(name, seed=20260701, dtype=torch.float32):
    """Random FP32 inputs matching the Phase 11 contract for profile `name`."""
    shapes = PROFILE_SHAPES[name]
    g = torch.Generator().manual_seed(seed)
    vid_parts, txt_parts = [], []
    vid_shape_rows, txt_shape_rows = [], []
    vid_dim = REAL_CFG["vid_dim"]
    for (t, h, w, l) in shapes:
        vid_parts.append(torch.randn(t * h * w, vid_dim, generator=g, dtype=dtype))
        txt_parts.append(torch.randn(l, vid_dim, generator=g, dtype=dtype))
        vid_shape_rows.append([t, h, w])
        txt_shape_rows.append([l])
    vs = torch.tensor(vid_shape_rows, dtype=torch.long)
    ts = torch.tensor(txt_shape_rows, dtype=torch.long)
    return {
        "vid": torch.cat(vid_parts),
        "txt": torch.cat(txt_parts),
        "vid_shape": vs,
        "txt_shape": ts,
        "emb": torch.randn(len(shapes), 6 * vid_dim, generator=g, dtype=dtype),
        "meta": {"samples": shapes,
                 "vid_tokens": int(vs.prod(-1).sum().item()),
                 "txt_tokens": int(ts.sum().item())},
    }


def make_scaled_inputs(shapes, vid_dim=256, seed=42):
    """Random FP32 inputs for scaled (non-3B) block configs."""
    g = torch.Generator().manual_seed(seed)
    vid_parts, txt_parts = [], []
    vid_shape_rows, txt_shape_rows = [], []
    for (t, h, w, l) in shapes:
        vid_parts.append(torch.randn(t * h * w, vid_dim, generator=g))
        txt_parts.append(torch.randn(l, vid_dim, generator=g))
        vid_shape_rows.append([t, h, w])
        txt_shape_rows.append([l])
    vs = torch.tensor(vid_shape_rows, dtype=torch.long)
    ts = torch.tensor(txt_shape_rows, dtype=torch.long)
    return {
        "vid": torch.cat(vid_parts), "txt": torch.cat(txt_parts),
        "vid_shape": vs, "txt_shape": ts,
        "emb": torch.randn(len(shapes), 6 * vid_dim, generator=g),
        "meta": {"samples": shapes},
    }


def make_structural_inputs(shapes, vid_value=0.0, txt_value=0.0):
    """Sentinel inputs (constant vid / constant txt) for ordering checks."""
    vid_parts, txt_parts = [], []
    vid_rows, txt_rows = [], []
    for (t, h, w, l) in shapes:
        vid_parts.append(torch.full((t * h * w, 2560), vid_value))
        txt_parts.append(torch.full((l, 2560), txt_value))
        vid_rows.append([t, h, w])
        txt_rows.append([l])
    return {
        "vid": torch.cat(vid_parts).float(),
        "txt": torch.cat(txt_parts).float(),
        "vid_shape": torch.tensor(vid_rows, dtype=torch.long),
        "txt_shape": torch.tensor(txt_rows, dtype=torch.long),
        "emb": torch.zeros(len(shapes), 6 * 2560),
        "meta": {"samples": shapes},
    }


def input_stats(t):
    tf = t.float()
    return {
        "shape": list(t.shape), "dtype": str(t.dtype), "numel": int(t.numel()),
        "bytes": int(t.numel() * t.element_size()),
        "finite": bool(torch.isfinite(tf).all().item()),
        "mean": float(tf.mean().item()), "std": float(tf.std().item()),
        "min": float(tf.min().item()), "max": float(tf.max().item()),
    }


def validate_input_contract(inputs, dim=None):
    """Enforce the Phase 11 input contract; raise TypeError/ValueError."""
    vid, txt = inputs["vid"], inputs["txt"]
    vid_shape, txt_shape = inputs["vid_shape"], inputs["txt_shape"]
    emb = inputs["emb"]
    if vid.dtype != torch.float32 or txt.dtype != torch.float32 or emb.dtype != torch.float32:
        raise TypeError(f"input dtype contract violated: vid={vid.dtype} txt={txt.dtype} emb={emb.dtype}")
    if vid.ndim != 2 or txt.ndim != 2:
        raise ValueError(f"vid/txt must be rank-2 [L,C], got {tuple(vid.shape)} / {tuple(txt.shape)}")
    c = vid.shape[1]
    if txt.shape[1] != c or (dim is not None and c != dim):
        raise ValueError(f"hidden dim mismatch: vid C={c}, txt C={txt.shape[1]}, expected dim={dim}")
    if emb.ndim != 2 or emb.shape[1] != 6 * c:
        raise ValueError(f"emb must be [b, 6*C], got {tuple(emb.shape)} (C={c})")
    if emb.shape[0] != vid_shape.shape[0] or vid_shape.shape[0] != txt_shape.shape[0]:
        raise ValueError("batch metadata inconsistent: emb/vid_shape/txt_shape batch mismatch")
    if vid_shape.shape[1] != 3 or txt_shape.shape[1] != 1:
        raise ValueError("vid_shape must be [b,3], txt_shape [b,1]")
    if vid.shape[0] != int(vid_shape.prod(-1).sum().item()):
        raise ValueError(f"vid token count {vid.shape[0]} != vid_shape sum {vid_shape.prod(-1).sum().item()}")
    if txt.shape[0] != int(txt_shape.sum().item()):
        raise ValueError(f"txt token count {txt.shape[0]} != txt_shape sum {txt_shape.sum().item()}")
    return True


class StageCapture:
    """Captures stats (and, on demand, cloned tensors) at module boundaries."""

    def __init__(self):
        self.runs = []
        self.keep_mode = False
        self.run_name = None
        self._stages = None
        self._kept = None

    def begin_run(self, name, keep_mode=False):
        self.run_name = name
        self.keep_mode = keep_mode
        self._stages = {}
        self._kept = {}

    def finish_run(self):
        self.runs.append({"name": self.run_name, "stages": self._stages, "kept": self._kept})

    def last(self):
        return self.runs[-1]

    def record(self, key, value, source="", keep=False):
        if value is None:
            self._stages.setdefault(key, []).append({"source": source, "null": True})
            return
        st = input_stats(value)
        st["source"] = source
        self._stages.setdefault(key, []).append(st)
        if keep and self.keep_mode and isinstance(value, torch.Tensor):
            self._kept.setdefault(key, []).append(value.detach().clone())

    def kept(self, key, idx=0):
        return self._kept[key][idx]

    def stages(self, key):
        return self._stages.get(key, [])


def _out_hook(cap, key, source, keep=False):
    def hook(module, args, output):
        cap.record(key, output, source, keep=keep)
    return hook


def _pre_hook(cap, key, source, keep=False):
    def hook(module, args):
        cap.record(key, args[0], source, keep=keep)
    return hook


def _wrap_forward(module, wrapper):
    orig = module.forward
    module.forward = wrapper(orig)
    return orig


def _wrap_method(obj, name, wrapper):
    orig = getattr(obj, name)
    setattr(obj, name, wrapper(orig))
    return orig


def install_capture(block, cap, capture_txt=True):
    """Register hooks/wrappers capturing every key stage on the REAL modules.

    Nothing in the block's forward is replaced or copied; only observation
    wrappers are attached.  In-place operations (AdaSingle) are snapshotted
    at capture time by the wrapper/hook.
    """

    def block_pre(module, args):
        vid, txt, vid_shape, txt_shape, emb, cache = args
        cap.record("block_input_vid", vid, "NaMMSRTransformerBlock.forward arg", keep=True)
        cap.record("block_input_txt", txt, "NaMMSRTransformerBlock.forward arg", keep=True)
        cap.record("block_input_emb", emb, "emb (b, 6*dim)", keep=True)
        cap.record("block_input_vid_shape", vid_shape.float(), "vid_shape (b,3)", keep=False)
        cap.record("block_input_txt_shape", txt_shape.float(), "txt_shape (b,1)", keep=False)

    def block_post(module, args, output):
        vid_out, txt_out, vs, ts = output
        cap.record("block_out_vid", vid_out, "NaMMSRTransformerBlock output", keep=True)
        cap.record("block_out_txt", txt_out, "NaMMSRTransformerBlock output", keep=True)

    block.register_forward_pre_hook(block_pre)
    block.register_forward_hook(block_post)

    # Attention-side modules always have both branches (attention is never vid_only).
    for side in ("vid", "txt"):
        getattr(block.attn_norm, side).register_forward_hook(
            _out_hook(cap, f"attn_norm_{side}", f"attn_norm.{side} (FusedRMSNorm)", keep=True))
        getattr(block.attn.proj_qkv, side).register_forward_hook(
            _out_hook(cap, f"qkv_{side}", f"attn.proj_qkv.{side} (Linear -> qkv_dim)"))
        getattr(block.attn.norm_q, side).register_forward_hook(
            _out_hook(cap, f"q_norm_{side}", f"attn.norm_q.{side} (FusedRMSNorm)", keep=True))
        getattr(block.attn.norm_k, side).register_forward_hook(
            _out_hook(cap, f"k_norm_{side}", f"attn.norm_k.{side} (FusedRMSNorm)", keep=True))
        getattr(block.attn.proj_out, side).register_forward_hook(
            _out_hook(cap, f"proj_out_{side}", f"attn.proj_out.{side} (Linear)", keep=True))
        getattr(block.attn.proj_out, side).register_forward_pre_hook(
            _pre_hook(cap, f"proj_out_{side}_input", f"attn.proj_out.{side} input", keep=True))

    # MLP/ada/mlp_norm branches (txt skipped when capture_txt=False for vid_only blocks)
    sides = ("vid", "txt") if capture_txt else ("vid",)
    for side in sides:
        getattr(block.mlp_norm, side).register_forward_hook(
            _out_hook(cap, f"mlp_norm_{side}", f"mlp_norm.{side} (FusedRMSNorm)", keep=True))
        getattr(block.mlp_norm, side).register_forward_pre_hook(
            _pre_hook(cap, f"mlp_norm_{side}_input", f"mlp_norm.{side} input (residual)", keep=True))
        mlp = getattr(block.mlp, side)
        mlp.proj_in_gate.register_forward_hook(
            _out_hook(cap, f"swiglu_gate_proj_{side}", f"mlp.{side}.proj_in_gate (Linear)", keep=True))
        mlp.proj_in.register_forward_hook(
            _out_hook(cap, f"swiglu_up_proj_{side}", f"mlp.{side}.proj_in (Linear)", keep=True))
        mlp.proj_out.register_forward_pre_hook(
            _pre_hook(cap, f"swiglu_gate_product_{side}", f"mlp.{side}.proj_out input (silu(gate)*up)", keep=True))
        mlp.proj_out.register_forward_hook(
            _out_hook(cap, f"swiglu_out_proj_{side}", f"mlp.{side}.proj_out (Linear)", keep=True))
        def make_mlp_wrapper(orig, side=side):
            def wrapped(x, side=side):
                cap.record(f"mlp_{side}_in", x, f"mlp.{side} input", keep=True)
                out = orig(x)
                cap.record(f"mlp_{side}_out", out, f"mlp.{side} output", keep=True)
                return out
            return wrapped

        _wrap_forward(mlp, make_mlp_wrapper)

        ada_mod = getattr(block.ada, side)

        def make_ada_wrapper(orig, side=side):
            def wrapped(hid, emb, layer, mode, cache=None, branch_tag="", hid_len=None, side=side):
                cap.record(f"ada_{side}_in_{layer}_{mode}", hid,
                           f"AdaSingle.{side} input", keep=True)
                out = orig(hid, emb, layer, mode, cache, branch_tag, hid_len)
                cap.record(f"ada_{side}_out_{layer}_{mode}", out,
                           f"AdaSingle.{side} output", keep=True)
                return out
            return wrapped

        _wrap_forward(ada_mod, make_ada_wrapper)

    def attn_pre(module, args):
        cap.record("attn_in_vid", args[0], "NaSwinAttention input vid", keep=False)
        cap.record("attn_in_txt", args[1], "NaSwinAttention input txt", keep=False)

    block.attn.register_forward_pre_hook(attn_pre)

    if block.attn.rope is not None:
        def make_rope_wrapper(orig):
            def wrapped(vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache):
                cap.record("rope_in_vid_q", vid_q, "NaMMRotaryEmbedding3d input vid_q", keep=True)
                cap.record("rope_in_vid_k", vid_k, "NaMMRotaryEmbedding3d input vid_k", keep=True)
                cap.record("rope_in_txt_q", txt_q, "NaMMRotaryEmbedding3d input txt_q (window-repeated)", keep=True)
                out = orig(vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache)
                cap.record("rope_out_vid_q", out[0], "NaMMRotaryEmbedding3d output vid_q", keep=True)
                cap.record("rope_out_vid_k", out[1], "NaMMRotaryEmbedding3d output vid_k", keep=True)
                cap.record("rope_out_txt_q", out[2], "NaMMRotaryEmbedding3d output txt_q", keep=True)
                return out
            return wrapped

        _wrap_forward(block.attn.rope, make_rope_wrapper)

        def make_get_freqs_wrapper(orig):
            def wrapped(vid_shape, txt_shape):
                out = orig(vid_shape, txt_shape)
                cap.record("rope_freqs_vid", out[0], "get_freqs vid_freqs (l:l+f, :h, :w) per window", keep=True)
                cap.record("rope_freqs_txt", out[1], "get_freqs txt_freqs", keep=True)
                cap._kept.setdefault("rope_freqs_input", []).append(
                    (vid_shape.detach().clone(), txt_shape.detach().clone()))
                return out
            return wrapped

        _wrap_method(block.attn.rope, "get_freqs", make_get_freqs_wrapper)

    def make_varlen_wrapper(orig):
        def wrapped(*args, **kwargs):
            cap.record("attn_varlen_q", kwargs["q"], "packed q (bf16, [vid_win; txt] per segment)", keep=True)
            cap.record("attn_varlen_k", kwargs["k"], "packed k", keep=False)
            cap.record("attn_varlen_v", kwargs["v"], "packed v", keep=False)
            cap._kept.setdefault("attn_varlen_cu_q", []).append(kwargs["cu_seqlens_q"].detach().clone())
            cap._kept.setdefault("attn_varlen_cu_k", []).append(kwargs["cu_seqlens_k"].detach().clone())
            cap._kept.setdefault("attn_varlen_max_q", []).append(kwargs["max_seqlen_q"])
            out = orig(*args, **kwargs)
            cap.record("attn_varlen_out", out, "varlen attention output (bf16)", keep=True)
            return out
        return wrapped

    _wrap_forward(block.attn.attn, make_varlen_wrapper)


def run_forward(block, cap, name, inputs, keep_mode=False, validate=True):
    """Run one forward through the REAL block with the capture active."""
    from common.cache import Cache
    if validate:
        validate_input_contract(inputs, dim=inputs["vid"].shape[1])
    cap.begin_run(name, keep_mode)
    with torch.no_grad():
        vid_out, txt_out, vs, ts = block(
            inputs["vid"], inputs["txt"], inputs["vid_shape"], inputs["txt_shape"],
            inputs["emb"], Cache(disable=True))
    cap.finish_run()
    return vid_out, txt_out, vs, ts


def run_forward_check_finite(block, inputs):
    """Run forward and raise ValueError on non-finite outputs (harness check)."""
    from common.cache import Cache
    validate_input_contract(inputs, dim=inputs["vid"].shape[1])
    with torch.no_grad():
        vo, to, _, _ = block(inputs["vid"], inputs["txt"], inputs["vid_shape"],
                             inputs["txt_shape"], inputs["emb"], Cache(disable=True))
    if not (torch.isfinite(vo.float()).all().item() and torch.isfinite(to.float()).all().item()):
        raise ValueError("non-finite block output detected")
    return vo, to


def window_partition_info(vid_shape, window=(4, 3, 3), window_method="720pwin_by_size_bysize"):
    """Real na.window_idx over the real window op; returns partition info."""
    from models.dit_v2 import na
    from models.dit_v2.window import get_window_op

    def make_window(x):
        t, h, w, _ = x.shape
        op = get_window_op(window_method)
        slices = op((t, h, w), window)
        return [x[st, sh, sw] for (st, sh, sw) in slices]

    partition, reverse, window_shape, window_count = na.window_idx(vid_shape, make_window)
    tgt_idx = partition(torch.arange(window_shape.prod(-1).sum(), device=vid_shape.device).unsqueeze(-1)).squeeze(-1)
    src_idx = torch.argsort(tgt_idx)
    packed_to_win = torch.cat([torch.full((int(window_shape[i].prod().item()),), i, dtype=torch.long)
                               for i in range(window_shape.shape[0])])
    orig_to_win = packed_to_win[src_idx]
    return {
        "window_shape": window_shape, "window_count": window_count,
        "window_sizes": window_shape.prod(-1).tolist(),
        "orig_to_win": orig_to_win,
    }


def clear_rope_cache(block):
    """Free the shared rotary freq caches of a rope-bearing block."""
    try:
        rope = block.attn.rope
        if rope is not None:
            rope.get_axial_freqs.cache_clear()
    except Exception:
        pass


def free_block(block):
    clear_rope_cache(block)
    del block


def gc_collect():
    gc.collect()


def block_source_file() -> str:
    from models.dit_v2.nablocks.mmsr_block import NaMMSRTransformerBlock
    return inspect.getsourcefile(NaMMSRTransformerBlock) or ""


def is_finite(t) -> bool:
    return bool(torch.isfinite(t.float()).all().item())