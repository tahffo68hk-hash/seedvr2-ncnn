"""Phase 14 capture: stage capture for the REAL shared-weight block 16.

The real checkpoint block 16 uses shared_weights=True, so every MMModule has a
single `.all` submodule called twice per forward (vid input then txt input).
This module installs observation-only hooks/wrappers on the REAL block modules
recording the same Phase 12/13 stage names; the alternating `.all` calls are
routed to the vid/txt stage names (AdaSingle uses its explicit branch_tag).
The block forward source is never copied or replaced.
"""
from __future__ import annotations

import torch


def _alt_side(state):
    side = "vid" if state["n"] % 2 == 0 else "txt"
    state["n"] += 1
    return side


def _wrap_forward(module, wrapper):
    orig = module.forward
    module.forward = wrapper(orig)
    return orig


def _wrap_method(obj, name, wrapper):
    orig = getattr(obj, name)
    setattr(obj, name, wrapper(orig))
    return orig


def install_phase14_capture(block, cap, capture_txt=True):
    """Attach shared-weight stage capture (keep=True everywhere) to a real block."""

    def block_pre(module, args):
        vid, txt, vid_shape, txt_shape, emb, cache = args
        cap.record("input_vid", vid, "block input vid", keep=True)
        cap.record("input_txt", txt, "block input txt", keep=True)
        cap.record("emb", emb, "block input emb", keep=True)
        cap.record("vid_shape", vid_shape.float(), "vid_shape", keep=True)
        cap.record("txt_shape", txt_shape.float(), "txt_shape", keep=True)

    def block_post(module, args, output):
        vid_out, txt_out, vs, ts = output
        cap.record("final_vid", vid_out, "block output vid", keep=True)
        cap.record("final_txt", txt_out, "block output txt", keep=True)

    block.register_forward_pre_hook(block_pre)
    block.register_forward_hook(block_post)

    # Shared `.all` attention-side modules: alternating vid/txt recording of OUTPUTS.
    # attn_norm lives at block level; qkv/norm_q/norm_k/proj_out live in attn.
    for mod, attr, base in (
            (block.attn_norm.all, "attn_norm", "attn_norm"),
            (block.attn.proj_qkv.all, "proj_qkv", "qkv"),
            (block.attn.norm_q.all, "norm_q", "q_norm"),
            (block.attn.norm_k.all, "norm_k", "k_norm")):
        state = {"n": 0}
        def make_out(orig, base=base, state=state, attr=attr):
            def wrapped(x, base=base, state=state, attr=attr):
                out = orig(x)
                side = _alt_side(state)
                cap.record(f"{base}_{side}", out, f"{attr}.all output ({side})", keep=True)
                return out
            return wrapped
        _wrap_forward(mod, make_out)

    proj_out = block.attn.proj_out.all
    pstate = {"n": 0}
    def make_proj_out(orig, pstate=pstate):
        def wrapped(x, pstate=pstate):
            side = _alt_side(pstate)
            cap.record(f"attn_merge_{side}", x, f"attn.proj_out.all input ({side})", keep=True)
            out = orig(x)
            cap.record(f"attn_proj_{side}", out, f"attn.proj_out.all output ({side})", keep=True)
            return out
        return wrapped
    _wrap_forward(proj_out, make_proj_out)

    # Shared mlp_norm.
    mnstate = {"n": 0}
    def make_mlp_norm(orig, mnstate=mnstate):
        def wrapped(x, mnstate=mnstate):
            out = orig(x)
            side = _alt_side(mnstate)
            cap.record(f"mlp_norm_{side}", out, f"mlp_norm.all output ({side})", keep=True)
            return out
        return wrapped
    _wrap_forward(block.mlp_norm.all, make_mlp_norm)

    # Shared MLP (SwiGLU) wrapper.
    mlp = block.mlp.all
    mlpstate = {"n": 0}
    def make_mlp_wrapper(orig, mlpstate=mlpstate):
        def wrapped(x, mlpstate=mlpstate):
            side = _alt_side(mlpstate)
            cap.record(f"mlp_{side}_in", x, f"mlp.all input ({side})", keep=True)
            cap.record(f"swiglu_gate_projection_{side}", mlp.proj_in_gate(x),
                       f"mlp.all.proj_in_gate ({side})", keep=True)
            cap.record(f"swiglu_up_projection_{side}", mlp.proj_in(x),
                       f"mlp.all.proj_in ({side})", keep=True)
            prod = torch.nn.functional.silu(mlp.proj_in_gate(x)) * mlp.proj_in(x)
            cap.record(f"swiglu_product_{side}", prod, f"mlp.all silu(gate)*up ({side})", keep=True)
            out = orig(x)
            cap.record(f"swiglu_output_{side}", out, f"mlp.all.proj_out ({side})", keep=True)
            cap.record(f"mlp_{side}_out", out, f"mlp.all output ({side})", keep=True)
            return out
        return wrapped
    _wrap_forward(mlp, make_mlp_wrapper)

    # Shared AdaSingle: use the explicit branch_tag ("vid"/"txt").
    def make_ada_wrapper(orig):
        def wrapped(hid, emb, layer, mode, cache=None, branch_tag="", hid_len=None):
            side = branch_tag if branch_tag in ("vid", "txt") else "vid"
            cap.record(f"ada_{side}_{layer}_{mode}", hid, f"AdaSingle.all {layer} {mode} input ({side})", keep=True)
            out = orig(hid, emb, layer, mode, cache, branch_tag, hid_len)
            cap.record(f"ada_{side}_{layer}_{mode}_out", out, f"AdaSingle.all {layer} {mode} output ({side})", keep=True)
            return out
        return wrapped
    _wrap_forward(block.ada.all, make_ada_wrapper)

    # Windowed vid rope + text q/k (same as phase12).
    if block.attn.rope is not None:
        def make_rope_wrapper(orig):
            def wrapped(vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache):
                cap.record("rope_before_q_vid", vid_q, "rope input vid_q (windowed)", keep=True)
                cap.record("rope_before_k_vid", vid_k, "rope input vid_k (windowed)", keep=True)
                cap.record("rope_before_q_txt", txt_q, "rope input txt_q (window-repeated)", keep=True)
                cap.record("rope_before_k_txt", txt_k, "rope input txt_k (window-repeated)", keep=True)
                out = orig(vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache)
                cap.record("rope_after_q_vid", out[0], "rope output vid_q", keep=True)
                cap.record("rope_after_k_vid", out[1], "rope output vid_k", keep=True)
                cap.record("rope_after_q_txt", out[2], "rope output txt_q", keep=True)
                cap.record("rope_after_k_txt", out[3], "rope output txt_k", keep=True)
                return out
            return wrapped
        _wrap_forward(block.attn.rope, make_rope_wrapper)

        def make_get_freqs_wrapper(orig):
            def wrapped(vid_shape, txt_shape):
                out = orig(vid_shape, txt_shape)
                cap.record("rope_freqs_vid", out[0], "get_freqs vid_freqs", keep=True)
                cap.record("rope_freqs_txt", out[1], "get_freqs txt_freqs", keep=True)
                return out
            return wrapped
        _wrap_method(block.attn.rope, "get_freqs", make_get_freqs_wrapper)

    def make_varlen_wrapper(orig):
        def wrapped(*args, **kwargs):
            cap.record("packed_q", kwargs["q"], "packed q [vid_win; txt]", keep=True)
            cap.record("packed_k", kwargs["k"], "packed k", keep=True)
            cap.record("packed_v", kwargs["v"], "packed v", keep=True)
            cap._kept.setdefault("cu_seqlens_q", []).append(kwargs["cu_seqlens_q"].detach().clone())
            cap._kept.setdefault("cu_seqlens_k", []).append(kwargs["cu_seqlens_k"].detach().clone())
            cap._kept.setdefault("max_seqlen_q", []).append(kwargs["max_seqlen_q"])
            cap._kept.setdefault("max_seqlen_k", []).append(kwargs["max_seqlen_k"])
            out = orig(*args, **kwargs)
            cap.record("attention_packed_output", out, "varlen attention output", keep=True)
            return out
        return wrapped
    _wrap_forward(block.attn.attn, make_varlen_wrapper)
