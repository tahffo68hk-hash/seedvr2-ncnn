"""Phase 12 capture: real NaMMSRTransformerBlock at the scaled C=512 config.

Registers observation-only hooks/wrappers on the REAL block modules and
records every stage required by the Phase 12 task (section 8) under stable
stage names with full tensor copies.  The block forward source is never
copied or replaced; only observation wrappers and the test-harness CPU
compat (flash_attn / apex) plus the documented bf16->fp32 attention-mode
patch are applied.
"""
from __future__ import annotations

import torch

from probe_lib import StageCapture


def _out_hook(cap, key, source, keep=True):
    def hook(module, args, output):
        cap.record(key, output, source, keep=keep)
    return hook


def _pre_hook(cap, key, source, keep=True):
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


def install_phase12_capture(block, cap, capture_txt=True):
    """Attach Phase 12 stage capture (keep=True everywhere) to a real block."""

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

    # attention branch norms / qkv / qk norm (attention is never vid_only)
    for side in ("vid", "txt"):
        getattr(block.attn_norm, side).register_forward_hook(
            _out_hook(cap, f"attn_norm_{side}", f"attn_norm.{side}"))
        getattr(block.attn.proj_qkv, side).register_forward_hook(
            _out_hook(cap, f"qkv_{side}", f"attn.proj_qkv.{side}"))
        getattr(block.attn.norm_q, side).register_forward_hook(
            _out_hook(cap, f"q_norm_{side}", f"attn.norm_q.{side}"))
        getattr(block.attn.norm_k, side).register_forward_hook(
            _out_hook(cap, f"k_norm_{side}", f"attn.norm_k.{side}"))
        getattr(block.attn.proj_out, side).register_forward_pre_hook(
            _pre_hook(cap, f"attn_merge_{side}", f"attn.proj_out.{side} input"))
        getattr(block.attn.proj_out, side).register_forward_hook(
            _out_hook(cap, f"attn_proj_{side}", f"attn.proj_out.{side}"))

    # mlp branches (txt skipped for vid_only last-layer blocks)
    sides = ("vid", "txt") if capture_txt else ("vid",)
    for side in sides:
        getattr(block.mlp_norm, side).register_forward_hook(
            _out_hook(cap, f"mlp_norm_{side}", f"mlp_norm.{side}"))
        getattr(block.mlp_norm, side).register_forward_pre_hook(
            _pre_hook(cap, f"mlp_norm_{side}_input", f"mlp_norm.{side} input"))
        mlp = getattr(block.mlp, side)
        mlp.proj_in_gate.register_forward_hook(
            _out_hook(cap, f"swiglu_gate_projection_{side}", f"mlp.{side}.proj_in_gate"))
        mlp.proj_in.register_forward_hook(
            _out_hook(cap, f"swiglu_up_projection_{side}", f"mlp.{side}.proj_in"))
        mlp.proj_out.register_forward_pre_hook(
            _pre_hook(cap, f"swiglu_product_{side}", f"mlp.{side}.proj_out input (silu(gate)*up)"))
        mlp.proj_out.register_forward_hook(
            _out_hook(cap, f"swiglu_output_{side}", f"mlp.{side}.proj_out"))

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
                cap.record(f"ada_{side}_{layer}_{mode}", hid,
                           f"AdaSingle.{side} input", keep=True)
                out = orig(hid, emb, layer, mode, cache, branch_tag, hid_len)
                cap.record(f"ada_{side}_{layer}_{mode}_out", out,
                           f"AdaSingle.{side} output", keep=True)
                return out
            return wrapped

        _wrap_forward(ada_mod, make_ada_wrapper)

    # windowed vid rope (text q/k are window-repeated inside the real rope)
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
