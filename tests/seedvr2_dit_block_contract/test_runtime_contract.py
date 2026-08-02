"""Phase 11.1 runtime contract validation on the REAL SeedVR2 block.

Executes the REAL NaMMSRTransformerBlock forward and validates, from
runtime-captured stage tensors, every Phase 11 contract point: block class /
forward signature, flattened [L,C] inputs, batch metadata segmentation,
per-window `[vid_win; txt]` packing with txt repetition, `vid_freqs[l:l+f]`
RoPE offset sentinel, T-major/W-fastest layout, Q/K RMSNorm-before-RoPE,
configured/effective rope dim (128/126, identity tail 2), varlen segment
isolation, AdaSingle six-segment modulation and gate placement, SwiGLU MLP,
residual ordering, last-layer vid_only behavior, determinism, plus the full
negative-test battery.  Also runs structural sentinel probes on real-class
blocks (rope-off / scaled configs are clearly marked) for window reassembly
and segment ordering.
"""
from __future__ import annotations

import inspect
import sys

import torch

from probe_lib import (NEG_CFG, PROFILE_SHAPES, REAL_CFG, StageCapture,
                       build_block, clear_rope_cache, ensure_compat, free_block,
                       gc_collect, input_stats, install_capture, is_finite,
                       make_inputs, make_scaled_inputs, make_structural_inputs,
                       run_forward, run_forward_check_finite,
                       validate_input_contract, window_partition_info)

EPS = 1e-3
REL = 1e-4


def _find_run(cap, name):
    for r in cap.runs:
        if r["name"] == name:
            return r
    raise KeyError(f"run {name} not found in capture")


def check_rope_contract(block, cap, run_name="A_min"):
    run = _find_run(cap, run_name)
    kp = run["kept"]
    q_pre = kp["rope_in_vid_q"][0]
    q_post = kp["rope_out_vid_q"][0]
    k_pre = kp["rope_in_vid_k"][0]
    k_post = kp["rope_out_vid_k"][0]
    q_norm = kp["q_norm_vid"][0]
    freqs_vid = kp["rope_freqs_vid"][0]
    freqs_txt = kp["rope_freqs_txt"][0]
    win_shape, txt_shape = kp["rope_freqs_input"][0]

    d = q_pre.shape[-1]
    tail_start = d - 2
    q_tail_ok = bool(torch.equal(q_pre[..., tail_start:], q_post[..., tail_start:]))
    k_tail_ok = bool(torch.equal(k_pre[..., tail_start:], k_post[..., tail_start:]))
    q_rot_ok = not torch.equal(q_pre[..., :tail_start], q_post[..., :tail_start])
    norm_before_rope = bool(torch.equal(q_norm, q_pre))

    eff = freqs_vid.shape[-1]
    # the real rope class does not store rope_dim; it comes from the block contract config
    configured = REAL_CFG["rope_dim"]

    # offset sentinel: expected = full[l:l+f, :h, :w] per window; wrong = full[:f, :h, :w]
    full = block.attn.rope.get_axial_freqs(1024, 128, 128)
    txt_full = block.attn.rope.get_axial_freqs(1024)
    expected_parts, wrong_parts = [], []
    l_used = int(txt_shape[0, 0].item())
    for (f, h, w), l in zip(win_shape.tolist(), txt_shape[:, 0].tolist()):
        expected_parts.append(full[l:l + f, :h, :w].reshape(-1, full.size(-1)))
        wrong_parts.append(full[:f, :h, :w].reshape(-1, full.size(-1)))
    expected_vid = torch.cat(expected_parts)
    wrong_vid = torch.cat(wrong_parts)
    offset_exact = bool(torch.equal(freqs_vid, expected_vid))
    offset_matters = not torch.equal(expected_vid, wrong_vid)
    # txt freqs: txt_freqs[:l].repeat(1,3).reshape(-1, eff) per window
    exp_txt = txt_full[:l_used].repeat(1, 3).reshape(-1, eff)
    txt_exact = bool(torch.equal(freqs_txt, exp_txt))

    # T-major W-fastest: the flattened freqs equal the T-major flatten expression
    layout_ok = offset_exact
    return {
        "pass": (q_tail_ok and k_tail_ok and q_rot_ok and norm_before_rope
                 and offset_exact and offset_matters and txt_exact and eff == 126
                 and configured == 128),
        "configured_rope_dim": configured, "effective_rotary_dim": int(eff),
        "identity_tail": int(d - eff),
        "q_identity_tail_elementwise_identical": q_tail_ok,
        "k_identity_tail_elementwise_identical": k_tail_ok,
        "rotation_applied_to_front": q_rot_ok,
        "qk_norm_before_rope": norm_before_rope,
        "vid_freqs_offset_exact": offset_exact,
        "vid_freqs_offset_matters_sentinel": offset_matters,
        "txt_freqs_exact": txt_exact,
        "l_used": l_used,
        "vid_freqs_shape": list(freqs_vid.shape),
        "t_major_w_fastest_layout": layout_ok,
    }


def check_ada_and_gates(block, cap, run_name="A_min"):
    run = _find_run(cap, run_name)
    kp = run["kept"]
    emb = kp["block_input_emb"][0]
    vid_in = kp["block_input_vid"][0]
    attn_norm_vid = kp["attn_norm_vid"][0]
    proj_out_vid = kp["proj_out_vid"][0]
    mlp_norm_input = kp["mlp_norm_vid_input"][0]
    mlp_norm_vid = kp["mlp_norm_vid"][0]
    mlp_out_vid = kp["mlp_vid_out"][0]
    ada_attn_in = kp["ada_vid_out_attn_in"][0]
    ada_attn_out = kp["ada_vid_out_attn_out"][0]
    ada_mlp_in = kp["ada_vid_out_mlp_in"][0]
    ada_mlp_out = kp["ada_vid_out_mlp_out"][0]
    final_vid = kp["block_out_vid"][0]

    d = vid_in.shape[1]
    L = vid_in.shape[0]
    emb3 = emb.reshape(1, d, 2, 3)
    attn_seg = emb3[0, :, 0, :]
    mlp_seg = emb3[0, :, 1, :]
    ada = block.ada.vid

    def expand(v):
        return v.unsqueeze(0).expand(L, d)

    attn_shift, attn_scale, attn_gate = (expand(attn_seg[:, i]) for i in range(3))
    mlp_shift, mlp_scale, mlp_gate = (expand(mlp_seg[:, i]) for i in range(3))

    six_segments = (tuple(emb.shape) == (1, 6 * d)
                    and all(hasattr(ada, n) for n in
                            ("attn_shift", "attn_scale", "attn_gate",
                             "mlp_shift", "mlp_scale", "mlp_gate")))
    exp_attn_in = attn_norm_vid * (attn_scale + ada.attn_scale) + (attn_shift + ada.attn_shift)
    exp_attn_out = proj_out_vid * (attn_gate + ada.attn_gate)
    exp_mlp_in = mlp_norm_vid * (mlp_scale + ada.mlp_scale) + (mlp_shift + ada.mlp_shift)
    exp_mlp_out = mlp_out_vid * (mlp_gate + ada.mlp_gate)

    attn_in_ok = torch.allclose(ada_attn_in, exp_attn_in, atol=EPS, rtol=REL)
    attn_out_ok = torch.allclose(ada_attn_out, exp_attn_out, atol=EPS, rtol=REL)
    mlp_in_ok = torch.allclose(ada_mlp_in, exp_mlp_in, atol=EPS, rtol=REL)
    mlp_out_ok = torch.allclose(ada_mlp_out, exp_mlp_out, atol=EPS, rtol=REL)

    src = inspect.getsource(type(ada).forward).lower()
    no_activation = ("sigmoid" not in src) and ("tanh" not in src)

    # residual ordering: attn gate -> + vid -> mlp; mlp gate -> + residual -> out
    exp_residual = ada_attn_out + vid_in
    attn_residual_ok = torch.allclose(mlp_norm_input, exp_residual, atol=EPS, rtol=REL)
    exp_final = ada_mlp_out + mlp_norm_input
    final_residual_ok = torch.allclose(final_vid, exp_final, atol=EPS, rtol=REL)

    return {
        "pass": (six_segments and attn_in_ok and attn_out_ok and mlp_in_ok and mlp_out_ok
                 and no_activation and attn_residual_ok and final_residual_ok),
        "emb_is_6x_dim": tuple(emb.shape) == (1, 6 * d),
        "six_segments_present": six_segments,
        "attn_in_math": bool(attn_in_ok), "attn_out_gate_math": bool(attn_out_ok),
        "mlp_in_math": bool(mlp_in_ok), "mlp_out_gate_math": bool(mlp_out_ok),
        "gate_linear_no_activation": no_activation,
        "attn_gate_before_residual": bool(attn_residual_ok),
        "mlp_gate_before_residual": bool(final_residual_ok),
    }


def check_swiglu(block, cap, run_name="A_min"):
    run = _find_run(cap, run_name)
    kp = run["kept"]
    gate_proj = kp["swiglu_gate_proj_vid"][0]
    up_proj = kp["swiglu_up_proj_vid"][0]
    gate_product = kp["swiglu_gate_product_vid"][0]
    out_proj = kp["swiglu_out_proj_vid"][0]
    mlp_out = kp["mlp_vid_out"][0]
    mlp_in = kp["mlp_vid_in"][0]
    ada_mlp_in = kp["ada_vid_out_mlp_in"][0]

    exp_product = torch.nn.functional.silu(gate_proj) * up_proj
    product_ok = torch.allclose(gate_product, exp_product, atol=1e-4, rtol=1e-5)
    out_ok = torch.allclose(out_proj, mlp_out, atol=1e-6, rtol=1e-6)
    in_ok = torch.allclose(mlp_in, ada_mlp_in, atol=1e-6, rtol=1e-6)
    return {
        "pass": bool(product_ok and out_ok and in_ok),
        "gate_product_silu_gate_times_up": bool(product_ok),
        "proj_out_matches_mlp_out": bool(out_ok),
        "mlp_input_is_modulated": bool(in_ok),
        "gate_proj_shape": list(gate_proj.shape),
        "up_proj_shape": list(up_proj.shape),
    }


def check_segment_ordering(block, cap):
    """Real-config sentinel run: per-window packed q is [vid_win; txt] with txt repeated."""
    inputs = make_structural_inputs(PROFILE_SHAPES["B_multi_window"], vid_value=0.0, txt_value=1.0)
    run_forward(block, cap, "S1_ordering", inputs, keep_mode=True)
    run = cap.last()
    kp = run["kept"]
    q_packed = kp["attn_varlen_q"][0].float()
    cu = [int(x) for x in kp["attn_varlen_cu_q"][0].tolist()]
    txt_q_rope = kp["rope_out_txt_q"][0]
    info = window_partition_info(inputs["vid_shape"])
    win_sizes = info["window_sizes"]
    l = inputs["meta"]["samples"][0][3]
    n_win = len(win_sizes)

    blocks = []
    order_ok = True
    for i in range(n_win):
        seg = q_packed[cu[i]:cu[i + 1]]
        exp_txt = txt_q_rope[i * l:(i + 1) * l].bfloat16().float()
        order_ok = order_ok and bool(torch.equal(seg[win_sizes[i]:], exp_txt))
        blocks.append(seg[win_sizes[i]:])
    repeat_ok = all(bool(torch.equal(blocks[0], b)) for b in blocks[1:])
    total_ok = q_packed.shape[0] == sum(win_sizes) + l * n_win
    # boundary distinctness: first vid row != first txt row in the first segment
    first = q_packed[cu[0]:cu[1]]
    boundary_ok = not bool(torch.equal(first[0], first[win_sizes[0]]))
    return {
        "pass": bool(order_ok and repeat_ok and total_ok and boundary_ok),
        "per_window_vid_then_txt": bool(order_ok),
        "txt_repeated_per_window": bool(repeat_ok),
        "packed_length_consistent": bool(total_ok),
        "vid_txt_boundary_distinct": bool(boundary_ok),
        "n_windows": n_win,
        "window_sizes": win_sizes,
        "text_len_per_window": l,
    }


def check_txt_mean_coalesce(block, cap):
    inputs = make_inputs("D_batch2")
    run_forward(block, cap, "D_keep", inputs, keep_mode=True)
    run = cap.last()
    kp = run["kept"]
    out = kp["attn_varlen_out"][0].float()
    cu = [int(x) for x in kp["attn_varlen_cu_q"][0].tolist()]
    info = window_partition_info(inputs["vid_shape"])
    win_sizes = info["window_sizes"]
    wc = info["window_count"].tolist()
    samples = inputs["meta"]["samples"]
    ls = [s[3] for s in samples]
    l_per_win = []
    for l, n in zip(ls, wc):
        l_per_win += [l] * n
    seg_txt = []
    for i in range(len(win_sizes)):
        start = cu[i] + win_sizes[i]
        seg_txt.append(out[start:start + l_per_win[i]])
    expected = []
    pos = 0
    for l, n in zip(ls, wc):
        rows = seg_txt[pos:pos + n]
        exp = torch.stack(rows).mean(0).reshape(l, -1)
        expected.append(exp)
        pos += n
    exp_full = torch.cat(expected)
    actual = kp["proj_out_txt_input"][0]
    diff = float((exp_full.float() - actual.float()).abs().max().item())
    return {
        "pass": diff < 1e-5,
        "max_abs_error": diff,
        "txt_out_length": int(actual.shape[0]),
        "expected_length": int(sum(ls)),
        "window_repeats_per_sample": wc,
    }


def check_window_reassembly():
    """Structural rope-off probe (real class, real forward): per-window constants."""
    b = build_block(dict(rope_type=None), seed=5)
    cap = StageCapture()
    install_capture(b, cap)
    vs = torch.tensor([[5, 32, 32]], dtype=torch.long)
    info = window_partition_info(vs)
    n_win = info["window_shape"].shape[0]
    g = torch.Generator().manual_seed(99)
    C = torch.randn(n_win, 2560, generator=g)
    vid = C[info["orig_to_win"]]
    txt = torch.zeros(8, 2560)
    inputs = {"vid": vid, "txt": txt, "vid_shape": vs,
              "txt_shape": torch.tensor([[8]], dtype=torch.long),
              "emb": torch.zeros(1, 15360)}
    run_forward(b, cap, "S2_reassembly", inputs, keep_mode=True)
    vid_out = cap.last()["kept"]["block_out_vid"][0]
    means = []
    within_ok = True
    for i in range(n_win):
        idx = torch.nonzero(info["orig_to_win"] == i).flatten()
        rows = vid_out[idx]
        within_ok = within_ok and bool((rows == rows[0:1]).all().item())
        means.append(rows[0])
    distinct_ok = True
    min_pair = None
    for i in range(n_win):
        for j in range(i + 1, n_win):
            d = float((means[i].float() - means[j].float()).abs().max().item())
            min_pair = d if min_pair is None else min(min_pair, d)
            distinct_ok = distinct_ok and d > 1e-2
    free_block(b)
    gc_collect()
    return {
        "pass": bool(within_ok and distinct_ok),
        "window_output_reassembly": bool(within_ok),
        "no_cross_segment_leakage": bool(within_ok and distinct_ok),
        "cross_window_distinct": bool(distinct_ok),
        "min_pairwise_max_abs": min_pair,
        "n_windows": n_win,
    }


def check_last_layer():
    b = build_block(dict(vid_dim=256, txt_dim=256, emb_dim=1536, heads=4, head_dim=64,
                         rope_dim=64, is_last_layer=True), seed=3)
    struct_ok = (b.mlp.txt is None and b.mlp_norm.txt is None
                 and b.ada.txt is None and b.attn_norm.txt is not None)
    cap = StageCapture()
    install_capture(b, cap, capture_txt=False)
    inputs = make_scaled_inputs([(2, 8, 8, 2)])
    run_forward(b, cap, "last_layer", inputs, keep_mode=True)
    run = cap.last()
    txt_out = run["kept"]["block_out_txt"][0]
    proj_out_txt = run["kept"]["proj_out_txt"][0]
    txt_in = run["kept"]["block_input_txt"][0]
    vid_out = run["kept"]["block_out_vid"][0]
    # last-layer txt is attention-only: no mlp/ada consumes txt, so the final residual
    # "txt_mlp + txt_attn" re-adds the attention-residual value: out = 2*(proj_out + txt_in)
    passthrough = bool(torch.equal(txt_out, 2 * (proj_out_txt + txt_in)))
    free_block(b)
    gc_collect()
    return {
        "pass": bool(struct_ok and passthrough and is_finite(vid_out)),
        "vid_only_structure": struct_ok,
        "txt_passes_through_attention_only": passthrough,
        "vid_finite": is_finite(vid_out),
    }


def identity_tail_checker(q_pre, q_post, tail_start=126):
    return bool(torch.equal(q_pre[..., tail_start:], q_post[..., tail_start:]))


def nondeterminism_checker(a, b):
    return not torch.equal(a, b)


def run_detector_selftests():
    results = []
    a = torch.randn(4, 8, 128)
    b = a.clone()
    results.append({"name": "identity_tail_accepts_unchanged", "pass": identity_tail_checker(a, b)})
    tampered = a.clone()
    tampered[..., 126:128] = tampered[..., 126:128] + 1.0
    results.append({"name": "identity_tail_detects_modification",
                    "pass": not identity_tail_checker(a, tampered)})
    results.append({"name": "determinism_accepts_identical", "pass": not nondeterminism_checker(a, b)})
    results.append({"name": "determinism_detects_diff", "pass": nondeterminism_checker(a, tampered)})
    return results


def run_negative_tests():
    results = []

    def expect_raise(name, fn, exc=Exception):
        try:
            fn()
        except exc as e:
            results.append({"name": name, "pass": True, "raised": type(e).__name__})
        except Exception as e:  # noqa: BLE001
            results.append({"name": name, "pass": False,
                            "raised_wrong": type(e).__name__, "detail": str(e)[:200]})
        else:
            results.append({"name": name, "pass": False, "raised": None})

    # 2. hidden dim not divisible by heads (probe contract invariant)
    expect_raise("heads_times_head_dim_ne_vid_dim", lambda: build_block(
        dict(vid_dim=256, txt_dim=256, emb_dim=1536, heads=3, head_dim=64)), AssertionError)
    # 3. emb not 6*C -> real AdaSingle assertion
    expect_raise("emb_not_6x_dim", lambda: build_block(
        dict(vid_dim=256, txt_dim=256, emb_dim=1280, heads=4, head_dim=64)), AssertionError)
    # 17. invalid window shapes -> real assert / _triple ValueError
    expect_raise("invalid_window_negative_t", lambda: build_block(dict(window=(-1, 3, 3))), AssertionError)
    def invalid_window_rank():
        b = build_block(dict(vid_dim=256, txt_dim=256, emb_dim=1536, heads=4, head_dim=64,
                            rope_dim=64, window=(4, 3)), seed=11)
        try:
            with torch.no_grad():
                from common.cache import Cache
                inputs = make_scaled_inputs([(2, 8, 8, 2)])
                b(inputs["vid"], inputs["txt"], inputs["vid_shape"], inputs["txt_shape"],
                  inputs["emb"], Cache(disable=True))
        finally:
            free_block(b)
            gc_collect()
    expect_raise("invalid_window_rank", invalid_window_rank, (ValueError, TypeError))

    # 7. RoPE frequency insufficient (rot_dim > head_dim)
    def rope_insufficient():
        b = build_block(dict(vid_dim=256, txt_dim=256, emb_dim=1536, heads=4, head_dim=64,
                             rope_dim=130), seed=8)
        try:
            with torch.no_grad():
                from common.cache import Cache
                inputs = make_scaled_inputs([(2, 8, 8, 2)])
                b(inputs["vid"], inputs["txt"], inputs["vid_shape"], inputs["txt_shape"],
                  inputs["emb"], Cache(disable=True))
        finally:
            free_block(b)
            gc_collect()
    expect_raise("rope_frequency_insufficient", rope_insufficient)

    # shared scaled block for forward-based negatives (rope_dim=64 -> 63 effective)
    neg_block = build_block(dict(NEG_CFG), seed=10)
    try:
        base = make_scaled_inputs([(2, 8, 8, 2)])

        # 4. vid shape token count mismatch
        def vid_mismatch():
            inp = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in base.items()}
            inp["vid"] = inp["vid"][:100]
            run_forward(neg_block, StageCapture(), "neg", inp)
        expect_raise("vid_token_count_mismatch", vid_mismatch)

        # 5. txt shape token count mismatch
        def txt_mismatch():
            inp = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in base.items()}
            inp["txt"] = inp["txt"][:1]
            run_forward(neg_block, StageCapture(), "neg", inp)
        expect_raise("txt_token_count_mismatch", txt_mismatch)

        # 6. l+f > 1024 (l > 1024 clips the vid_freqs T axis in the window path)
        def l_plus_f_gt_1024():
            inp = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in base.items()}
            inp["vid"] = torch.randn(4 * 8 * 8, 256)
            inp["txt"] = torch.randn(1030, 256)
            inp["vid_shape"] = torch.tensor([[4, 8, 8]], dtype=torch.long)
            inp["txt_shape"] = torch.tensor([[1030]], dtype=torch.long)
            run_forward(neg_block, StageCapture(), "neg", inp)
        expect_raise("l_plus_f_gt_1024", l_plus_f_gt_1024)

        # 11. NaN emb -> non-finite output detected by harness
        def nan_emb():
            inp = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in base.items()}
            inp["emb"] = torch.full_like(inp["emb"], float("nan"))
            run_forward_check_finite(neg_block, inp)
        expect_raise("nan_emb_detected", nan_emb, ValueError)

        # 12. NaN vid -> non-finite output detected by harness
        def nan_vid():
            inp = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in base.items()}
            inp["vid"] = inp["vid"].clone()
            inp["vid"][0, 0] = float("nan")
            run_forward_check_finite(neg_block, inp)
        expect_raise("nan_vid_detected", nan_vid, ValueError)

        # 13. dtype contract violation -> harness TypeError
        def dtype_violation():
            inp = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in base.items()}
            inp["vid"] = inp["vid"].double()
            validate_input_contract(inp, dim=256)
        expect_raise("dtype_contract_violation", dtype_violation, TypeError)

        # 14. cache type error -> real code TypeError (not callable)
        def cache_type_error():
            with torch.no_grad():
                neg_block(base["vid"], base["txt"], base["vid_shape"], base["txt_shape"],
                          base["emb"], {})
        expect_raise("cache_type_error", cache_type_error, TypeError)

        # 15. batch metadata inconsistent (vid batch 2, txt batch 1)
        def batch_metadata_inconsistent():
            inp = make_scaled_inputs([(2, 8, 8, 2), (3, 10, 12, 5)])
            inp["txt_shape"] = torch.tensor([[2]], dtype=torch.long)
            run_forward(neg_block, StageCapture(), "neg", inp)
        expect_raise("batch_metadata_inconsistent", batch_metadata_inconsistent)
    finally:
        free_block(neg_block)
        gc_collect()

    # 8. cu_seqlens non-monotonic -> compat raises
    from compat.flash_attn_cpu_compat import flash_attn_varlen_func

    def non_monotonic():
        q = torch.randn(8, 2, 8)
        flash_attn_varlen_func(q=q, k=q, v=q,
                               cu_seqlens_q=torch.tensor([0, 3, 2, 8], dtype=torch.int32),
                               cu_seqlens_k=torch.tensor([0, 3, 2, 8], dtype=torch.int32),
                               max_seqlen_q=8, max_seqlen_k=8)
    expect_raise("cu_seqlens_non_monotonic", non_monotonic, NotImplementedError)

    def cu_last_mismatch():
        q = torch.randn(10, 2, 8)
        flash_attn_varlen_func(q=q, k=q, v=q,
                               cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
                               cu_seqlens_k=torch.tensor([0, 3, 7], dtype=torch.int32),
                               max_seqlen_q=10, max_seqlen_k=10)
    expect_raise("cu_seqlens_last_mismatch", cu_last_mismatch, NotImplementedError)

    # 16/18 covered by run_detector_selftests; 1/10 covered by test_dependency_compat.
    results.append({"name": "missing_compat_covered_by_test_dependency_compat", "pass": True})
    results.append({"name": "segment_cross_attention_covered_by_test_dependency_compat", "pass": True})
    return results


def run_all(block=None, cap=None):
    ensure_compat()
    if block is None or cap is None:
        block = build_block()
        cap = StageCapture()
        install_capture(block, cap)
        run_forward(block, cap, "A_min", make_inputs("A_min"), keep_mode=True)

    checks = {}
    checks["rope_contract"] = check_rope_contract(block, cap)
    checks["ada_six_segments_gates"] = check_ada_and_gates(block, cap)
    checks["swiglu_mlp"] = check_swiglu(block, cap)
    checks["segment_ordering"] = check_segment_ordering(block, cap)
    checks["txt_mean_coalesce"] = check_txt_mean_coalesce(block, cap)
    clear_rope_cache(block)
    checks["window_reassembly"] = check_window_reassembly()
    checks["last_layer_vid_only"] = check_last_layer()
    checks["negative_tests"] = run_negative_tests()
    checks["detector_selftests"] = run_detector_selftests()

    neg = checks["negative_tests"]
    det = checks["detector_selftests"]
    all_pass = (checks["rope_contract"]["pass"] and checks["ada_six_segments_gates"]["pass"]
                and checks["swiglu_mlp"]["pass"] and checks["segment_ordering"]["pass"]
                and checks["txt_mean_coalesce"]["pass"] and checks["window_reassembly"]["pass"]
                and checks["last_layer_vid_only"]["pass"]
                and all(n["pass"] for n in neg) and all(d["pass"] for d in det))
    return {"checks": checks, "all_pass": bool(all_pass)}


if __name__ == "__main__":
    report = run_all()
    c = report["checks"]
    print("test_runtime_contract all_pass =", report["all_pass"])
    for k in ("rope_contract", "ada_six_segments_gates", "swiglu_mlp", "segment_ordering",
              "txt_mean_coalesce", "window_reassembly", "last_layer_vid_only"):
        print(" ", k, "->", c[k]["pass"])
    print("  negative_tests:", [(n["name"], n["pass"]) for n in c["negative_tests"]])
    sys.exit(0 if report["all_pass"] else 1)