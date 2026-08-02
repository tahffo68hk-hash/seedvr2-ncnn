"""Phase 11.1 real-block forward tests.

Imports the REAL NaMMSRTransformerBlock from third_party/SeedVR2, instantiates
it with the real 3B config (hidden_dim=2560, heads=20, head_dim=128,
mlp_hidden=6912, window=(4,3,3)), and runs real random-weight forwards for
profiles A (real width, minimal), B (multi-window), C (asymmetric
spatiotemporal) and D (batch=2).  CPU FP32 eval()/no_grad(), fixed seed,
random init weights, real_seedvr2_weights=false.  Verifies output shapes,
finiteness, input contract and determinism.
"""
from __future__ import annotations

import inspect
import sys

import torch

from probe_lib import (REAL_CFG, SEEDVR2_ROOT, StageCapture, build_block,
                       ensure_compat, install_capture, input_stats, is_finite,
                       make_inputs, run_forward)

EXIT_PASS = 0
EXIT_FAIL = 1


def check_real_class(block):
    cls = type(block)
    src = inspect.getsourcefile(cls) or ""
    checks = {
        "class_name": cls.__name__ == "NaMMSRTransformerBlock",
        "module": cls.__module__ == "models.dit_v2.nablocks.mmsr_block",
        "source_under_seedvr2": src.replace("\\", "/").startswith(SEEDVR2_ROOT.replace("\\", "/")),
        "source_file": src,
    }
    sig = inspect.signature(cls.forward)
    params = list(sig.parameters)
    checks["forward_signature"] = params == ["self", "vid", "txt", "vid_shape", "txt_shape", "emb", "cache"]
    return checks


def check_config_dims(block):
    cfg = REAL_CFG
    n_params = sum(p.numel() for p in block.parameters())
    return {
        "vid_dim": cfg["vid_dim"], "txt_dim": cfg["txt_dim"], "emb_dim": cfg["emb_dim"],
        "heads": cfg["heads"], "head_dim": cfg["head_dim"],
        "inner_dim_eq_c": cfg["heads"] * cfg["head_dim"] == cfg["vid_dim"],
        "emb_is_6c": cfg["emb_dim"] == 6 * cfg["vid_dim"],
        "window": list(cfg["window"]),
        "mlp_hidden_derived": 6912,
        "mlp_hidden_actual_vid": block.mlp.vid.proj_in.out_features,
        "params": int(n_params),
    }


def run_profile(block, cap, name):
    inputs = make_inputs(name)
    vid_out, txt_out, vs, ts = run_forward(block, cap, name, inputs,
                                           keep_mode=(name == "A_min"))
    exp_vid = inputs["meta"]["vid_tokens"]
    exp_txt = inputs["meta"]["txt_tokens"]
    return {
        "name": name,
        "pass": (tuple(vid_out.shape) == (exp_vid, 2560)
                 and tuple(txt_out.shape) == (exp_txt, 2560)
                 and is_finite(vid_out) and is_finite(txt_out)
                 and vs.tolist() == inputs["vid_shape"].tolist()
                 and ts.tolist() == inputs["txt_shape"].tolist()),
        "vid_out_shape": list(vid_out.shape), "txt_out_shape": list(txt_out.shape),
        "vid_shape_out": vs.tolist(), "txt_shape_out": ts.tolist(),
        "vid_finite": is_finite(vid_out), "txt_finite": is_finite(txt_out),
        "inputs": {k: (input_stats(v) if isinstance(v, torch.Tensor) else v)
                   for k, v in inputs.items() if k != "meta"},
        "meta": inputs["meta"],
    }


def check_determinism(block, cap):
    inputs = make_inputs("A_min")
    v1, t1, _, _ = run_forward(block, cap, "det1", inputs, keep_mode=False)
    v2, t2, _, _ = run_forward(block, cap, "det2", inputs, keep_mode=False)
    return {
        "pass": torch.equal(v1, v2) and torch.equal(t1, t2),
        "vid_bit_identical": bool(torch.equal(v1, v2)),
        "txt_bit_identical": bool(torch.equal(t1, t2)),
        "max_abs_diff": float((v1.float() - v2.float()).abs().max().item()),
    }


def run_all(block=None):
    ensure_compat()
    cap = StageCapture()
    if block is None:
        block = build_block()
    checks = {}
    checks["real_class"] = check_real_class(block)
    checks["config_dims"] = check_config_dims(block)
    install_capture(block, cap)
    profiles = []
    for name in ("A_min", "B_multi_window", "C_asymmetric", "D_batch2"):
        profiles.append(run_profile(block, cap, name))
    checks["profiles"] = profiles
    checks["determinism"] = check_determinism(block, cap)
    ok = (checks["real_class"]["class_name"] and checks["real_class"]["module"]
          and checks["real_class"]["source_under_seedvr2"]
          and checks["real_class"]["forward_signature"]
          and checks["config_dims"]["inner_dim_eq_c"] and checks["config_dims"]["emb_is_6c"]
          and all(p["pass"] for p in profiles)
          and checks["determinism"]["pass"])
    checks["all_pass"] = ok
    return {"checks": checks, "cap": cap, "all_pass": ok}


if __name__ == "__main__":
    report = run_all()
    c = report["checks"]
    print("test_real_block_forward all_pass =", report["all_pass"])
    print("  class/module/source/signature:", c["real_class"]["class_name"], c["real_class"]["module"],
          c["real_class"]["source_under_seedvr2"], c["real_class"]["forward_signature"])
    print("  profiles:", [(p["name"], p["pass"]) for p in c["profiles"]])
    print("  determinism:", c["determinism"]["pass"])
    sys.exit(EXIT_PASS if report["all_pass"] else EXIT_FAIL)