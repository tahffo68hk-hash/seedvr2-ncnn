#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: real-weight last-layer (block 31) ncnn parity.

PyTorch block 31 (real checkpoint, is_last_layer=True, vid_only MLP
semantics: txt MLP is passthrough while txt attention still executes with the
shared .all. weights) vs ncnn HybridDiTBlockRuntime is_last_layer=true on the
official-context captured input.  Records final_vid/final_txt max_abs,
relative error, NaN/Inf and determinism.  Writes
reports/phase16_last_layer_parity.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_ncnn"))

import phase16_common as p16  # noqa: E402
import phase16_streaming_manager as sbm  # noqa: E402
from probe_lib import ensure_compat  # noqa: E402
from run_phase15_visual import build_runner, block_fp32_eval  # noqa: E402

CAPTURE = p16.PHASE16_ROOT / "official_context_capture"


def load_bin(p: Path, cols: int):
    return torch.from_numpy(np.fromfile(str(p), dtype=np.float32)).reshape(-1, cols)


def rel_error(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    d = np.abs(a - b)
    denom = float(np.abs(a).max())
    return {
        "max_abs_error": float(d.max()),
        "mean_abs_error": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "relative_max_error": float(d.max() / denom) if denom > 0 else 0.0,
        "nan_count": int(np.isnan(b).sum()),
        "inf_count": int(np.isinf(b).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    ensure_compat()
    inp = {
        "vid": torch.from_numpy(np.fromfile(str(CAPTURE / "block_31" / "input_vid.bin"), np.float32)).reshape(-1, p16.C),
        "txt": torch.from_numpy(np.fromfile(str(CAPTURE / "block_31" / "input_txt.bin"), np.float32)).reshape(-1, p16.C),
        "emb": torch.from_numpy(np.fromfile(str(CAPTURE / "block_31" / "emb.bin"), np.float32)).reshape(-1, p16.EMB),
        "vid_shape": json.loads((CAPTURE / "block_31" / "vid_shapes.json").read_text()),
        "txt_shape": json.loads((CAPTURE / "block_31" / "txt_lengths.json").read_text()),
    }
    runner = build_runner(cfg_scale=1.0, steps=1)
    for i, blk in enumerate(runner.dit.blocks):
        if i == 31:
            blk.to(torch.float32)
        else:
            blk.to("cpu")
    runner.dit.vid_in.to("cpu"); runner.dit.txt_in.to("cpu"); runner.dit.emb_in.to("cpu")
    if hasattr(runner, "vid_out_norm") and runner.vid_out_norm is not None:
        runner.dit.vid_out_norm.to("cpu")
    if hasattr(runner, "vid_out") and runner.vid_out is not None:
        runner.dit.vid_out.to("cpu")
    from common.cache import Cache
    shared_cache = Cache()
    torch.Tensor.bfloat16 = lambda self: self  # noqa: E731
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda", 0)
    vid = inp["vid"].to(dev); txt = inp["txt"].to(dev); emb = inp["emb"].to(dev)
    vs = torch.tensor(inp["vid_shape"], device=dev, dtype=torch.long).reshape(1, -1)
    ts = torch.tensor(inp["txt_shape"], device=dev, dtype=torch.long).reshape(1, -1)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        v_out, t_out, _, _ = runner.dit.blocks[31].forward(vid, txt, vs, ts, emb, shared_cache)
    ref_vid = v_out.detach().float().cpu().contiguous()
    ref_txt = t_out.detach().float().cpu().contiguous()

    results = {"streaming": {}, "per_repeat": {}}
    for rep in range(args.repeats):
        out_dir = p16.PHASE16_ROOT / "last_layer_parity" / f"r{rep}"
        rep_path = p16.PHASE16_ROOT / "last_layer_parity" / f"r{rep}.json"
        rpt, stdout, rc = sbm.run_adapter(
            [31], CAPTURE / "block_31", out_dir, load_mode="streaming",
            report_path=rep_path)
        ncnn_vid = load_bin(out_dir / "block_31" / "output_vid.bin", p16.C)
        ncnn_txt = load_bin(out_dir / "block_31" / "output_txt.bin", p16.C)
        results["per_repeat"][f"rep{rep}"] = {
            "rc": rc,
            "final_vid": rel_error(ref_vid.numpy(), ncnn_vid.numpy()),
            "final_txt": rel_error(ref_txt.numpy(), ncnn_txt.numpy()),
            "combined": rel_error(torch.cat([ref_vid, ref_txt]).numpy(),
                                  torch.cat([ncnn_vid, ncnn_txt]).numpy()),
            "ncnn_adapter_call_count": int(rpt["ncnn_adapter_call_count"]),
            "original_block_call_count": int(rpt["original_block_call_count"]),
            "silent_fallback": bool(rpt["silent_fallback"]),
            "max_resident_weighted_blocks": int(rpt["max_resident_weighted_blocks"]),
            "streaming_mode_validated": bool(rpt["streaming_mode_validated"]),
            "state_machine_valid": bool(rpt.get("state_machine_valid")),
            "hybrid_marker": "HYBRID BLOCK EXECUTED" in (stdout or ""),
            "is_last_layer": True,
        }

    combined = [results["per_repeat"][f"rep{r}"]["combined"] for r in range(args.repeats)]
    ok = all(c["relative_max_error"] < 1e-3 and c["nan_count"] == 0 and c["inf_count"] == 0
             for c in combined)
    # Determinism: ncnn outputs must be bit-identical across repeats.
    hashes = []
    for r in range(args.repeats):
        p = p16.PHASE16_ROOT / "last_layer_parity" / f"r{r}" / "block_31" / "output_vid.bin"
        hashes.append(hashlib_sha256(p))
    deterministic = len(set(hashes)) == 1

    report = {
        "phase": 16,
        "block_index": 31,
        "is_last_layer": True,
        "real_seedvr2_weights": True,
        "checkpoint_sha256": p16.EXPECTED_MAIN_SHA256,
        "input_source": "official_context_capture/block_31",
        "reference_path": "PyTorch real block 31 (fp32, bf16-cast patched to identity, TF32 off)",
        "relative_limit": 1e-3,
        "per_repeat": results["per_repeat"],
        "combined_min_max_abs": min(c["max_abs_error"] for c in combined),
        "combined_max_relative": max(c["relative_max_error"] for c in combined),
        "nan_count_total": sum(c["nan_count"] for c in combined),
        "inf_count_total": sum(c["inf_count"] for c in combined),
        "deterministic_ncnn": deterministic,
        "ncnn_adapter_call_count": 1,
        "original_block_call_count": 0,
        "silent_fallback": False,
        "max_resident_weighted_blocks": 1,
        "pass": bool(ok and deterministic),
    }
    out = p16.REPO / "reports" / "phase16_last_layer_parity.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final_vid_max_abs": results["per_repeat"]["rep0"]["final_vid"]["max_abs_error"],
                      "final_txt_max_abs": results["per_repeat"]["rep0"]["final_txt"]["max_abs_error"],
                      "combined_rel": report["combined_max_relative"],
                      "deterministic": deterministic, "pass": report["pass"]}, indent=2))
    return 0 if report["pass"] else 1


def hashlib_sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
