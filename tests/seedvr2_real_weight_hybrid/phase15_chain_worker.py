#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: chain parity worker (one chain per process, GPU-memory isolated).

Runs in the .conda-cuda interpreter.  Loads the captured official-context
input for `--input-block`, computes the FP32 PyTorch chain reference for
`--blocks` (bf16 cast patched to identity, TF32 off), runs the
multiblock_streaming_adapter for streaming and resident mode `--repeats`
times, and writes `<out>/chain_<n>_result.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEEDVR2 = Path(r"D:\AIProjects\third_party\SeedVR2")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SEEDVR2))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_contract"))
sys.path.insert(0, str(HERE.parent / "seedvr2_dit_block_ncnn"))

import phase15_common as p15  # noqa: E402
import streaming_block_manager as sbm  # noqa: E402
from probe_lib import ensure_compat  # noqa: E402
from run_official_pipeline import build_runner  # noqa: E402

CAPTURE = p15.PHASE15_ROOT / "official_context_capture"
PARITY_OUT = p15.PHASE15_ROOT / "chain_parity"


def chain_error(ref, hyb):
    a = ref.detach().cpu().float().numpy().reshape(-1)
    b = hyb.detach().cpu().float().numpy().reshape(-1)
    d = np.abs(a - b)
    max_abs = float(d.max())
    mean_abs = float(d.mean())
    rmse = float(np.sqrt((d ** 2).mean()))
    denom = float(np.abs(a).max())
    rel = float(max_abs / denom) if denom > 0 else 0.0
    return {
        "max_abs_error": max_abs, "mean_abs_error": mean_abs, "rmse": rmse,
        "relative_max_error": rel, "nan_count": int(np.isnan(b).sum()),
        "inf_count": int(np.isinf(b).sum()),
    }


def load_bin(p: Path, cols: int):
    return torch.from_numpy(np.fromfile(str(p), dtype=np.float32)).reshape(-1, cols)


def fp32_chain(block_indices, first_input):
    dev = torch.device("cuda", 0)
    runner = build_runner(cfg_scale=1.0, steps=1)
    # Keep ONLY the target blocks on GPU in FP32; offload everything else to
    # CPU so the transient MM-RoPE3D frequency grid (~8.4 GiB FP32) fits.
    for i, blk in enumerate(runner.dit.blocks):
        if i in p15.BLOCK_SET:
            blk.to(torch.float32).to(dev)
        else:
            blk.to("cpu")
    runner.dit.vid_in.to("cpu"); runner.dit.txt_in.to("cpu"); runner.dit.emb_in.to("cpu")
    if hasattr(runner, "vid_out_norm") and runner.vid_out_norm is not None:
        runner.dit.vid_out_norm.to("cpu")
    if hasattr(runner, "vid_out") and runner.vid_out is not None:
        runner.dit.vid_out.to("cpu")
    torch.Tensor.bfloat16 = lambda self: self  # noqa: E731
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from common.cache import Cache
    vid = first_input["vid"].to(dev)
    txt = first_input["txt"].to(dev)
    emb = first_input["emb"].to(dev)
    vs = torch.tensor(first_input["vid_shape"], device=dev, dtype=torch.long).reshape(1, -1)
    ts = torch.tensor(first_input["txt_shape"], device=dev, dtype=torch.long).reshape(1, -1)
    refs = []
    shared_cache = Cache()  # official pipeline semantics: freqs/windows computed once
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for bi in block_indices:
            out = runner.dit.blocks[bi].forward(vid, txt, vs, ts, emb, shared_cache)
            refs.append({"vid": out[0].detach().float().cpu().contiguous(),
                         "txt": out[1].detach().float().cpu().contiguous()})
            vid = out[0].detach().to(dev).contiguous()
            txt = out[1].detach().to(dev).contiguous()
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--input-block", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ensure_compat()
    block_indices = [int(x) for x in args.blocks.split(",")]
    bi0 = args.input_block
    _vid = torch.from_numpy(np.fromfile(str(CAPTURE / f"block_{bi0}" / "input_vid.bin"), np.float32))
    _txt = torch.from_numpy(np.fromfile(str(CAPTURE / f"block_{bi0}" / "input_txt.bin"), np.float32))
    _emb = torch.from_numpy(np.fromfile(str(CAPTURE / f"block_{bi0}" / "emb.bin"), np.float32))
    first = {
        "vid": _vid.reshape(-1, p15.C).contiguous(),
        "txt": _txt.reshape(-1, p15.C).contiguous(),
        "emb": _emb.reshape(-1, p15.EMB).contiguous(),
        "vid_shape": json.loads((CAPTURE / f"block_{bi0}" / "vid_shapes.json").read_text()),
        "txt_shape": json.loads((CAPTURE / f"block_{bi0}" / "txt_lengths.json").read_text()),
    }
    t0 = time.time()
    print('before chain: allocated=%.2fGB' % (torch.cuda.memory_allocated()/1e9), flush=True)
    refs = fp32_chain(block_indices, first)
    print('after chain: allocated=%.2fGB' % (torch.cuda.memory_allocated()/1e9), flush=True)
    ref_elapsed = time.time() - t0

    result = {"chain": args.chain, "range": block_indices, "input_block": bi0,
              "ref_elapsed_s": round(ref_elapsed, 3), "streaming": {}, "resident": {},
              "final_error": [], "per_block_error": {}}
    for rep in range(args.repeats):
        for mode in ("streaming", "resident"):
            out_dir = PARITY_OUT / f"{args.chain}_{mode}_r{rep}"
            rep_path = PARITY_OUT / f"{args.chain}_{mode}_r{rep}.json"
            rpt, stdout, rc = sbm.run_adapter(block_indices, CAPTURE / f"block_{bi0}", out_dir,
                                              load_mode=mode, report_path=rep_path)
            result[mode][f"rep{rep}"] = {"rc": rc, "report": rpt}
    for rep in range(args.repeats):
        if f"rep{rep}" not in result["streaming"]:
            continue
        out_dir = PARITY_OUT / f"{args.chain}_streaming_r{rep}"
        final_vid = load_bin(out_dir / "final_vid.bin", p15.C)
        final_txt = load_bin(out_dir / "final_txt.bin", p15.C)
        result["final_error"].append(chain_error(torch.cat([refs[-1]["vid"], refs[-1]["txt"]]),
                                                 torch.cat([final_vid, final_txt])))
        blk_errs = []
        for j, bi in enumerate(block_indices):
            out_vid = load_bin(out_dir / f"block_{bi}" / "output_vid.bin", p15.C)
            out_txt = load_bin(out_dir / f"block_{bi}" / "output_txt.bin", p15.C)
            blk_errs.append({"block_index": bi, **chain_error(
                torch.cat([refs[j]["vid"], refs[j]["txt"]]),
                torch.cat([out_vid, out_txt]))})
        result["per_block_error"][rep] = blk_errs
    limit = p15.CHAIN_REL_LIMITS[len(block_indices)]
    result["rel_limit"] = limit
    result["pass"] = bool(result["final_error"]) and all(
        e["relative_max_error"] < limit and e["nan_count"] == 0 and e["inf_count"] == 0
        for e in result["final_error"])
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.chain}: pass={result['pass']} final_rel="
          f"{result['final_error'][0]['relative_max_error'] if result['final_error'] else None} limit={limit}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
