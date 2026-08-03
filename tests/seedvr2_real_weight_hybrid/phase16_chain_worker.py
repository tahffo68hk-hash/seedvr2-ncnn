#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: chain parity worker (one chain per process, GPU-memory isolated).

Runs in the .conda-cuda interpreter.  Loads the captured official-context
input for `--input-block`, computes the FP32 PyTorch chain reference for
`--blocks` (bf16 cast patched to identity, TF32 off) using a sequential
move-to-GPU strategy (only ONE weighted block resident on GPU at a time, so
the full 32-block chain fits inside 16 GB VRAM together with the shared
MM-RoPE3D grid), then runs the Phase 16 full_dit_streaming_adapter
(streaming mode, max resident = 1) `--repeats` times and writes per-block and
final chain errors.
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

import phase16_common as p16  # noqa: E402
import phase16_streaming_manager as sbm  # noqa: E402
from probe_lib import ensure_compat  # noqa: E402
from run_official_pipeline import build_runner  # noqa: E402

CAPTURE = p16.PHASE16_ROOT / "official_context_capture"
PARITY_OUT = p16.PHASE16_ROOT / "chain_parity"


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


def fp32_chain(block_indices, first_input, dev):
    """Sequential FP32 reference: one weighted block resident on GPU at a time."""
    runner = build_runner(cfg_scale=1.0, steps=1)
    for i, blk in enumerate(runner.dit.blocks):
        blk.to("cpu")
        if i in set(block_indices):
            blk.to(torch.float32)
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
    shared_cache = Cache()  # official pipeline semantics: freqs/windows once
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for bi in block_indices:
            blk = runner.dit.blocks[bi]
            blk.to(dev)
            out = blk.forward(vid, txt, vs, ts, emb, shared_cache)
            refs.append({"vid": out[0].detach().float().cpu().contiguous(),
                         "txt": out[1].detach().float().cpu().contiguous()})
            vid = out[0].detach().to(dev).contiguous()
            txt = out[1].detach().to(dev).contiguous()
            blk.to("cpu")
    return refs, runner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--input-block", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-resident", action="store_true",
                    help="skip resident (diagnostic) runs; full 32-block resident "
                         "would need ~10 GB in the adapter process")
    args = ap.parse_args()

    ensure_compat()
    block_indices = [int(x) for x in args.blocks.split(",")]
    bi0 = args.input_block
    dev = torch.device("cuda", 0)
    _vid = torch.from_numpy(np.fromfile(str(CAPTURE / f"block_{bi0}" / "input_vid.bin"), np.float32))
    _txt = torch.from_numpy(np.fromfile(str(CAPTURE / f"block_{bi0}" / "input_txt.bin"), np.float32))
    _emb = torch.from_numpy(np.fromfile(str(CAPTURE / f"block_{bi0}" / "emb.bin"), np.float32))
    first = {
        "vid": _vid.reshape(-1, p16.C).contiguous(),
        "txt": _txt.reshape(-1, p16.C).contiguous(),
        "emb": _emb.reshape(-1, p16.EMB).contiguous(),
        "vid_shape": json.loads((CAPTURE / f"block_{bi0}" / "vid_shapes.json").read_text()),
        "txt_shape": json.loads((CAPTURE / f"block_{bi0}" / "txt_lengths.json").read_text()),
    }
    t0 = time.time()
    print('before chain: allocated=%.2fGB' % (torch.cuda.memory_allocated()/1e9), flush=True)
    refs, runner = fp32_chain(block_indices, first, dev)
    print('after chain: allocated=%.2fGB' % (torch.cuda.memory_allocated()/1e9), flush=True)
    ref_elapsed = time.time() - t0

    result = {"chain": args.chain, "range": block_indices, "input_block": bi0,
              "ref_elapsed_s": round(ref_elapsed, 3), "streaming": {}, "resident": {},
              "final_error": [], "per_block_error": {}}
    modes = ("streaming", "resident") if not args.no_resident else ("streaming",)
    for rep in range(args.repeats):
        for mode in modes:
            out_dir = PARITY_OUT / f"{args.chain}_{mode}_r{rep}"
            rep_path = PARITY_OUT / f"{args.chain}_{mode}_r{rep}.json"
            rpt, stdout, rc = sbm.run_adapter(block_indices, CAPTURE / f"block_{bi0}", out_dir,
                                              load_mode=mode, report_path=rep_path)
            result[mode][f"rep{rep}"] = {"rc": rc, "report": rpt}
    result["resident_mode_run"] = not args.no_resident
    for rep in range(args.repeats):
        if f"rep{rep}" not in result["streaming"]:
            continue
        out_dir = PARITY_OUT / f"{args.chain}_streaming_r{rep}"
        final_vid = load_bin(out_dir / "final_vid.bin", p16.C)
        final_txt = load_bin(out_dir / "final_txt.bin", p16.C)
        result["final_error"].append(chain_error(torch.cat([refs[-1]["vid"], refs[-1]["txt"]]),
                                                 torch.cat([final_vid, final_txt])))
        blk_errs = []
        for j, bi in enumerate(block_indices):
            out_vid = load_bin(out_dir / f"block_{bi}" / "output_vid.bin", p16.C)
            out_txt = load_bin(out_dir / f"block_{bi}" / "output_txt.bin", p16.C)
            blk_errs.append({"block_index": bi, "is_last_layer": bi == p16.LAST_LAYER_INDEX,
                             **chain_error(torch.cat([refs[j]["vid"], refs[j]["txt"]]),
                                           torch.cat([out_vid, out_txt]))})
        result["per_block_error"][rep] = blk_errs
    limit = p16.CHAIN_REL_LIMITS[len(block_indices)]
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
