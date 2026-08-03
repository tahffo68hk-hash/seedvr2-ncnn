#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare regular blocks key pattern and qk-norm values in the real checkpoint."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import torch

ckpt = Path(r"D:\AIProjects\seedvr2-ncnn\.phase14_local_checkpoint\seedvr2_ema_3b.pth")
state = torch.load(str(ckpt), map_location="cpu", weights_only=True)

def keys_of(b):
    return sorted(k for k in state if re.match(rf"blocks\.{b}\.", k))

out = {"blocks": {}}
for b in [0, 15, 17, 31]:
    ks = keys_of(b)
    out["blocks"][str(b)] = {
        "key_count": len(ks),
        "numel": int(sum(state[k].numel() for k in ks)),
        "pattern_vid_txt": all((".vid." in k or ".txt." in k or k.endswith("rope.rope.freqs")) for k in ks),
    }
    qk = {}
    for probe in ["attn.norm_q.vid.weight", "attn.norm_q.txt.weight", "attn.norm_k.vid.weight", "attn.norm_k.txt.weight"]:
        for k in ks:
            if k.endswith(probe):
                t = state[k]
                qk[probe] = {"shape": list(t.shape), "all_ones": bool((t == 1).all().item()),
                             "min": float(t.min().item()), "max": float(t.max().item()),
                             "mean": float(t.mean().item()), "unique": int(t.unique().numel())}
                break
    out["blocks"][str(b)]["qk_norm"] = qk
    out["blocks"][str(b)]["keys_sample"] = ks[:8]
print(json.dumps(out, ensure_ascii=False, indent=1))
(root/"reports/phase14_regular_block_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)+"\n", encoding="utf-8", newline="\n")
