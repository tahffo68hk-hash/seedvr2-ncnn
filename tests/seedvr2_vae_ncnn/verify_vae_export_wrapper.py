#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the graph-equivalent export wrappers against the official modules.

Runs the official whole-tensor VAE forward and the export wrapper chain for
Profile A/B and compares every stage output (max abs diff).  Also compares the
final latent and decoded output.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase17_common as pc  # noqa: E402
import vae_export_model as vem  # noqa: E402
from capture_vae_reference import make_input_video, cut_videos  # noqa: E402


def build_pair():
    vae = pc.build_real_vae()
    sd, _ = pc.load_vae_state_dict()
    vae.load_state_dict(sd, strict=True)
    vae.eval().to(torch.float32)
    vae.disable_slicing()
    vae.set_memory_limit(None, None)
    enc_stages = vem.build_encoder_stages(vae.encoder).values()
    dec_stages = vem.build_decoder_stages(vae.decoder).values()
    return vae, enc_stages, dec_stages


def run_chain(modules, x):
    outs = []
    h = x
    for m in modules:
        m.eval()
        with torch.no_grad():
            h = m(h)
        outs.append(h)
    return outs


def main() -> int:
    vae, enc_stages, dec_stages = build_pair()
    report = {"phase": 17, "wrapper_equivalence": {}}
    for pname, prof in pc.PROFILES.items():
        video = make_input_video(prof["frames"], prof["height"], prof["width"], prof["seed"])
        cond = video.permute(0, 3, 1, 2).permute(1, 0, 2, 3)
        cond = cut_videos(cond, 1)
        x = ((cond.float() - 0.5) / 0.5).unsqueeze(0)
        with torch.no_grad():
            latent = vae.encode(x).posterior.mode()
            decoded = vae.decode(latent).sample
        # official stage outputs
        enc_outs = []
        h = x
        enc_handles = []
        enc = vae.encoder
        mods = [enc.conv_in, *enc.down_blocks, enc.mid_block]
        for m in mods:
            enc_handles.append(m.register_forward_hook(lambda mod, inp, out, _l=enc_outs: _l.append(out.detach())))
        enc_handles.append(enc.register_forward_hook(lambda mod, inp, out, _l=enc_outs: _l.append(out.detach())))
        with torch.no_grad():
            enc(x)
        for hh in enc_handles:
            hh.remove()
        dec_outs = []
        dec = vae.decoder
        mods = [dec.conv_in, dec.mid_block, *dec.up_blocks]
        dhandles = []
        for m in mods:
            dhandles.append(m.register_forward_hook(lambda mod, inp, out, _l=dec_outs: _l.append(out.detach())))
        dhandles.append(dec.register_forward_hook(lambda mod, inp, out, _l=dec_outs: _l.append(out.detach())))
        with torch.no_grad():
            dec(latent)
        for hh in dhandles:
            hh.remove()
        my_enc = run_chain(list(enc_stages), x)
        my_dec = run_chain(list(dec_stages), latent)
        names_e = list(vem.build_encoder_stages(enc).keys())
        names_d = list(vem.build_decoder_stages(dec).keys())
        per = {}
        for nm, o, m in zip(names_e, enc_outs, my_enc):
            d = (o.float() - m.float()).abs().max().item()
            per[nm] = {"max_abs_diff": d, "shape": list(o.shape)}
        for nm, o, m in zip(names_d, dec_outs, my_dec):
            d = (o.float() - m.float()).abs().max().item()
            per[nm] = {"max_abs_diff": d, "shape": list(o.shape)}
        dlat = (latent - my_enc[-1][:, :16]).abs().max().item()
        per["latent_vs_enc_out16"] = dlat
        ddec = (decoded - my_dec[-1]).abs().max().item()
        per["decoded_vs_dec_out"] = ddec
        report["wrapper_equivalence"][pname] = per
        worst = max(v["max_abs_diff"] for k, v in per.items() if isinstance(v, dict) and "max_abs_diff" in v)
        print(f"profile {pname}: worst stage diff = {worst:.3e}", flush=True)
        for k, v in per.items():
            print(f"   {k}: {v}", flush=True)
    pc.write_report(str(Path(r"D:\AIProjects\seedvr2-ncnn\reports\phase17_wrapper_equivalence.json")), report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
