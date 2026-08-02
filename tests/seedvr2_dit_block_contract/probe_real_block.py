"""Phase 11: real SeedVR2 block random-weight forward probe.

Directly imports the REAL NaMMSRTransformerBlock from third_party/SeedVR2 and,
when all non-model dependencies are present, instantiates it with the REAL 3B
config dims and runs eval()/no_grad/CPU/FP32 forward probes with stage hooks.

When a dependency is missing (einops/diffusers/flash_attn/apex/
rotary_embedding_torch), it fails SAFELY: records the blocker to
probe_result.json and exits 3. No packages are installed, no weights are
downloaded, no SeedVR2 source is modified.
"""
import json
import os
import sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDVR2_ROOT = r"D:\AIProjects\third_party\SeedVR2"
MARKER = os.path.join(OUT_DIR, "probe_result.json")

MISSING = []
for mod in ["torch", "einops", "diffusers", "flash_attn", "apex", "rotary_embedding_torch"]:
    import importlib.util
    try:
        if importlib.util.find_spec(mod) is None:
            MISSING.append(mod)
    except Exception:
        MISSING.append(mod)

if MISSING:
    with open(MARKER, "w", encoding="utf-8") as f:
        json.dump({"status": "BLOCKED", "blocker_modules": MISSING,
                   "reason": "missing non-model Python deps; task forbids installing them"}, f, indent=2)
    print("probe_real_block=BLOCKED missing=" + ",".join(MISSING))
    sys.exit(3)

import torch  # noqa: E402

def main():
    sys.path.insert(0, SEEDVR2_ROOT)
    from models.dit_v2.nablocks.mmsr_block import NaMMSRTransformerBlock  # noqa: E402
    from common.cache import Cache  # noqa: E402

    torch.manual_seed(0)
    torch.set_num_threads(1)
    cfg = dict(vid_dim=2560, txt_dim=2560, emb_dim=15360, heads=20, head_dim=128,
               expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
               qk_bias=False, qk_norm="fusedrms", mlp_type="swiglu",
               shared_weights=False, rope_type="mmrope3d", rope_dim=128,
               is_last_layer=False, window=(4, 3, 3), window_method="720pwin_by_size_bysize")
    block = NaMMSRTransformerBlock(**cfg).eval()
    cache = Cache(disable=True)
    profiles = [
        ("A_min", (1, 4, 4), 2),
        ("B_multi_window", (5, 32, 32), 8),
        ("C_asymmetric", (7, 45, 64), 11),
    ]
    stages = {}
    for name, (t, h, w), l in profiles:
        vid = torch.randn(t * h * w, 2560)
        txt = torch.randn(l, 2560)
        vid_shape = torch.tensor([[t, h, w]], dtype=torch.long)
        txt_shape = torch.tensor([[l]], dtype=torch.long)
        emb = torch.randn(1, 15360)
        with torch.no_grad():
            vid_out, txt_out, vs, ts = block(vid, txt, vid_shape, txt_shape, emb, cache)
        def stats(x):
            xf = x.float()
            return {"shape": list(x.shape), "dtype": str(x.dtype), "numel": x.numel(),
                    "finite": bool(torch.isfinite(xf).all().item()),
                    "mean": float(xf.mean().item()), "std": float(xf.std().item()),
                    "min": float(xf.min().item()), "max": float(xf.max().item())}
        stages[name] = {"vid_out": stats(vid_out), "txt_out": stats(txt_out),
                        "vid_shape_out": vs.tolist(), "txt_shape_out": ts.tolist()}
    with open(MARKER, "w", encoding="utf-8") as f:
        json.dump({"status": "PASS", "profiles": stages}, f, indent=2)
    print("probe_real_block=PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        with open(MARKER, "w", encoding="utf-8") as f:
            json.dump({"status": "ERROR", "exception": str(e),
                       "traceback": traceback.format_exc().splitlines()[-8:]}, f, indent=2)
        print("probe_real_block=ERROR " + str(e))
        sys.exit(4)
