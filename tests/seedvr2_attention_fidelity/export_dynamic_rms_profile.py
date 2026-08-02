"""Export case-driven Dynamic RMS Hybrid references using one fixed model."""
from pathlib import Path
import argparse
import json
import torch
from attention_fidelity import partition_bthwc
from rms_attention import RMSPerHeadBranchRewrite, SingleRMSAttentionHead
from rope3d_reference import host_rope_cos_sin, host_dynamic_rope3d_cos_sin, host_lang_rope3d_cos_sin
from output_projection_core import OutputProjectionCore

ROOT = Path(__file__).resolve().parent
PROFILES = [
    {"name": "dynamic_a", "shape": [1, 4, 8, 8, 32], "window": [4, 4, 4], "grid": [1, 2, 2], "padding": [0, 0, 0]},
    {"name": "dynamic_b", "shape": [1, 4, 9, 10, 32], "window": [4, 4, 4], "grid": [1, 3, 3], "padding": [0, 3, 2]},
    {"name": "dynamic_c", "shape": [1, 5, 9, 10, 32], "window": [4, 4, 4], "grid": [2, 3, 3], "padding": [3, 3, 2]},
]

# Phase 10 synthetic head_dim=128 / rope_dim=128 shape-only fixture.  The axis
# split below is a documented SYNTHETIC contract (even pair dims summing to
# rope_dim=128 with the same theta/axis-scale scheme as the Phase 9.5 C=32
# fixture).  It does not claim to reproduce the real SeedVR2 mmrope3d split,
# which cannot be resolved locally because rotary-embedding-torch is absent.
HD128_PROFILES = [
    {"name": "hd128_single", "shape": [1, 4, 4, 4, 512], "window": [4, 4, 4], "grid": [1, 1, 1], "padding": [0, 0, 0]},
    {"name": "hd128_multi", "shape": [1, 4, 8, 8, 512], "window": [4, 4, 4], "grid": [1, 2, 2], "padding": [0, 0, 0]},
    {"name": "hd128_padded", "shape": [1, 5, 9, 10, 512], "window": [4, 4, 4], "grid": [2, 3, 3], "padding": [3, 3, 2]},
]
HD128_VARIANT = "synthetic_hd128_dynamic_rms_rope3d_hybrid"
HD128_ROPE = {"rope_dim": 128, "theta": 10000.0, "axis_dims": [44, 42, 42], "axis_scales": [1.0, 0.01, 1e-4]}

# Phase 10.1 real SeedVR2 MM-RoPE3D contract (rotary-embedding-torch==0.5.3,
# SeedVR2 commit e4de8c24441a67e1b7df56abea10645059bb1185).  SeedVR2 configures
# rope_dim=128 and builds RotaryEmbedding(dim=128 // 3=42, freqs_for="lang",
# theta=10000); three axial 42-dim blocks concatenate to an effective rotary
# dim of 126 with a 2-dim identity tail.  rope_dim below is the effective
# value and freq_mode="lang_rope" selects the per-axis lang formula.
HD128_REAL_VARIANT = "seedvr2_hd128_mmrope3d_hybrid"
HD128_REAL_ROPE = {"rope_dim": 126, "theta": 10000.0, "axis_dims": [42, 42, 42],
                   "axis_scales": [1.0, 1.0, 1.0], "freq_mode": "lang_rope"}

def write(value, path):
    value.detach().cpu().contiguous().float().numpy().tofile(path)

def validity(shape, window, grid):
    batch, time, height, width, _ = shape
    wt, wh, ww = window
    rows = []
    for _batch in range(batch):
        for gt in range(grid[0]):
            for gh in range(grid[1]):
                for gw in range(grid[2]):
                    rows.append([gt * wt + lt < time and gh * wh + lh < height and gw * ww + lw < width
                                 for lt in range(wt) for lh in range(wh) for lw in range(ww)])
    return torch.tensor(rows, dtype=torch.uint8)

def export_profile(model, profile):
    shape, window, grid = profile["shape"], tuple(profile["window"]), profile["grid"]
    artifact = ROOT / "artifacts/attention_core_fidelity" / f"hybrid_dynamic_rms_rope444_grid_{grid[0]}x{grid[1]}x{grid[2]}"
    root = artifact / "reference"
    root.mkdir(parents=True, exist_ok=True)
    x = torch.randn(*shape)
    windows, _ = partition_bthwc(x, window)
    valid = validity(shape, window, grid)
    mask = torch.where(valid[:, None, :].bool(), torch.zeros(1), torch.full((1,), -10000.0)).expand(-1, 64, -1).clone()
    cos, sin = host_rope_cos_sin(*window, head_dim=8, rotary_dim=6)
    with torch.no_grad():
        stages = model.forward_stages(x, mask, cos, sin)
    merged = torch.cat(tuple(stages[f"head{i}_av"] for i in range(4)), dim=-1)
    projection = model.output(merged.reshape(-1, 32))
    values = {"input": x, "windows": windows, "validity": valid, "mask": mask, "rope_cos": cos, "rope_sin": sin,
              "merged": merged, "projection": projection, "projection_2d": projection,
              "windows_output": stages["windows"], "bthwc_output": stages["output"]}
    values.update({f"head{i}": stages[f"head{i}_av"] for i in range(4)})
    for name, value in values.items(): write(value, root / f"{name}.bin")

def write_metadata(artifact, grid, variant=HD128_VARIANT, rope=HD128_ROPE):
    models = {"heads": [{"name": f"head{i}", "ncnn_param": f"attention_heads/head{i}/model.ncnn.param",
                        "ncnn_bin": f"attention_heads/head{i}/model.ncnn.bin"} for i in range(4)],
              "output_projection": {"name": "output_projection", "ncnn_param": "output_projection/model.ncnn.param",
                                    "ncnn_bin": "output_projection/model.ncnn.bin"}}
    manifest = {"name": artifact.name, "variant": variant, "execution": "staged_host_head_merge",
                "window": [4, 4, 4], "grid": list(grid), "tokens_per_window": 64, "dtype": "fp32",
                "features": {"qk_norm": "rmsnorm", "rope3d": "dynamic_mmrope3d", "masked_attention": True,
                             "head_merge": "host", "output_projection": "ncnn_2d_gemm"},
                "auxiliary_inputs": {"mask": {"mode": "padding_validity", "invalid_key_value": -10000.0},
                                     "rope": {"mode": "dynamic_mmrope3d", **rope}},
                "models": models}
    (artifact / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def export_hd128():
    """One shared synthetic C=512/heads=4/head_dim=128 model with three shape-only profiles."""
    torch.manual_seed(20260805)
    model = RMSPerHeadBranchRewrite(channels=512, heads=4, head_dim=128).eval()
    source = ROOT / "artifacts/attention_core_fidelity" / "hybrid_synthetic_hd128_rope444_grid_1x1x1"
    heads_root, projection_root = source / "attention_heads", source / "output_projection"
    for directory in (heads_root, projection_root): directory.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 4, 4, 4, 512)
    windows, _ = partition_bthwc(x, (4, 4, 4))
    valid = validity([1, 4, 4, 4, 512], (4, 4, 4), (1, 1, 1))
    mask = torch.where(valid[:, None, :].bool(), torch.zeros(1), torch.full((1,), -10000.0)).expand(-1, 64, -1).clone()
    cos, sin = host_dynamic_rope3d_cos_sin(4, 4, 4, head_dim=128, rope_dim=HD128_ROPE["rope_dim"],
                                           theta=HD128_ROPE["theta"], axis_dims=HD128_ROPE["axis_dims"],
                                           axis_scales=HD128_ROPE["axis_scales"])
    with torch.no_grad(): stages = model.forward_stages(x, mask, cos, sin)
    merged = torch.cat(tuple(stages[f"head{i}_av"] for i in range(4)), dim=-1)
    for i in range(4):
        head = SingleRMSAttentionHead(model, i).eval()
        directory = heads_root / f"head{i}"; directory.mkdir(exist_ok=True)
        torch.jit.trace(head, (windows[0], mask[0], cos, sin), check_trace=True).save(str(directory / "model.pt"))
    torch.jit.trace(OutputProjectionCore(model.output).eval(), (merged.reshape(-1, 512),), check_trace=True).save(str(projection_root / "model.pt"))
    for profile in HD128_PROFILES:
        shape, window, grid = profile["shape"], tuple(profile["window"]), profile["grid"]
        artifact = ROOT / "artifacts/attention_core_fidelity" / f"hybrid_synthetic_hd128_rope444_grid_{grid[0]}x{grid[1]}x{grid[2]}"
        root = artifact / "reference"; root.mkdir(parents=True, exist_ok=True)
        x = torch.randn(*shape)
        windows, _ = partition_bthwc(x, window)
        valid = validity(shape, window, grid)
        mask = torch.where(valid[:, None, :].bool(), torch.zeros(1), torch.full((1,), -10000.0)).expand(-1, 64, -1).clone()
        with torch.no_grad(): stages = model.forward_stages(x, mask, cos, sin)
        merged = torch.cat(tuple(stages[f"head{i}_av"] for i in range(4)), dim=-1)
        projection = model.output(merged.reshape(-1, 512))
        values = {"input": x, "windows": windows, "validity": valid, "mask": mask, "rope_cos": cos, "rope_sin": sin,
                  "merged": merged, "projection": projection, "projection_2d": projection,
                  "windows_output": stages["windows"], "bthwc_output": stages["output"]}
        values.update({f"head{i}": stages[f"head{i}_av"] for i in range(4)})
        for name, value in values.items(): write(value, root / f"{name}.bin")
        write_metadata(artifact, grid)


def export_hd128_real():
    """Real SeedVR2 MM-RoPE3D contract variant reusing the synthetic ncnn models.

    The head/projection weights do not depend on the RoPE contract (cos/sin are
    model inputs), so the ncnn files are hard-linked from the synthetic S
    artifact and only the reference and metadata are regenerated.  The RNG
    sequence matches export_hd128 so the input/windows/validity/mask reference
    files are byte-identical to the synthetic ones.
    """
    torch.manual_seed(20260805)
    model = RMSPerHeadBranchRewrite(channels=512, heads=4, head_dim=128).eval()
    _ = torch.randn(1, 4, 4, 4, 512)  # keep RNG aligned with export_hd128 trace step
    cos, sin = host_lang_rope3d_cos_sin(4, 4, 4, head_dim=128, rope_dim=HD128_REAL_ROPE["rope_dim"],
                                        theta=HD128_REAL_ROPE["theta"], axis_dims=HD128_REAL_ROPE["axis_dims"],
                                        axis_scales=HD128_REAL_ROPE["axis_scales"])
    synthetic_source = ROOT / "artifacts/attention_core_fidelity" / "hybrid_synthetic_hd128_rope444_grid_1x1x1"
    for profile in HD128_PROFILES:
        shape, window, grid = profile["shape"], tuple(profile["window"]), profile["grid"]
        artifact = ROOT / "artifacts/attention_core_fidelity" / f"hybrid_seedvr2_hd128_mmrope444_grid_{grid[0]}x{grid[1]}x{grid[2]}"
        root = artifact / "reference"
        root.mkdir(parents=True, exist_ok=True)
        x = torch.randn(*shape)
        windows, _ = partition_bthwc(x, window)
        valid = validity(shape, window, grid)
        mask = torch.where(valid[:, None, :].bool(), torch.zeros(1), torch.full((1,), -10000.0)).expand(-1, 64, -1).clone()
        with torch.no_grad(): stages = model.forward_stages(x, mask, cos, sin)
        merged = torch.cat(tuple(stages[f"head{i}_av"] for i in range(4)), dim=-1)
        projection = model.output(merged.reshape(-1, 512))
        values = {"input": x, "windows": windows, "validity": valid, "mask": mask, "rope_cos": cos, "rope_sin": sin,
                  "merged": merged, "projection": projection, "projection_2d": projection,
                  "windows_output": stages["windows"], "bthwc_output": stages["output"]}
        values.update({f"head{i}": stages[f"head{i}_av"] for i in range(4)})
        for name, value in values.items(): write(value, root / f"{name}.bin")
        write_metadata(artifact, grid, variant=HD128_REAL_VARIANT, rope=HD128_REAL_ROPE)
        for directory in (artifact / "attention_heads", artifact / "output_projection"):
            directory.mkdir(parents=True, exist_ok=True)
    # Hard-link ncnn models plus conversion provenance from the synthetic S artifact.
    import os as _os
    for suffix in ("1x1x1", "1x2x2", "2x3x3"):
        target = ROOT / "artifacts/attention_core_fidelity" / f"hybrid_seedvr2_hd128_mmrope444_grid_{suffix}"
        for head in range(4):
            src = synthetic_source / "attention_heads" / f"head{head}"
            dst_dir = target / "attention_heads" / f"head{head}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in ("model.ncnn.param", "model.ncnn.bin", "model.pnnx.param", "model_ncnn.py", "model_pnnx.py"):
                dst = dst_dir / name
                if dst.exists(): _os.remove(dst)
                _os.link(src / name, dst)
        src = synthetic_source / "output_projection"
        dst_dir = target / "output_projection"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.ncnn.param", "model.ncnn.bin", "model.pnnx.param", "model_ncnn.py", "model_pnnx.py"):
            dst = dst_dir / name
            if dst.exists(): _os.remove(dst)
            _os.link(src / name, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hd128", action="store_true", help="export Phase 10 synthetic head_dim=128 profiles")
    parser.add_argument("--hd128-real", action="store_true", help="export Phase 10.1 real SeedVR2 MM-RoPE3D contract variant")
    args = parser.parse_args()
    if args.hd128_real:
        export_hd128_real()
        return
    if args.hd128:
        export_hd128()
        return
    torch.manual_seed(20260805)
    model = RMSPerHeadBranchRewrite().eval()
    for profile in PROFILES: export_profile(model, profile)

if __name__ == "__main__": main()
