"""Export case-driven Dynamic RMS Hybrid references using one fixed model."""
from pathlib import Path
import torch
from attention_fidelity import partition_bthwc
from rms_attention import RMSPerHeadBranchRewrite
from rope3d_reference import host_rope_cos_sin

ROOT = Path(__file__).resolve().parent
PROFILES = [
    {"name": "dynamic_a", "shape": [1, 4, 8, 8, 32], "window": [4, 4, 4], "grid": [1, 2, 2], "padding": [0, 0, 0]},
    {"name": "dynamic_b", "shape": [1, 4, 9, 10, 32], "window": [4, 4, 4], "grid": [1, 3, 3], "padding": [0, 3, 2]},
    {"name": "dynamic_c", "shape": [1, 5, 9, 10, 32], "window": [4, 4, 4], "grid": [2, 3, 3], "padding": [3, 3, 2]},
]

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

def main():
    torch.manual_seed(20260805)
    model = RMSPerHeadBranchRewrite().eval()
    for profile in PROFILES: export_profile(model, profile)

if __name__ == "__main__": main()
