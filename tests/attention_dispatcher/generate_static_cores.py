"""Generate deterministic FP32 TorchScript/static-core inputs for registry entries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MASK_FIXTURE = ROOT.parent / "dynamic_mask_fixture"
sys.path.insert(0, str(MASK_FIXTURE))
from dynamic_mask_attention import DynamicMaskAttention, DynamicMaskAttentionCore


CASES = {
    "core_444_grid_1x4x4": (1, 4, 16, 16, 32),
    "core_344_grid_1x4x4": (1, 3, 16, 16, 32),
    "core_444_grid_1x5x5": (1, 4, 17, 17, 32),
    "core_444_grid_2x8x8": (1, 8, 32, 32, 32),
    "core_444_grid_2x5x6": (1, 5, 19, 21, 32),
}


def write_tensor(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().to(torch.float32).numpy().tofile(path)


def main() -> None:
    torch.manual_seed(20260803)
    model = DynamicMaskAttention().cpu().eval()
    core = DynamicMaskAttentionCore().cpu().eval()
    core.load_state_dict(model.core.state_dict())
    root = ROOT / "artifacts" / "attention_core"
    for name, shape in CASES.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        x = torch.randn(shape, dtype=torch.float32)
        with torch.no_grad():
            windows, mask, _, metadata = model.partition_with_mask(x)
            output = model(x)
        torch.jit.trace(core, (windows, mask), check_trace=True).save(str(directory / f"{name}.pt"))
        torch.save(x, directory / "reference_input.pt")
        torch.save(output, directory / "reference_output.pt")
        write_tensor(x, directory / "reference_input.bin")
        write_tensor(output, directory / "reference_output.bin")
        write_tensor(windows, directory / "windows.bin")
        write_tensor(mask, directory / "mask.bin")
        manifest = {
            "name": name,
            "window": metadata["window"], "grid": metadata["grid"],
            "padding_rule": "right-edge-to-window-multiple",
            "tokens_per_window": metadata["tokens_per_window"],
            "input_shape": shape, "dtype": "fp32",
            "ncnn_param": f"{name}.ncnn.param", "ncnn_bin": f"{name}.ncnn.bin",
        }
        required = {"name", "window", "grid", "padding_rule", "tokens_per_window", "input_shape", "dtype", "ncnn_param", "ncnn_bin"}
        assert required <= manifest.keys()
        assert manifest["dtype"] == "fp32"
        assert manifest["tokens_per_window"] == manifest["window"][0] * manifest["window"][1] * manifest["window"][2]
        assert not Path(manifest["ncnn_param"]).is_absolute() and not Path(manifest["ncnn_bin"]).is_absolute()
        (directory / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(directory)


if __name__ == "__main__":
    main()
