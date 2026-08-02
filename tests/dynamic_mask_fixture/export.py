"""Export the fixed Case-C dynamic-mask attention core on CPU."""

from __future__ import annotations

from pathlib import Path

import torch

from dynamic_mask_attention import DynamicMaskAttention, DynamicMaskAttentionCore
from dynamic_window import dynamic_window_reverse


ROOT = Path(__file__).resolve().parent


def _write_float32(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().to(torch.float32).numpy().tofile(path)


def export_fixture() -> Path:
    """Write Case-C BTHWC references and the two-input pre-partitioned core."""
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.manual_seed(20260803)
    model = DynamicMaskAttention().cpu().eval()
    core = DynamicMaskAttentionCore().cpu().eval()
    core.load_state_dict(model.core.state_dict())
    x = torch.randn((1, 4, 17, 17, 32), dtype=torch.float32)
    with torch.no_grad():
        windows, mask, valid, metadata = model.partition_with_mask(x)
        output = model(x)
        core_output = core(windows, mask)
        restored = dynamic_window_reverse(core_output, metadata)
    if not torch.equal(output, restored):
        raise RuntimeError("core output does not match the BTHWC wrapper")

    torch.jit.trace(model, x, check_trace=True).save(str(artifacts / "dynamic_mask_full.pt"))
    torch.jit.trace(core, (windows, mask), check_trace=True).save(str(artifacts / "dynamic_mask_core.pt"))
    for name, tensor in {
        "reference_input": x,
        "reference_output": output,
        "windows": windows,
        "mask": mask,
        "valid": valid.to(torch.float32),
        "core_output": core_output,
    }.items():
        torch.save(tensor, artifacts / f"{name}.pt")
        _write_float32(tensor, artifacts / f"{name}.bin")
    (artifacts / "metadata.txt").write_text(repr(metadata) + "\n", encoding="utf-8")
    (artifacts / "export-summary.txt").write_text(
        "TorchScript full wrapper: PASS (fixed Case C trace)\n"
        "TorchScript two-input core: PASS\n"
        f"input_shape={tuple(x.shape)}\n"
        f"windows_shape={tuple(windows.shape)}\n"
        f"mask_shape={tuple(mask.shape)}\n"
        f"mask_dtype={mask.dtype}\n"
        f"padding={metadata['padding']}\n",
        encoding="utf-8",
    )
    return artifacts


if __name__ == "__main__":
    print(export_fixture())
