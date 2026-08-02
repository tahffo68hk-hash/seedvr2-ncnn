"""CPU-only TorchScript export for the fixed-window QKV rewrite fixture."""

from __future__ import annotations

from pathlib import Path

import torch

from model_rewrite import FixedWindowAttentionCoreRewrite, FixedWindowAttentionRewrite


ROOT = Path(__file__).resolve().parent


def _write_float32(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().to(torch.float32).numpy().tofile(path)


def export_fixture() -> Path:
    """Write deterministic BTHWC reference data and the partitioned ncnn export unit."""
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.manual_seed(20260802)
    model = FixedWindowAttentionRewrite().cpu().eval()
    core = FixedWindowAttentionCoreRewrite().cpu().eval()
    core.load_state_dict(model.state_dict())
    x = torch.randn((1, 2, 8, 8, 32), dtype=torch.float32)

    with torch.no_grad():
        qkv, windows = model.project_qkv(x)
        output = model(x)
        reference = model.reference_forward(x)
        core_output = core(windows)
    if not torch.allclose(output, reference, atol=1e-6, rtol=1e-5):
        raise RuntimeError("rewrite output diverges from direct Linear reference")

    torch.jit.trace(model, x, check_trace=True).save(str(artifacts / "fixed_attention_rewrite_full.pt"))
    torch.jit.trace(core, windows, check_trace=True).save(str(artifacts / "fixed_attention_rewrite_core.pt"))
    torch.save(x, artifacts / "reference_input.pt")
    torch.save(output, artifacts / "reference_output.pt")
    torch.save(windows, artifacts / "core_input.pt")
    torch.save(core_output, artifacts / "core_output.pt")
    _write_float32(x, artifacts / "reference_input.bin")
    _write_float32(output, artifacts / "reference_output.bin")
    _write_float32(windows, artifacts / "core_input.bin")
    _write_float32(core_output, artifacts / "core_output.bin")
    (artifacts / "export-summary.txt").write_text(
        "TorchScript full wrapper: PASS\n"
        "TorchScript ncnn core: PASS\n"
        f"input_shape={tuple(x.shape)}\n"
        f"windows_shape={tuple(windows.shape)}\n"
        f"qkv_shape={tuple(qkv.shape)}\n"
        f"output_shape={tuple(output.shape)}\n"
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}\n",
        encoding="utf-8",
    )
    return artifacts


if __name__ == "__main__":
    print(export_fixture())
