"""Export the fixed-shape fixture through TorchScript and torch.export on CPU."""

from pathlib import Path

import torch

from adaptive_window import window_partition
from model import AdaptiveWindowAttentionFixture, ExportAttentionCore, WindowAttentionCore


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def write_float32(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().to(torch.float32).numpy().tofile(path)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    torch.manual_seed(20250802)
    model = AdaptiveWindowAttentionFixture().cpu().eval()
    x = torch.randn((1, 2, 8, 8, 32), dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        windows = window_partition(x, model.window_size)
    core = WindowAttentionCore().cpu().eval()
    core.attention.load_state_dict(model.attention.state_dict())
    export_core = ExportAttentionCore().cpu().eval()
    export_core.load_state_dict(core.state_dict())
    with torch.no_grad():
        core_output = core(windows)
        export_core_output = export_core(windows)
    if not torch.allclose(core_output, export_core_output, atol=1e-6, rtol=1e-5):
        raise RuntimeError("unrolled export core does not match reference core")

    trace_path = ARTIFACTS / "adaptive_window_trace.pt"
    traced = torch.jit.trace(model, x, check_trace=True)
    traced.save(str(trace_path))

    core_trace_path = ARTIFACTS / "adaptive_window_core_trace.pt"
    torch.jit.trace(core, windows, check_trace=True).save(str(core_trace_path))
    export_core_trace_path = ARTIFACTS / "adaptive_window_export_core_trace.pt"
    torch.jit.trace(export_core, windows, check_trace=True).save(str(export_core_trace_path))

    export_path = ARTIFACTS / "adaptive_window_export.pt2"
    exported = torch.export.export(model, (x,))
    torch.export.save(exported, export_path)

    torch.save(x, ARTIFACTS / "reference_input.pt")
    torch.save(output, ARTIFACTS / "reference_output.pt")
    write_float32(x, ARTIFACTS / "reference_input.bin")
    write_float32(output, ARTIFACTS / "reference_output.bin")
    torch.save(windows, ARTIFACTS / "core_input.pt")
    torch.save(core_output, ARTIFACTS / "core_output.pt")
    write_float32(windows, ARTIFACTS / "core_input.bin")
    write_float32(core_output, ARTIFACTS / "core_output.bin")
    torch.save(model.state_dict(), ARTIFACTS / "fixture_state_dict.pt")

    graph_ops = [str(node.target) for node in exported.graph.nodes if node.op == "call_function"]
    (ARTIFACTS / "export-summary.txt").write_text(
        "TorchScript: PASS\n"
        "torch.export: PASS\n"
        f"input_shape={tuple(x.shape)}\n"
        f"output_shape={tuple(output.shape)}\n"
        f"parameters={sum(p.numel() for p in model.parameters())}\n"
        f"core_input_shape={tuple(windows.shape)}\n"
        f"core_trace={core_trace_path.name}\n"
        f"export_core_trace={export_core_trace_path.name}\n"
        "torch_export_call_functions:\n" + "\n".join(graph_ops) + "\n",
        encoding="utf-8",
    )
    print(f"TorchScript={trace_path}")
    print(f"torch.export={export_path}")
    print(f"parameters={sum(p.numel() for p in model.parameters())}")
    print(f"output={tuple(output.shape)}")


if __name__ == "__main__":
    main()
