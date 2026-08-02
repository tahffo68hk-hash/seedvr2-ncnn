"""Export original and flattened-Gemm QKV projections with identical weights."""

import json
from pathlib import Path

import torch

from model_original import OriginalQkvProjection
from model_rewrite import RewriteQkvProjection


def write_raw(tensor: torch.Tensor, path: Path) -> None:
    tensor.detach().cpu().contiguous().numpy().tofile(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.manual_seed(20250804)
    original = OriginalQkvProjection().eval()
    rewrite = RewriteQkvProjection().eval()
    rewrite.load_state_dict(original.state_dict())
    x = torch.randn((4, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        original_output = original(x)
        rewrite_output = rewrite(x)
    difference = (original_output - rewrite_output).abs()
    if not torch.equal(original_output, rewrite_output):
        raise RuntimeError("rewrite output differs from original")

    for name, model, output in (("original", original, original_output), ("rewrite", rewrite, rewrite_output)):
        torch.jit.trace(model, x, check_trace=True).save(str(artifacts / f"{name}.pt"))
        torch.save(output, artifacts / f"{name}_output.pt")
        write_raw(output, artifacts / f"{name}_output.bin")
    torch.save(x, artifacts / "input.pt")
    write_raw(x, artifacts / "input.bin")
    (artifacts / "pytorch_accuracy.json").write_text(json.dumps({"max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean())}, indent=2) + "\n", encoding="utf-8")
    print("original/rewrite output=[4,32,96]")
    print((artifacts / "pytorch_accuracy.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
