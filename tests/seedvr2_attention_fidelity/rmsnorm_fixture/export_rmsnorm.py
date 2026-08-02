"""Export deterministic RMSNorm A-D TorchScript fixtures and raw references."""
from pathlib import Path
import json
import torch

from rmsnorm_reference import RMSNormReference
from rmsnorm_rewrite import RMSNormRewriteA, RMSNormRewriteB, RMSNormRewriteC, RMSNormRewriteD

ROOT = Path(__file__).resolve().parent
EPS = 1.0e-5


def write_raw(value: torch.Tensor, path: Path) -> None:
    value.detach().cpu().contiguous().to(torch.float32).numpy().tofile(path)


def main() -> Path:
    torch.manual_seed(20260804)
    values = torch.randn(4, 64, 8)
    weight = torch.linspace(0.5, 1.5, 8)
    reference = RMSNormReference(8, EPS, weight).eval()
    variants = {"A": RMSNormRewriteA(8, EPS, weight), "B": RMSNormRewriteB(8, EPS, weight),
                "C": RMSNormRewriteC(8, EPS, weight), "D": RMSNormRewriteD(8, EPS, weight)}
    ROOT.mkdir(parents=True, exist_ok=True)
    write_raw(values, ROOT / "input.bin")
    write_raw(weight, ROOT / "weight.bin")
    with torch.no_grad():
        write_raw(reference(values), ROOT / "reference.bin")
        for name, model in variants.items():
            model.eval()
            traced = torch.jit.trace(model, values, check_trace=True)
            traced.save(str(ROOT / f"rmsnorm_{name}.pt"))
            write_raw(model(values), ROOT / f"reference_{name}.bin")
    (ROOT / "export-summary.json").write_text(json.dumps({
        "shape": [4, 64, 8], "dtype": "fp32", "eps": EPS,
        "elementwise_affine": True, "bias": False, "seed": 20260804,
        "variants": ["A", "B", "C", "D"],
    }, indent=2) + "\n", encoding="utf-8")
    return ROOT


if __name__ == "__main__":
    print(main())
