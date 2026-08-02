"""CPU baseline for future ncnn/Vulkan comparison."""

import json
import time
from pathlib import Path

import torch

from model import AdaptiveWindowAttentionFixture


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    report_path = root / "reports" / "fixture_benchmark.json"
    torch.manual_seed(20250802)
    torch.set_num_threads(1)
    model = AdaptiveWindowAttentionFixture().cpu().eval()
    x = torch.randn((1, 2, 8, 8, 32), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        start = time.perf_counter()
        for _ in range(20):
            model(x)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / 20
    report = {
        "input_shape": [1, 2, 8, 8, 32],
        "dtype": "float32",
        "python_cpu_avg_ms": elapsed_ms,
        "python_cpu_threads": 1,
        "estimated_input_output_bytes": 32768,
        "ncnn_cpu_avg_ms": None,
        "ncnn_status": "BLOCKED: runtime access violation during extract()",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
