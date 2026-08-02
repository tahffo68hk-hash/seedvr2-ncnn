from pathlib import Path

import torch

from adaptive_window import reference_attention, window_partition, window_reverse


ROOT = Path(__file__).resolve().parent


def main() -> None:
    torch.manual_seed(20250730)
    shape = (1, 2, 8, 8, 32)
    window_size = 4
    x = torch.randn(shape, dtype=torch.float32)

    windows = window_partition(x, window_size)
    expected_windows = shape[0] * (shape[1] // min(shape[1], window_size)) * (shape[2] // window_size) * (shape[3] // window_size)
    expected_tokens = min(shape[1], window_size) * window_size * window_size
    assert windows.shape == (expected_windows, expected_tokens, shape[-1])

    restored = window_reverse(windows, shape, window_size)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)

    output = reference_attention(windows)
    assert output.shape == windows.shape
    assert torch.isfinite(output).all()

    torch.save(x, ROOT / "reference_input.pt")
    torch.save(output, ROOT / "reference_output.pt")
    print(f"input={tuple(x.shape)}")
    print(f"windows={tuple(windows.shape)}")
    print(f"output={tuple(output.shape)}")
    print("partition/reverse=PASS")
    print("attention=PASS")


if __name__ == "__main__":
    main()
