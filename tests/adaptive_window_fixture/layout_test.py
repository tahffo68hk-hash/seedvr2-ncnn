"""Generate a human-readable proof of B,T,H,W,C window token ordering."""

from pathlib import Path

import torch

from adaptive_window import window_partition, window_reverse


ROOT = Path(__file__).resolve().parent
DOC = ROOT.parents[1] / "docs" / "token-layout.md"


def main() -> None:
    x = torch.zeros((1, 2, 8, 8, 32), dtype=torch.float32)
    for t in range(2):
        for h in range(8):
            for w in range(8):
                x[0, t, h, w, 0] = 100 * t + 10 * h + w
    windows = window_partition(x, window_size=4)
    assert torch.equal(window_reverse(windows, tuple(x.shape), 4), x)

    lines = [
        "# Token Layout Proof",
        "",
        "PyTorch source layout is contiguous `[B,T,H,W,C]`. Fixed window layout is `[num_windows,tokens,C]`.",
        "For the fixture, windows enumerate `(b, block_t, block_h, block_w)` and tokens enumerate `(local_t, local_h, local_w)` with `local_w` varying fastest.",
        "",
        "## First window",
        "",
        "| token index | (t,h,w) | channel-0 value |",
        "|---:|---|---:|",
    ]
    for index in range(windows.shape[1]):
        local_t = index // 16
        remainder = index % 16
        local_h = remainder // 4
        local_w = remainder % 4
        value = windows[0, index, 0].item()
        lines.append(f"| {index} | ({local_t},{local_h},{local_w}) | {value:.0f} |")
    lines += [
        "",
        "## ncnn mapping",
        "",
        "ncnn::Mat does not preserve a PyTorch five-dimensional layout directly. The C++ runner stores a single fixture tensor as `w=C`, `h=W`, `d=H`, `c=T` (batch is fixed at 1). Thus element `(t,h,w,c)` is addressed as `mat.channel(t).depth(h).row(w)[c]`. A window is represented as `w=C`, `h=tokens`, `c=num_windows`; element `(window,token,channel)` is `mat.channel(window).row(token)[channel]`.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DOC)


if __name__ == "__main__":
    main()
