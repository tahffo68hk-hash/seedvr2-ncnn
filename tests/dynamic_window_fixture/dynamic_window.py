"""Reference dynamic BTHWC windowing with explicit host-side metadata."""

from __future__ import annotations

from typing import Any

import torch


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def calculate_window(shape: tuple[int, int, int, int, int], base_window: tuple[int, int, int]) -> tuple[int, int, int]:
    """Use the configured spatial tile and shrink only a short temporal tile."""
    _, time, _, _, _ = shape
    base_t, base_h, base_w = base_window
    if min(base_window) <= 0:
        raise ValueError("base_window must contain positive dimensions")
    return min(time, base_t), base_h, base_w


def dynamic_window_partition(
    x: torch.Tensor, base_window: tuple[int, int, int] = (4, 4, 4)
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pad BTHWC right edges, partition it, and return all reverse-mapping metadata."""
    if x.ndim != 5:
        raise ValueError(f"expected [B,T,H,W,C], got {tuple(x.shape)}")
    original_shape = tuple(int(value) for value in x.shape)
    batch, time, height, width, channels = original_shape
    window_t, window_h, window_w = calculate_window(original_shape, base_window)
    padded_t = _round_up(time, window_t)
    padded_h = _round_up(height, window_h)
    padded_w = _round_up(width, window_w)
    padding = (padded_t - time, padded_h - height, padded_w - width)

    padded = x.new_zeros((batch, padded_t, padded_h, padded_w, channels))
    padded[:, :time, :height, :width, :] = x
    grid_t, grid_h, grid_w = padded_t // window_t, padded_h // window_h, padded_w // window_w
    windows = padded.reshape(batch, grid_t, window_t, grid_h, window_h, grid_w, window_w, channels)
    windows = windows.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        batch * grid_t * grid_h * grid_w, window_t * window_h * window_w, channels
    )
    metadata: dict[str, Any] = {
        "original_shape": original_shape,
        "padded_shape": (batch, padded_t, padded_h, padded_w, channels),
        "window": (window_t, window_h, window_w),
        "window_t": window_t,
        "window_h": window_h,
        "window_w": window_w,
        "grid": (grid_t, grid_h, grid_w),
        "num_windows": int(windows.shape[0]),
        "tokens_per_window": window_t * window_h * window_w,
        "padding": padding,
    }
    return windows, metadata


def dynamic_window_reverse(windows: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    """Restore BTHWC from the partition output and crop right-edge padding."""
    batch, time, height, width, channels = metadata["original_shape"]
    _, padded_t, padded_h, padded_w, _ = metadata["padded_shape"]
    window_t, window_h, window_w = metadata["window"]
    grid_t, grid_h, grid_w = metadata["grid"]
    expected = (batch * grid_t * grid_h * grid_w, window_t * window_h * window_w, channels)
    if tuple(windows.shape) != expected:
        raise ValueError(f"windows shape {tuple(windows.shape)} does not match metadata {expected}")
    padded = windows.reshape(batch, grid_t, grid_h, grid_w, window_t, window_h, window_w, channels)
    padded = padded.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(batch, padded_t, padded_h, padded_w, channels)
    return padded[:, :time, :height, :width, :]
