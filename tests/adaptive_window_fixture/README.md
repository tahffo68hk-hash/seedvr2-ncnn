# Adaptive Window Attention Fixture

This is a standalone CPU reference fixture. It does not import SeedVR2, load model weights, use a GPU, or use FlashAttention.

Input layout is `[B,T,H,W,C]`. With `B=1, T=2, H=8, W=8, C=32` and `window_size=4`, the fixture uses a temporal window of `min(T,4)=2` and spatial windows of `4x4`. It creates `1 * 1 * 2 * 2 = 4` windows. Each window contains `2*4*4 = 32` tokens and 32 channels.

Run from this directory:

```powershell
python test_fixture.py
```

The test verifies partition shape, exact reverse reconstruction, finite reference attention output, and saves `reference_input.pt` and `reference_output.pt` beside the scripts.
