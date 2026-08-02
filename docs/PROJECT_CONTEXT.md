# Project context

This repository reproduces a narrow SeedVR2-style adaptive-window attention path using ncnn for CPU FP32 deployment study. It is not a complete model. The current fixture is 4 heads × 8 channels, local 4×4×4 windows, baseline and staged hybrid artifacts. ncnn is used to validate artifact routing, tensor contracts, and numerical parity without downloading real weights. See `CURRENT_STATUS.md` for current scope and exclusions.
