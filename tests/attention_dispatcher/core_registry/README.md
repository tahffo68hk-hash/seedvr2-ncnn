# Core Registry

Each entry is keyed by selected window dimensions and grid dimensions. A pnnx
Gemm core currently hard-codes the flattened row count, so window dimensions
alone are not enough to select a valid static ncnn model.

`core_444_grid_1x5x5` corresponds to the verified Case-C FP32 artifact from the
dynamic-mask fixture. The Case-A and Case-B names are registered as planned but
unavailable entries; no missing model is silently substituted.
