# QKV Graph Rewrite

## Original Graph

Input is logical `[windows=4,tokens=32,C=32]`. The original implementation applies `Linear(32,96)` directly and returns `[4,32,96]`.

PyTorch output shape is correct, but pnnx lowers this graph to:

```text
Input [4,32,32] -> InnerProduct -> output
```

The generated ncnn model loads and accepts input, then reproduces the access violation during `Extractor::extract()` (exit `-1073741819`). It is not a usable QKV projection path.

## Rewrite Graph

The rewrite uses the same `Linear(32,96)` weights and bias:

```text
[4,32,32]
  -> reshape [128,32]
  -> Linear(32,96)
  -> reshape [4,32,96]
```

## pnnx Nodes

| Model | pnnx/ncnn lowering |
|---|---|
| Original | `Input -> InnerProduct` |
| Rewrite | `Input -> Reshape -> Gemm -> Reshape` |

## ncnn Nodes

The rewrite uses only rank-supported reshapes around `Gemm`. Its C++ runner maps the input to `ncnn::Mat(w=32,h=32,c=4)` and reports `stage=extract ret=0`, with output dimensions `w=96,h=32,c=4`, i.e. logical `[4,32,96]`.

## Accuracy

| Comparison | Max abs error | Mean abs error |
|---|---:|---:|
| PyTorch original vs PyTorch rewrite | 0.0 | 0.0 |
| PyTorch rewrite vs ncnn rewrite | 3.5762787e-07 | 3.6732821e-08 |

## Conclusion

QKV projection: **READY with graph rewrite**.

For fixed windows, flatten the `windows` and `tokens` axes before every Q/K/V projection, use the 2D `Gemm` lowering, and restore the 3D layout before split/head reshape. This avoids the failing 3D `InnerProduct` path without changing ncnn or pnnx source.

This resolves only fixed-shape QKV projection. Dynamic SeedVR2 window partition/reverse, multimodal text paths, Q/K norms, 3D RoPE and varlen attention remain separate conversion work.
