# Fixed Attention Rewrite Report

## Graph

Fixed input: `[B,T,H,W,C] = [1,2,8,8,32]`; fixed local window: `[2,4,4]`.

```text
BTHWC -> window_partition -> [4,32,32]
      -> reshape [128,32] -> Gemm QKV -> reshape [4,32,96]
      -> Split Q/K/V -> 4 x (QK^T -> scale -> Softmax -> AV)
      -> Concat heads -> reshape [128,32] -> Gemm output -> [4,32,32]
      -> window_reverse -> BTHWC
```

The TorchScript wrapper remains the BTHWC reference. The ncnn export unit is the
pre-partitioned `[4,32,32]` core; the existing fixture runner performs the fixed,
lossless C++ partition and reverse mapping around `Extractor::extract`.

## Rewrite Rules

| Original boundary | Rewrite boundary | Reason |
| --- | --- | --- |
| `[windows,tokens,C] -> Linear(3C)` | `[windows*tokens,C] -> Gemm(3C) -> [windows,tokens,3C]` | Avoid pnnx lowering 3D Linear to failing `InnerProduct`. |
| `[windows,tokens,C] -> Linear(C)` | `[windows*tokens,C] -> Gemm(C) -> [windows,tokens,C]` | Avoid the same 3D `InnerProduct` path for the output projection. |

## ncnn Nodes

`fixed_attention_rewrite_core.ncnn.param` has 29 layers and 41 blobs:

- 2 `Gemm` nodes: QKV `128x32 -> 128x96`; output `128x32 -> 128x32`.
- 4 independent `MatMul -> BinaryOp(scale) -> Softmax -> MatMul` head paths.
- `Reshape`, `Slice`, and `Concat` only for fixed shape bookkeeping.

No `InnerProduct` node remains in the converted core.

## Accuracy

| Runtime | Output shape | max abs error | mean abs error | Result |
| --- | --- | ---: | ---: | --- |
| PyTorch rewrite vs direct reference | `[1,2,8,8,32]` | 0 | 0 | PASS |
| ncnn CPU, pnnx default `fp16=1` | `[1,2,8,8,32]` | `3.5128e-4` | `7.6864e-5` | Does not meet target |
| ncnn CPU, pnnx `fp16=0` | `[1,2,8,8,32]` | `1.1921e-7` | `1.9188e-8` | PASS |

The final ncnn verification used a real `Extractor::extract`; model load, input,
and all 10 extract calls passed. CPU timing was `0.140 ms` average over 9 timed
runs on this fixed core. `fp16=0` is the required correctness baseline because
the default pnnx FP16 weight conversion dominates the otherwise small error.

## Remaining Issues

- This is deliberately a fixed-shape migration unit. Its BTHWC wrapper has
  TorchScript coverage, but ncnn conversion targets the core because the known
  full-wrapper 8-rank partition/reverse graph is not a portable pnnx target.
- Window partition/reverse is currently explicit C++ host-side layout handling;
  arbitrary runtime shapes need either graph-rewrite rules or a custom layer.
- Real SeedVR2 uses dynamic windows, variable-length sequences, RoPE, text
  conditioning, and FlashAttention-oriented execution. Those are not covered by
  this fixed fixture.

## Custom Layer Outlook

For fixed local windows, the attention kernel and both projections now use
standard ncnn nodes and need no custom layer. A future dynamic-window layer
should own runtime partition/reverse, variable window metadata, padding/masks,
and Vulkan dispatch. The fixed QKV flatten-Gemm-restore rule should remain a
portable fallback inside that design.
