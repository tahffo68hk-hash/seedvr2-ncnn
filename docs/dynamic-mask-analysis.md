# Dynamic Window Attention Mask Analysis

## Fixture Scope

Case C only: FP32 CPU input `[1,4,17,17,32]`, base window `[4,4,4]`, right-edge
padding `[T,H,W]=[0,3,3]`, 25 windows, and 64 tokens per window. No
FlashAttention, Vulkan, SeedVR2 weights, or custom layer code is used.

## Mask Semantics

The validity map is partitioned with the same dynamic metadata as the latent.
For each window, invalid padded keys receive an additive logit bias of `-10000`:

```text
scores = Q @ K^T / sqrt(head_dim)
weights = softmax(scores + mask)
output = weights @ V
```

The mask blocks padded tokens as keys for every query. Padded query outputs may
still be computed, but reverse/crop removes them; no padded key or value can
influence any retained valid position.

## Mask Tensor

| Property | Value |
| --- | --- |
| PyTorch shape | `[windows, query_tokens, key_tokens] = [25,64,64]` |
| dtype | `float32` |
| layout | Window-major, then query-token row, then key-token column |
| ncnn::Mat mapping | `w=64` key columns, `h=64` query rows, `c=25` windows |
| use | The same mask is added to each of the four head score matrices before Softmax. |

## Verification

| Check | Result |
| --- | --- |
| Changing every padded input token to large random FP32 values changes no valid-token output | PASS |
| Masked wrapper reverse/crop returns `[1,4,17,17,32]` | PASS |
| TorchScript full fixed-Case-C wrapper | PASS, shape-specialized trace warnings expected |
| TorchScript two-input core | PASS |
| pnnx FP32 conversion | PASS |
| ncnn CPU real `Extractor::extract` | PASS |
| ncnn core vs PyTorch core | max `1.7881e-7`, mean `1.6767e-8` |
| ncnn output + host reverse/crop vs PyTorch BTHWC | max `1.7881e-7`, mean `1.6054e-8` |

The converted ncnn graph has 35 layers and 50 blobs. It retains `in1` as a
second input and emits four `BinaryOp Add` nodes, one directly before each head's
Softmax. QKV and output projections remain the proven 2D FP32 Gemm rewrite.

## Convertibility Decision

| Item | Classification | Reason |
| --- | --- | --- |
| A fixed-shape FP32 additive mask supplied as an ncnn input | A: ordinary tensor | pnnx preserves it as `in1`; ncnn runs `BinaryOp Add` plus Softmax correctly. |
| Building mask tensor from known per-shape metadata before extract | B: graph rewrite / host wrapper | Shape-specialized code can prepare `[W,N,N]`, but traced Python shape logic becomes constants. |
| Generating padding, validity, mask, partition/reverse, and crop for arbitrary runtime shape within one graph | C: custom layer | It requires runtime allocation, metadata propagation, dynamic reshape/grid control, and mask construction. |

## Boundary for a Future Layer

No custom layer is needed for attention computation or a precomputed fixed mask.
A general dynamic-window custom layer is justified only for the runtime metadata
path and optional partition/mask/attention/reverse/crop fusion. This report makes
no Vulkan implementation decision.
