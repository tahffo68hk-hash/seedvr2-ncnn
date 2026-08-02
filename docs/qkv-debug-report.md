# QKV Projection ncnn Debug Report

## Scope

The test uses a weight-bearing but random `Linear(32,96)` only; no SeedVR2 checkpoint, CUDA, FlashAttention, ncnn source edit, or pnnx source edit is involved. Input is logical `[windows=4,tokens=32,channels=32]`.

## Results

| Case | PyTorch output | pnnx conversion | ncnn `extract()` | Result |
|---|---|---|---|---|
| A: Linear-only | `[4,32,96]` | PASS | crash, exit `-1073741819` | FAIL |
| B: Linear + Split | three `[4,32,32]` | PASS | crash, exit `-1073741819` | FAIL |
| C: B + reshape | three `[4,32,4,8]` | PASS | crash, exit `-1073741819` | FAIL |
| D: C + transpose | three `[4,4,32,8]` | PASS | crash, exit `-1073741819` | FAIL |
| Boundary: 2D Linear-only | `[32,96]` | PASS | `extract ret=0` | PASS |

## First failing node combination

The first failure is Case A: pnnx lowers `nn.Linear` to ncnn `InnerProduct`, and its input is a 3D `Mat(w=32,h=32,c=4)`. Model loading and `Extractor::input()` both return success; the process fails during `Extractor::extract()` before output is returned.

The boundary case uses the same PyTorch `Linear(32,96)` but a 2D `Mat(w=32,h=32)` and passes. Its pnnx graph is lowered to `Gemm`; the 3D case is lowered to `InnerProduct`. Therefore the evidence identifies the pnnx-selected 3D `InnerProduct` execution/lowering path as the first problematic combination in this environment. It does not prove that all ncnn linear execution fails. Split, reshape, and transpose are downstream and are not the trigger.

## Relation to Phase 4.1

Phase 4.1 proved fixed-shape ncnn MatMul, Softmax, and attention-value MatMul all execute and align with PyTorch. The earlier Adaptive Window Attention core fails because QKV `InnerProduct` receives `[windows,tokens,channels]` directly, not because attention math fails.

## Recommended graph rewrite

For fixed windows, rewrite each QKV projection as:

```text
[windows,tokens,C]
  -> reshape [windows*tokens,C]
  -> InnerProduct(C -> 3*C) on 2D tensor
  -> reshape [windows,tokens,3*C]
  -> split/reshape/transpose
```

The pre-projection flatten and post-projection restore must remain at ncnn-supported rank. For the real dynamic SeedVR2 path, dynamic window partition/reverse, RoPE, text repetition and varlen attention remain custom-layer concerns; ordinary fixed QKV projection itself does not need a custom attention kernel once this graph rewrite is applied.
