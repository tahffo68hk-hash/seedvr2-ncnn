# pnnx Analysis: Adaptive Window Attention Fixture

## Export inputs

- TorchScript: `tests/adaptive_window_fixture/artifacts/adaptive_window_trace.pt` — PASS
- torch.export: `tests/adaptive_window_fixture/artifacts/adaptive_window_export.pt2` — PASS
- pnnx input: TorchScript only. The `.pt2` artifact is recorded for graph inspection but was not accepted as a pnnx conversion input.

The model has 4,224 trainable parameters: four `Linear(32,32)` layers, each with 1,024 weights and 32 bias values.

## PyTorch graph nodes

`torch.export` records: reshape, permute, transpose, linear, matmul, mul (the `1/sqrt(8)` scale), softmax, and final reshape. There is no FlashAttention, CUDA operation, model weight download, or SeedVR2 import.

## pnnx results

| Graph | pnnx result | ncnn result |
|---|---|---|
| Full `[1,2,8,8,32]` fixture | Conversion files generated | 28 layers / 30 blobs, but pnnx reports unsupported 8-rank permute/physical reshape |
| Window core `[4,32,32]` | Conversion files generated | 26 layers / 37 blobs; standard Linear/Slice/MatMul/Softmax/Concat graph |

The full fixture includes 8-rank partition/reverse reshapes and is not runnable by the current ncnn runtime. This is a real conversion warning, not an assumed limitation.

The core ncnn graph contains `InnerProduct`, `Slice`, `MatMul`, `BinaryOp`, `Softmax`, and `Concat`; no pnnx unsupported-op diagnostic was emitted for the core. Phase 4.2 isolated the crash to `InnerProduct` receiving a 3D `[windows,tokens,channels]` ncnn tensor. The same `Linear(32,96)` passes on a 2D tensor, while fixed MatMul/Softmax/AV attention passes. See `docs/qkv-debug-report.md`.

## Decision

| Scope | Decision | Reason |
|---|---|---|
| Fixed partition/reverse | B: pnnx + graph rewrite | ncnn supports only up to 4D `Mat`; source uses 5D/8D logical transforms |
| Fixed window attention core | B: pnnx + runtime investigation | pnnx graph is generated, but the current CPU runtime crashes during `extract()` |
| Real SeedVR2 adaptive window attention | C: custom layer candidate | Dynamic windows, index mapping, text repetition, RoPE and varlen attention cannot be represented safely as this fixed graph |
