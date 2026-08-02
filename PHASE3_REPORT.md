# Adaptive Window Attention Migration Study

## 1. Reference Implementation

`tests/adaptive_window_fixture` is a CPU-only, weight-free fixture. It accepts `[B,T,H,W,C]=[1,2,8,8,32]`, partitions into four `(2,4,4)` windows, executes explicit four-head QKV attention (`4 x head_dim 8`), and reverses output to the original layout. Four unit tests pass, including a hand-derived two-token attention result and an equivalence check between reference and export-oriented four-head cores.

## 2. Tensor Layout

See `docs/token-layout.md`. The runner maps logical PyTorch `[B=1,T,H,W,C]` to `ncnn::Mat(w=C,h=W,d=H,c=T)`. The fixed window tensor `[windows,tokens,C]` maps to `ncnn::Mat(w=C,h=tokens,c=windows)`.

## 3. Export Result

TorchScript export PASS: `artifacts/adaptive_window_trace.pt`.

torch.export PASS: `artifacts/adaptive_window_export.pt2`.

TorchScript is fixed to the fixture shape; it emits expected tracer warnings for shape checks. pnnx receives TorchScript, not the `.pt2` artifact.

## 4. pnnx Graph

The full graph converts to files but has 28 layers / 30 blobs and pnnx reports unsupported 8-rank permute/reshape. The window-core graph converts to 26 layers / 37 blobs using `InnerProduct`, `Slice`, `MatMul`, `BinaryOp`, `Softmax`, and `Concat`. Details: `docs/pnnx-analysis.md`.

## 5. ncnn Result

`ncnn-fixture-runner` compiles in Release mode and explicitly performs BTHWC partition/reverse. Its pnnx-generated projected-QKV core still crashes during `extract()` (exit code `-1073741819`). Phase 4 subsequently proved that ncnn executes isolated MatMul, MatMul+Softmax, and complete fixed attention successfully; see `docs/ncnn-attention-debug.md`. No output was produced for the projected-QKV fixture itself.

## 6. Accuracy

`reports/ncnn_accuracy.json` is BLOCKED with null error fields because no ncnn output exists. This avoids reporting fabricated numerical alignment.

## 7. Custom Layer Decision

Fixed-shape partition/reverse requires a graph rewrite. The real adaptive SeedVR2 operation is a custom-layer candidate because it additionally needs dynamic window slices, index reverse, multimodal text repetition, RoPE and variable-length attention. Design only: `docs/custom-layer-design.md`.

## 8. Vulkan Implementation Plan

Do not begin Vulkan implementation yet. First isolate the ncnn CPU `extract()` fault with a minimal MatMul model. After CPU correctness, implement partition/index, QKV, RoPE, attention softmax/AV and reverse scatter shaders in that order.

## 9. Remaining Problems

- Current ncnn CPU runtime crashes on the generated attention core during `extract()`.
- Full BTHWC graph contains ncnn-unsupported 8-rank transforms.
- No real SeedVR2 checkpoint, text branch, RoPE, norm, varlen attention, or VAE has been converted.

| Module | Status |
|---|---|
| PyTorch reference | READY |
| Tensor layout | READY |
| TorchScript export | READY |
| torch.export | READY |
| pnnx | PARTIAL |
| ncnn inference | BLOCKED |
| accuracy validation | BLOCKED |
| custom layer design | READY |
| Vulkan plan | READY (design only) |
