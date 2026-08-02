# ncnn Attention Minimal

This CPU-only crash isolator uses a fixed tensor with logical shape `[B=1,heads=4,tokens=32,dim=8]`; the batch is represented by the single test case and input is stored as `[heads,tokens,dim]`.

Three TorchScript models are exported by `export_attention.py`:

- `matmul`: `QK^T / sqrt(8)`
- `softmax`: MatMul plus Softmax
- `attention`: MatMul plus Softmax plus `V`

The C++ runner maps input to `ncnn::Mat(w=8,h=32,c=4)`, logs model-load/input/extract stages, and writes raw float32 output only after `extract()` succeeds. `compare.py` reports the error against the corresponding PyTorch reference.

No SeedVR2 source, checkpoint, GPU operation, or FlashAttention is involved.
