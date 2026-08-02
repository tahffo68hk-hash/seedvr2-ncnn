# ncnn QKV Debug

This CPU-only graph-prefix isolator uses input `[windows=4,tokens=32,channels=32]` and an `nn.Linear(32,96)` QKV projection.

- `linear`: projection only, `[4,32,96]`
- `linear2d`: projection only with `[32,32]` input, used to separate 2D from 3D InnerProduct behavior
- `split`: projection plus Q/K/V split, three `[4,32,32]` outputs
- `reshape`: split plus `[4,32,4,8]` head view
- `transpose`: reshape plus `(tokens,heads)` transpose, `[4,4,32,8]`

The runner logs load, input and every output `extract()` result. It does not use model weights, CUDA, or SeedVR2 code.
