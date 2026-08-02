# Fixed Attention Rewrite

This fixture fixes B=1, T=2, H=W=8, C=32 and uses 2x4x4 local windows.

`model_rewrite.py` keeps all export-boundary Linear operations 2D:

- `[4,32,32] -> [128,32] -> Gemm -> [128,96] -> [4,32,96]`
- `[4,32,32] -> [128,32] -> Gemm -> [128,32] -> [4,32,32]`

`export.py` writes a BTHWC reference plus a pre-partitioned attention core. The existing
`D:\AIProjects\seedvr2-ncnn\build\ncnn-fixture-runner\ncnn_fixture_runner.exe` performs
the fixed partition/reverse mapping around the converted core.
