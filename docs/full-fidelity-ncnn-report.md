# Full-fidelity ncnn status

`Single graph full fidelity = BLOCKED`: the pre-existing vectorized 4-D merge fails at blob 89, the per-head graph Concat fails at `cat_0`, and the contribution-Gemm single graph has no usable final output. Those artifacts and reports are retained as failure evidence.

`Hybrid staged full fidelity = READY`: the ArtifactRegistry-driven Release
runtime executes four 2-D static head nets, explicit host merge, one ncnn 2-D
output Gemm, and reverse/crop. Its final BTHWC maximum absolute error was
`1.49012e-07`, with mean `2.10550e-08`, zero NaN, and zero Inf values.

The first attempted tuple-head core is also preserved under the hybrid artifact's legacy `attention_heads/model.ncnn.*`; extracting `out0` returned shape `[64,64,8]` (32,768 elements), while its required shape is `[4,64,8]` (2,048 elements). The ready path replaces that invalid rank-3 batched layout with four valid static 2-D head artifacts.
