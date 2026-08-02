# Head merge minimal repro status

This pass added and executed the C++ `host_head_merge_tests` rather than the requested three independent ncnn Concat/contribution-Gemm models. The test validates tagged channel order, size mismatch rejection, and non-finite rejection for the exact `[W,N,D] x 4 -> [W,N,4D]` host operation.

The available direct evidence for the deployment blocker is the first tuple-output head model: ncnn extraction produced `[64,64,8]` instead of `[4,64,8]` in the rank-3 batched path. The standalone 3-D Concat, 2-D Concat, and 2-D contribution-projection experiments remain unimplemented and must not be interpreted as having passed or failed.
