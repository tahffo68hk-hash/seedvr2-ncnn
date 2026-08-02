# Adaptive Window Attention Runtime

The Release CPU runtime is:

`AttentionRequest -> metadata -> CoreKey -> ArtifactRegistry -> NcnnModelCache -> partition/mask -> ncnn -> reverse/crop`.

Dynamic shape control remains on the C++ host. Each supported window/grid pair
has an independent static ncnn core because the converted Gemm graph embeds
the flattened row count. Registry lookup never selects a nearest variant.

Five FP32 profiles are currently available under
`tests/attention_dispatcher/artifacts/attention_core`: A `[1,4,16,16,32]`, B
`[1,3,16,16,32]`, C `[1,4,17,17,32]`, D `[1,8,32,32,32]`, and E
`[1,5,19,21,32]`. The Release validation executable performs real extraction
for every profile, compares final BTHWC output with PyTorch references, and
records timing in `reports/runtime_validation.json`.

The cache uses normalized param/bin absolute paths and only retains a model
after both ncnn loads succeed. It is intentionally single-threaded for this
prototype (`ncnn::Net::opt.num_threads = 1`).
