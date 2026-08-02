#include "attention_runtime.h"
#include <chrono>
#include <cstring>
namespace attention_dispatcher {
namespace { double elapsed(std::chrono::steady_clock::time_point start) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); } }
RuntimeResult run_attention(const AttentionRequest& request, const std::vector<float>& input, const ArtifactRegistry& registry, NcnnModelCache& cache) {
    RuntimeResult r; const auto total = std::chrono::steady_clock::now(); const auto metadata_start = total; const CoreInfo* core = nullptr;
    if (!validate_attention_request(request, r.error_message)) { r.metadata_ms = elapsed(metadata_start); r.total_runtime_ms = elapsed(total); return r; }
    try { r.metadata = dispatch_shape({request.batch, request.time, request.height, request.width, request.channels}); } catch (const std::exception& error) { r.error_message = error.what(); r.metadata_ms = elapsed(metadata_start); r.total_runtime_ms = elapsed(total); return r; }
    r.metadata_ms = elapsed(metadata_start); const auto lookup_start = std::chrono::steady_clock::now(); CoreKey key = make_core_key(r.metadata); key.variant = request.variant; core = registry.lookup(key); r.registry_lookup_ms = elapsed(lookup_start);
    if (!core) { Metadata ignored{}; dispatch_request(request, registry, ignored, core, r.error_message); r.total_runtime_ms = elapsed(total); return r; }
    r.selected_core = core->name;
    if (!core->single_model()) { r.error_message = "baseline runtime requires a single-model executor; core=" + core->name + " variant=" + core->key.variant + " execution=staged_host_head_merge"; r.total_runtime_ms = elapsed(total); return r; }
    const std::size_t expected = static_cast<std::size_t>(request.batch) * request.time * request.height * request.width * request.channels;
    if (input.size() != expected) { r.error_message = "input size does not match request"; r.total_runtime_ms = elapsed(total); return r; }
    const auto load_start = std::chrono::steady_clock::now(); ncnn::Net* net = cache.get_or_load(*core, r.model_load_ms, r.error_message); (void)load_start;
    if (!net) { r.total_runtime_ms = elapsed(total); return r; }
    const auto partition_start = std::chrono::steady_clock::now(); const auto windows_data = partition_bthwc(input, r.metadata); r.partition_ms = elapsed(partition_start);
    const auto mask_start = std::chrono::steady_clock::now(); const auto mask_data = make_additive_key_mask(r.metadata); r.mask_generation_ms = elapsed(mask_start);
    ncnn::Mat windows(request.channels, r.metadata.tokens_per_window, r.metadata.num_windows), mask(r.metadata.tokens_per_window, r.metadata.tokens_per_window, r.metadata.num_windows);
    std::memcpy(windows.data, windows_data.data(), windows_data.size() * sizeof(float)); std::memcpy(mask.data, mask_data.data(), mask_data.size() * sizeof(float));
    const auto infer_start = std::chrono::steady_clock::now(); ncnn::Extractor ex = net->create_extractor(); ex.set_light_mode(false); ncnn::Mat result;
    if (ex.input("in0", windows) != 0 || ex.input("in1", mask) != 0 || ex.extract("out0", result) != 0) { r.ncnn_inference_ms = elapsed(infer_start); r.error_message = "ncnn extract failed"; r.total_runtime_ms = elapsed(total); return r; }
    r.ncnn_inference_ms = elapsed(infer_start); if (result.dims != 3 || result.w != request.channels || result.h != r.metadata.tokens_per_window || result.c != r.metadata.num_windows) { r.error_message = "unexpected ncnn output shape"; r.total_runtime_ms = elapsed(total); return r; }
    std::vector<float> output_windows(result.total()); std::memcpy(output_windows.data(), result.data, output_windows.size() * sizeof(float)); const auto reverse_start = std::chrono::steady_clock::now(); r.output = reverse_crop_bthwc(output_windows, r.metadata); r.reverse_crop_ms = elapsed(reverse_start);
    r.success = true; r.total_runtime_ms = elapsed(total); return r;
}
}
