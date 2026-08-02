#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <net.h>
#include "artifact_registry.h"
#include "attention_dispatcher.h"

namespace {
bool read_f32(const std::filesystem::path& path, std::vector<float>& data, std::size_t count) {
    std::ifstream stream(path, std::ios::binary); if (!stream) return false;
    data.resize(count); stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return stream.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}
bool write_f32(const std::filesystem::path& path, const std::vector<float>& data) {
    std::ofstream stream(path, std::ios::binary); if (!stream) return false;
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float))); return static_cast<bool>(stream);
}
}
int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "usage: attention_dispatcher_runner <artifact_root> <input.bin> <reference.bin> <output.bin>\n"; return 2; }
    try {
        namespace ad = attention_dispatcher;
        ad::ArtifactRegistry registry; std::string error;
        if (!registry.scan(argv[1], error)) { std::cerr << "registry scan failed: " << error << '\n'; return 3; }
        std::cout << "registry_cores=" << registry.size() << '\n';
        const ad::AttentionRequest request{1, 4, 17, 17, 32, "fp32"}; ad::Metadata metadata{}; const ad::CoreInfo* core = nullptr;
        if (!ad::dispatch_request(request, registry, metadata, core, error)) { std::cerr << "dispatch failed: " << error << '\n'; return 4; }
        std::cout << "request=[1,4,17,17,32]\nwindow=" << metadata.window_t << 'x' << metadata.window_h << 'x' << metadata.window_w
                  << "\ngrid=" << metadata.grid_t << 'x' << metadata.grid_h << 'x' << metadata.grid_w << "\nselected_core=" << core->name << '\n';
        const std::size_t input_size = static_cast<std::size_t>(request.batch) * request.time * request.height * request.width * request.channels;
        std::vector<float> input, reference; if (!read_f32(argv[2], input, input_size) || !read_f32(argv[3], reference, input_size)) { std::cerr << "input/reference read failed\n"; return 5; }
        const ad::SingleModelArtifact* single = core->single_model();
        if (!single) { std::cerr << "baseline runner requires a single-model executor; core=" << core->name << " variant=" << core->key.variant << "\n"; return 6; }
        const std::vector<float> windows_data = ad::partition_bthwc(input, metadata); const std::vector<float> mask_data = ad::make_additive_key_mask(metadata);
        ncnn::Net net; net.opt.use_vulkan_compute = false; net.opt.num_threads = 1;
        const std::string param = single->model.param_path.string(), bin = single->model.bin_path.string();
        if (net.load_param(param.c_str()) != 0 || net.load_model(bin.c_str()) != 0) { std::cerr << "model load failed\n"; return 7; }
        ncnn::Mat windows(request.channels, metadata.tokens_per_window, metadata.num_windows), mask(metadata.tokens_per_window, metadata.tokens_per_window, metadata.num_windows);
        std::memcpy(windows.data, windows_data.data(), windows_data.size() * sizeof(float)); std::memcpy(mask.data, mask_data.data(), mask_data.size() * sizeof(float));
        ncnn::Extractor extractor = net.create_extractor(); extractor.set_light_mode(false); ncnn::Mat result;
        if (extractor.input("in0", windows) != 0 || extractor.input("in1", mask) != 0 || extractor.extract("out0", result) != 0) { std::cerr << "extract=FAIL\n"; return 8; }
        if (result.dims != 3 || result.w != request.channels || result.h != metadata.tokens_per_window || result.c != metadata.num_windows) { std::cerr << "unexpected output shape\n"; return 8; }
        std::vector<float> output_windows(result.total()); std::memcpy(output_windows.data(), result.data, output_windows.size() * sizeof(float));
        const std::vector<float> output = ad::reverse_crop_bthwc(output_windows, metadata); if (!write_f32(argv[4], output)) { std::cerr << "output write failed\n"; return 9; }
        double total_error = 0.0; float max_error = 0.f; std::size_t nan_count = 0, inf_count = 0;
        for (std::size_t i = 0; i < output.size(); ++i) { if (std::isnan(output[i])) ++nan_count; if (std::isinf(output[i])) ++inf_count; const float difference = std::fabs(output[i] - reference[i]); max_error = std::max(max_error, difference); total_error += difference; }
        const double mean_error = total_error / output.size(); std::cout << "extract=PASS\nmax_abs_error=" << max_error << "\nmean_abs_error=" << mean_error << "\nnan_count=" << nan_count << "\ninf_count=" << inf_count << '\n';
        return max_error < 1e-4f && nan_count == 0 && inf_count == 0 ? 0 : 10;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 11; }
}
