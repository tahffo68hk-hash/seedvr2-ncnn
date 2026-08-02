#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <net.h>

namespace fs = std::filesystem;
struct Stats { float max_abs = 0.f, max_relative = 0.f; double mean_abs = 0.; size_t index = 0; float reference = 0.f, actual = 0.f; };
static bool read(const fs::path& path, std::vector<float>& values, size_t count) { std::ifstream file(path, std::ios::binary); values.resize(count); file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float))); return file.gcount() == static_cast<std::streamsize>(count * sizeof(float)); }
static Stats compare(const std::vector<float>& actual, const std::vector<float>& reference) { Stats result; for (size_t i = 0; i < actual.size(); ++i) { const float error = std::fabs(actual[i] - reference[i]); result.mean_abs += error; result.max_relative = std::max(result.max_relative, error / std::max(1.e-12f, std::fabs(reference[i]))); if (error > result.max_abs) { result.max_abs = error; result.index = i; result.reference = reference[i]; result.actual = actual[i]; } } result.mean_abs /= actual.size(); return result; }
static Stats compare_slice(const std::vector<float>& actual, const std::vector<float>& reference, size_t offset, size_t count) { return compare(std::vector<float>(actual.begin() + offset, actual.begin() + offset + count), std::vector<float>(reference.begin() + offset, reference.begin() + offset + count)); }
static void range(const std::vector<float>& values, float& minimum, float& maximum, double& mean) { minimum = std::numeric_limits<float>::infinity(); maximum = -minimum; mean = 0.; for (float value : values) { minimum = std::min(minimum, value); maximum = std::max(maximum, value); mean += value; } mean /= values.size(); }
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: ncnn_rms_head_stages_runner <stage_root>\n"; return 2; }
    const fs::path root = argv[1]; std::vector<float> input, mask, cos, sin;
    if (!read(root / "input.bin", input, 64 * 32) || !read(root / "mask.bin", mask, 64 * 64) || !read(root / "rope_cos.bin", cos, 64 * 4) || !read(root / "rope_sin.bin", sin, 64 * 4)) return 3;
    ncnn::Mat in0(32, 64), in1(64, 64), in2(4, 64), in3(4, 64); std::memcpy(in0.data, input.data(), input.size() * sizeof(float)); std::memcpy(in1.data, mask.data(), mask.size() * sizeof(float)); std::memcpy(in2.data, cos.data(), cos.size() * sizeof(float)); std::memcpy(in3.data, sin.data(), sin.size() * sizeof(float));
    const size_t counts[] = {0, 64 * 24, 64 * 24, 64 * 24, 64 * 64, 64 * 64, 64 * 64, 64 * 8};
    bool extract_ok = true; int first_bad = 0;
    for (int stage = 1; stage <= 7; ++stage) {
        ncnn::Net net; net.opt.use_packing_layout = false; net.opt.num_threads = 1;
        const fs::path directory = root / ("stage" + std::to_string(stage));
        if (net.load_param((directory / "model.ncnn.param").string().c_str()) || net.load_model((directory / "model.ncnn.bin").string().c_str())) return 4;
        ncnn::Extractor extractor = net.create_extractor(); ncnn::Mat output; int status = extractor.input("in0", in0);
        if (stage >= 3) { status |= extractor.input("in1", in2); status |= extractor.input("in2", in3); }
        if (stage >= 5) { status = extractor.input("in0", in0); status |= extractor.input("in1", in1); status |= extractor.input("in2", in2); status |= extractor.input("in3", in3); }
        status |= extractor.extract("out0", output); std::vector<float> reference;
        if (status || output.total() != counts[stage] || !read(directory / "reference.bin", reference, counts[stage])) return 5;
        std::vector<float> actual(counts[stage]); std::memcpy(actual.data(), output.data, actual.size() * sizeof(float)); const Stats stats = compare(actual, reference);
        std::cout << "stage" << stage << " max_abs_error=" << stats.max_abs << " mean_abs_error=" << stats.mean_abs << " max_relative_error=" << stats.max_relative << " first_bad_index=" << stats.index << " reference=" << stats.reference << " ncnn=" << stats.actual << '\n';
        if (stage <= 3) { const char* labels[][3] = {{"","",""}, {"q_raw","k_raw","v_raw"}, {"q_rms","k_rms","v_raw"}, {"q_rope","k_rope","v_raw"}}; for (int part = 0; part < 3; ++part) { const Stats slice = compare_slice(actual, reference, static_cast<size_t>(part) * 512, 512); std::cout << labels[stage][part] << " max_abs_error=" << slice.max_abs << " mean_abs_error=" << slice.mean_abs << " max_relative_error=" << slice.max_relative << " first_bad_index=" << slice.index << " reference=" << slice.reference << " ncnn=" << slice.actual << '\n'; } }
        if (stage >= 4) { float rmin, rmax, amin, amax; double rmean, amean; range(reference, rmin, rmax, rmean); range(actual, amin, amax, amean); std::cout << "stage" << stage << " reference_range=[" << rmin << ',' << rmax << "] reference_mean=" << rmean << " ncnn_range=[" << amin << ',' << amax << "] ncnn_mean=" << amean << '\n'; }
        if (!first_bad && stats.max_abs >= 1.e-4f) first_bad = stage;
    }
    std::cout << "first_over_threshold_stage=" << first_bad << " ncnn_rms_head_stages_runner=" << (extract_ok ? "PASS" : "FAIL") << '\n'; return extract_ok ? 0 : 1;
}
