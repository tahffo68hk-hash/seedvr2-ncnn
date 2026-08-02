#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <net.h>

namespace
{
bool read_f32(const std::string& path, std::vector<float>& values, size_t count)
{
    values.resize(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return file && file.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}

bool compare(const ncnn::Mat& actual, const std::vector<float>& expected, const char* label)
{
    if (actual.total() != expected.size())
    {
        std::cerr << label << " output count=" << actual.total() << " expected=" << expected.size() << "\n";
        return false;
    }
    const float* data = static_cast<const float*>(actual.data);
    float max_error = 0.f;
    double sum_error = 0.;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        const float error = std::fabs(data[i] - expected[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }
    std::cout << label << " max_abs_error=" << max_error << " mean_abs_error=" << sum_error / expected.size() << "\n";
    return max_error < 1.e-4f;
}

bool run_qk(const std::string& root)
{
    std::vector<float> input, expected;
    if (!read_f32(root + "/tokens.bin", input, 64 * 4 * 8) || !read_f32(root + "/qk_norm_output.bin", expected, 64 * 4 * 8)) return false;
    ncnn::Net net;
    net.opt.use_packing_layout = false;
    if (net.load_param((root + "/qk_norm.ncnn.param").c_str()) || net.load_model((root + "/qk_norm.ncnn.bin").c_str())) return false;
    ncnn::Mat in(8, 4, 64); std::memcpy(in.data, input.data(), input.size() * sizeof(float));
    auto ex = net.create_extractor(); ncnn::Mat out;
    return ex.input("in0", in) == 0 && ex.extract("out0", out) == 0 && compare(out, expected, "qk_norm");
}

bool run_rope(const std::string& root)
{
    std::vector<float> input, phases, expected;
    if (!read_f32(root + "/tokens.bin", input, 64 * 4 * 8) || !read_f32(root + "/phases.bin", phases, 64 * 4) || !read_f32(root + "/rope_output.bin", expected, 64 * 4 * 8)) return false;
    ncnn::Net net;
    net.opt.use_packing_layout = false;
    if (net.load_param((root + "/rope_apply.ncnn.param").c_str()) || net.load_model((root + "/rope_apply.ncnn.bin").c_str())) return false;
    ncnn::Mat in0(8, 4, 64), in1(4, 64); std::memcpy(in0.data, input.data(), input.size() * sizeof(float)); std::memcpy(in1.data, phases.data(), phases.size() * sizeof(float));
    auto ex = net.create_extractor(); ncnn::Mat out;
    return ex.input("in0", in0) == 0 && ex.input("in1", in1) == 0 && ex.extract("out0", out) == 0 && compare(out, expected, "rope_apply");
}

bool run_combined(const std::string& root)
{
    std::vector<float> windows, phases, q, k, v;
    if (!read_f32(root + "/windows.bin", windows, 4 * 64 * 32) || !read_f32(root + "/phases.bin", phases, 64 * 4) ||
        !read_f32(root + "/qkv_norm_rope_q.bin", q, 4 * 64 * 4 * 8) || !read_f32(root + "/qkv_norm_rope_k.bin", k, 4 * 64 * 4 * 8) || !read_f32(root + "/qkv_norm_rope_v.bin", v, 4 * 64 * 4 * 8)) return false;
    ncnn::Net net;
    net.opt.use_packing_layout = false;
    if (net.load_param((root + "/qkv_norm_rope.ncnn.param").c_str()) || net.load_model((root + "/qkv_norm_rope.ncnn.bin").c_str())) return false;
    ncnn::Mat in0(32, 64, 4), in1(4, 64); std::memcpy(in0.data, windows.data(), windows.size() * sizeof(float)); std::memcpy(in1.data, phases.data(), phases.size() * sizeof(float));
    auto ex = net.create_extractor(); ncnn::Mat oq, ok, ov;
    return ex.input("in0", in0) == 0 && ex.input("in1", in1) == 0 && ex.extract("out0", oq) == 0 && ex.extract("out1", ok) == 0 && ex.extract("out2", ov) == 0 &&
           compare(oq, q, "qkv_norm_rope.q") && compare(ok, k, "qkv_norm_rope.k") && compare(ov, v, "qkv_norm_rope.v");
}
}

int main(int argc, char** argv)
{
    if (argc != 2) { std::cerr << "usage: ncnn_fidelity_runner <artifacts>\n"; return 2; }
    const std::string root = argv[1];
    const bool pass = run_qk(root) && run_rope(root) && run_combined(root);
    std::cout << "ncnn_fidelity_runner=" << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}
