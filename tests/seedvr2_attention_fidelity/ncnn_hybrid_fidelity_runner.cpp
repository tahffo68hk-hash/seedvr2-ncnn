#include "host_head_merge.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <net.h>

namespace
{
bool read_floats(const std::string& path, std::vector<float>& values, std::size_t count)
{
    values.resize(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return stream && stream.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}

bool compare(const char* stage, const std::vector<float>& actual, const std::vector<float>& expected)
{
    if (actual.size() != expected.size()) return false;
    float maximum = 0.f; double sum = 0.; std::size_t nan = 0, inf = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const float error = std::fabs(actual[i] - expected[i]);
        maximum = std::max(maximum, error); sum += error;
        nan += std::isnan(actual[i]); inf += std::isinf(actual[i]);
    }
    std::cout << stage << " max_abs_error=" << maximum << " mean_abs_error=" << sum / actual.size()
              << " nan_count=" << nan << " inf_count=" << inf << "\n";
    return maximum < 1e-4f && nan == 0 && inf == 0;
}

std::vector<float> reverse_windows(const std::vector<float>& windows)
{
    std::vector<float> output(8192);
    for (int window = 0; window < 4; ++window)
        for (int token = 0; token < 64; ++token)
            for (int channel = 0; channel < 32; ++channel)
                output[((token / 16 * 8 + (window / 2) * 4 + (token / 4) % 4) * 8 +
                        (window % 2) * 4 + token % 4) * 32 + channel] =
                    windows[(window * 64 + token) * 32 + channel];
    return output;
}
}

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const std::string root = argv[1]; const std::string reference = root + "/reference";
    std::vector<float> windows, mask, cos, sin, merged_reference, projection_reference, windows_reference, output_reference;
    if (!read_floats(reference + "/windows.bin", windows, 8192) || !read_floats(reference + "/mask.bin", mask, 16384) ||
        !read_floats(reference + "/rope_cos.bin", cos, 256) || !read_floats(reference + "/rope_sin.bin", sin, 256) ||
        !read_floats(reference + "/merged.bin", merged_reference, 8192) || !read_floats(reference + "/projection_2d.bin", projection_reference, 8192) ||
        !read_floats(reference + "/windows_output.bin", windows_reference, 8192) || !read_floats(reference + "/bthwc_output.bin", output_reference, 8192)) return 3;

    ncnn::Net projection_net; projection_net.opt.use_packing_layout = false;
    if (projection_net.load_param((root + "/output_projection/model.ncnn.param").c_str()) || projection_net.load_model((root + "/output_projection/model.ncnn.bin").c_str())) return 4;
    std::vector<ncnn::Net> head_nets(4);
    for (int head = 0; head < 4; ++head)
    {
        head_nets[head].opt.use_packing_layout = false;
        const std::string directory = root + "/attention_heads/head" + std::to_string(head);
        if (head_nets[head].load_param((directory + "/model.ncnn.param").c_str()) || head_nets[head].load_model((directory + "/model.ncnn.bin").c_str())) return 4;
    }
    std::vector<std::vector<float>> heads(4); bool ok = true;
    for (int head = 0; head < 4; ++head)
    {
        heads[head].resize(2048);
        for (int window = 0; window < 4; ++window)
        {
            ncnn::Mat input(32, 64), attention_mask(64, 64), rope_cos(4, 64), rope_sin(4, 64), output;
            std::memcpy(input.data, windows.data() + window * 2048, 2048 * sizeof(float));
            std::memcpy(attention_mask.data, mask.data() + window * 4096, 4096 * sizeof(float));
            std::memcpy(rope_cos.data, cos.data(), 256 * sizeof(float)); std::memcpy(rope_sin.data, sin.data(), 256 * sizeof(float));
            ncnn::Extractor extractor = head_nets[head].create_extractor();
            const int status = extractor.input("in0", input) || extractor.input("in1", attention_mask) || extractor.input("in2", rope_cos) || extractor.input("in3", rope_sin) || extractor.extract("out0", output);
            if (status || output.total() != 512)
            {
                std::cerr << "extract head" << head << " window=" << window << " status=" << status << " total=" << output.total() << " dims=" << output.dims
                          << " w=" << output.w << " h=" << output.h << " c=" << output.c << " d=" << output.d << "\n";
                return 6;
            }
            std::memcpy(heads[head].data() + window * 512, output.data, 512 * sizeof(float));
        }
        std::vector<float> expected; if (!read_floats(reference + "/head" + std::to_string(head) + ".bin", expected, 2048)) return 3;
        ok &= compare(("head" + std::to_string(head)).c_str(), heads[head], expected);
    }
    std::vector<float> merged; std::string merge_error;
    if (!seedvr2::merge_attention_heads(heads, 4, 64, 8, merged, merge_error)) { std::cerr << merge_error << "\n"; return 8; }
    ok &= compare("host_merged", merged, merged_reference);

    ncnn::Mat projection_input(32, 256), projection_output; std::memcpy(projection_input.data, merged.data(), 8192 * sizeof(float));
    ncnn::Extractor projection_extractor = projection_net.create_extractor();
    if (projection_extractor.input("in0", projection_input) || projection_extractor.extract("out0", projection_output) || projection_output.total() != 8192) return 7;
    std::vector<float> flat(static_cast<float*>(projection_output.data), static_cast<float*>(projection_output.data) + 8192);
    ok &= compare("projection_2d", flat, projection_reference);
    ok &= compare("windows_output", flat, windows_reference);
    ok &= compare("bthwc_output", reverse_windows(flat), output_reference);
    std::cout << "ncnn_hybrid_fidelity_runner=" << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
