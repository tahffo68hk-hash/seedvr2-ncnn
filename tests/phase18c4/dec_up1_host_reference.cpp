#include <net.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "usage: runner model.param model.bin input.f32 output.f32\n";
        return 2;
    }
    std::ifstream input_file(argv[3], std::ios::binary | std::ios::ate);
    if (!input_file) return 3;
    const size_t bytes = static_cast<size_t>(input_file.tellg());
    input_file.seekg(0);
    std::vector<float> input(bytes / sizeof(float));
    input_file.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(bytes));
    if (input.size() != 512u * 3u * 32u * 32u) return 4;

    ncnn::Net net;
    net.opt.num_threads = 5;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    int rc = net.load_param(argv[1]);
    if (!rc) rc = net.load_model(argv[2]);
    if (rc) return 10 - rc;
    ncnn::Mat source(32, 32, 3, 512, input.data());
    ncnn::Extractor extractor = net.create_extractor();
    rc = extractor.input("in0", source);
    ncnn::Mat result;
    const auto started = std::chrono::steady_clock::now();
    if (!rc) rc = extractor.extract("out0", result);
    const long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    if (rc || result.w != 64 || result.h != 64 || result.d != 5 || result.c != 512) return 20 - rc;
    const size_t plane = static_cast<size_t>(result.w) * result.h * result.d;
    std::vector<float> output(plane * result.c);
    for (int channel = 0; channel < result.c; ++channel)
        std::memcpy(output.data() + static_cast<size_t>(channel) * plane,
                    static_cast<const float*>(result.data) + static_cast<size_t>(channel) * result.cstep,
                    plane * sizeof(float));
    const bool finite = std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); });
    std::ofstream output_file(argv[4], std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!output_file || !finite) return 30;
    const auto bounds = std::minmax_element(output.begin(), output.end());
    std::cout << std::setprecision(9)
              << "{\"backend\":\"host_ncnn_cpu_fp32_original_conv3d\",\"threads\":5,"
              << "\"input_shape\":[512,3,32,32],\"output_shape\":[512,5,64,64],"
              << "\"inference_time_ms\":" << elapsed << ",\"finite\":true,\"nan_count\":0,\"inf_count\":0,"
              << "\"min\":" << *bounds.first << ",\"max\":" << *bounds.second << "}\n";
    return 0;
}
