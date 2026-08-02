#include "tensor_io.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

#include <net.h>

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "usage: ncnn_fixture_runner <param> <bin> <reference_input.bin> <ncnn_output.bin>\n";
        return 2;
    }

    constexpr int T = 2;
    constexpr int H = 8;
    constexpr int W = 8;
    constexpr int C = 32;
    constexpr int WINDOW_T = 2;
    constexpr int WINDOW_HW = 4;
    constexpr int ELEMENTS = T * H * W * C;

    std::vector<float> input_data;
    if (!read_float32_file(argv[3], input_data, ELEMENTS))
    {
        std::cerr << "failed to read exactly " << ELEMENTS << " floats from " << argv[3] << "\n";
        return 3;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
    {
        std::cerr << "failed to load ncnn model\n";
        return 4;
    }
    std::cerr << "model_load=PASS\n";

    const ncnn::Mat bthwc = bthwc_to_ncnn_mat(input_data, T, H, W, C);
    const ncnn::Mat windows = partition_fixed_windows(bthwc, T, H, W, C, WINDOW_T, WINDOW_HW);
    std::cerr << "layout_partition=PASS dims=" << windows.dims << ",w=" << windows.w << ",h=" << windows.h << ",c=" << windows.c << "\n";
    ncnn::Mat window_output;

    constexpr int RUNS = 10;
    double total_ms = 0.0;
    for (int i = 0; i < RUNS; ++i)
    {
        ncnn::Extractor extractor = net.create_extractor();
        extractor.set_light_mode(false);
        const int input_ret = extractor.input("in0", windows);
        std::cerr << "input_ret=" << input_ret << "\n";
        const int extract_ret = input_ret == 0 ? extractor.extract("out0", window_output) : input_ret;
        std::cerr << "extract_ret=" << extract_ret << "\n";
        if (input_ret != 0 || extract_ret != 0)
        {
            std::cerr << "ncnn inference failed\n";
            return 5;
        }
        if (i > 0)
        {
            const auto start = std::chrono::steady_clock::now();
            ncnn::Extractor timed = net.create_extractor();
            timed.set_light_mode(false);
            timed.input("in0", windows);
            timed.extract("out0", window_output);
            const auto end = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    ncnn::Mat output_bthwc(C, W, H, T);
    reverse_fixed_windows(window_output, output_bthwc, T, H, W, C, WINDOW_T, WINDOW_HW);
    std::vector<float> output_data;
    ncnn_mat_to_bthwc(output_bthwc, output_data, T, H, W, C);
    if (!write_float32_file(argv[4], output_data))
    {
        std::cerr << "failed to write output\n";
        return 6;
    }

    const std::size_t io_bytes = static_cast<std::size_t>(ELEMENTS) * sizeof(float) * 2;
    std::cout << "input_layout=[B=1,T=2,H=8,W=8,C=32], ncnn_mat=[w=32,h=8,d=8,c=2]\n";
    std::cout << "window_layout=[windows=4,tokens=32,C=32], ncnn_mat=[w=32,h=32,c=4]\n";
    std::cout << "ncnn_cpu_avg_ms=" << (total_ms / (RUNS - 1)) << " over " << (RUNS - 1) << " timed runs\n";
    std::cout << "input_output_bytes=" << io_bytes << "\n";
    return 0;
}
