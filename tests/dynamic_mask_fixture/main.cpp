#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <net.h>

namespace
{
bool read_f32(const char* path, std::vector<float>& values, std::size_t expected)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    values.resize(expected);
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected * sizeof(float)));
    return stream.gcount() == static_cast<std::streamsize>(expected * sizeof(float));
}

bool write_f32(const char* path, const ncnn::Mat& tensor)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    stream.write(reinterpret_cast<const char*>(tensor.data), static_cast<std::streamsize>(tensor.total() * sizeof(float)));
    return static_cast<bool>(stream);
}
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "usage: ncnn_dynamic_mask_runner <param> <bin> <windows.bin> <mask.bin> <output.bin>\n";
        return 2;
    }
    constexpr int WINDOWS = 25;
    constexpr int TOKENS = 64;
    constexpr int CHANNELS = 32;
    std::vector<float> windows_data;
    std::vector<float> mask_data;
    if (!read_f32(argv[3], windows_data, WINDOWS * TOKENS * CHANNELS) || !read_f32(argv[4], mask_data, WINDOWS * TOKENS * TOKENS))
    {
        std::cerr << "input read failed\n";
        return 3;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
    {
        std::cerr << "model load failed\n";
        return 4;
    }
    ncnn::Mat windows(CHANNELS, TOKENS, WINDOWS);
    ncnn::Mat mask(TOKENS, TOKENS, WINDOWS);
    std::memcpy(windows.data, windows_data.data(), windows_data.size() * sizeof(float));
    std::memcpy(mask.data, mask_data.data(), mask_data.size() * sizeof(float));

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", windows) != 0 || extractor.input("in1", mask) != 0)
    {
        std::cerr << "input failed\n";
        return 5;
    }
    ncnn::Mat output;
    const int extract_ret = extractor.extract("out0", output);
    if (extract_ret != 0 || output.dims != 3 || output.w != CHANNELS || output.h != TOKENS || output.c != WINDOWS)
    {
        std::cerr << "extract failed ret=" << extract_ret << " dims=" << output.dims << " w=" << output.w << " h=" << output.h << " c=" << output.c << "\n";
        return 6;
    }
    if (!write_f32(argv[5], output))
    {
        std::cerr << "output write failed\n";
        return 7;
    }
    std::cout << "extract=PASS windows=[25,64,32] mask=[25,64,64] output=[25,64,32]\n";
    return 0;
}
