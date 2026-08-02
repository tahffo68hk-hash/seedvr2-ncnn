#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <net.h>

static bool read_f32(const char* path, std::vector<float>& values, std::size_t count)
{
    std::ifstream file(path, std::ios::binary);
    values.resize(count);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return file.good() || file.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "usage: ncnn_qkv_debug <linear|linear2d|split|reshape|transpose> <param> <bin> <input.bin>\n";
        return 2;
    }
    const std::string test_case = argv[1];
    const int outputs = (test_case == "linear" || test_case == "linear2d") ? 1 : 3;
    if (test_case != "linear" && test_case != "linear2d" && test_case != "split" && test_case != "reshape" && test_case != "transpose")
        return 2;

    std::vector<float> data;
    const bool is_2d = test_case == "linear2d";
    if (!read_f32(argv[4], data, (is_2d ? 1 : 4) * 32 * 32))
        return 3;

    // Logical [windows=4,tokens=32,channels=32] -> ncnn Mat(w=32,h=32,c=4).
    ncnn::Mat input = is_2d ? ncnn::Mat(32, 32) : ncnn::Mat(32, 32, 4);
    std::memcpy(input.data, data.data(), data.size() * sizeof(float));

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 1;
    if (net.load_param(argv[2]) != 0 || net.load_model(argv[3]) != 0)
    {
        std::cerr << "stage=load FAIL\n";
        return 4;
    }
    std::cerr << "stage=load PASS\n";
    ncnn::Extractor ex = net.create_extractor();
    const int input_ret = ex.input("in0", input);
    std::cerr << "stage=input ret=" << input_ret << "\n";
    if (input_ret != 0)
        return 5;

    for (int i = 0; i < outputs; ++i)
    {
        ncnn::Mat output;
        const std::string name = "out" + std::to_string(i);
        const int ret = ex.extract(name.c_str(), output);
        std::cerr << "stage=extract output=" << name << " ret=" << ret << "\n";
        if (ret != 0)
            return 6;
        std::cout << "case=" << test_case << " output=" << name << " dims=" << output.dims << " w=" << output.w << " h=" << output.h << " d=" << output.d << " c=" << output.c << " total=" << output.total() << "\n";
    }
    return 0;
}
