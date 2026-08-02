#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <net.h>

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "usage: qkv_graph_rewrite <original|rewrite> <param> <bin> <input.bin> <output.bin>\n";
        return 2;
    }
    std::ifstream input_file(argv[4], std::ios::binary);
    std::vector<float> input_data(4 * 32 * 32);
    input_file.read(reinterpret_cast<char*>(input_data.data()), static_cast<std::streamsize>(input_data.size() * sizeof(float)));
    if (!input_file || input_file.gcount() != static_cast<std::streamsize>(input_data.size() * sizeof(float)))
        return 3;

    // Logical [windows=4,tokens=32,C=32] -> ncnn Mat(w=32,h=32,c=4).
    ncnn::Mat input(32, 32, 4);
    std::memcpy(input.data, input_data.data(), input_data.size() * sizeof(float));
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 1;
    if (net.load_param(argv[2]) != 0 || net.load_model(argv[3]) != 0)
        return 4;
    std::cerr << "stage=load PASS\n";
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", input) != 0)
        return 5;
    std::cerr << "stage=input PASS\n";
    ncnn::Mat output;
    const int extract_ret = ex.extract("out0", output);
    std::cerr << "stage=extract ret=" << extract_ret << "\n";
    if (extract_ret != 0)
        return 6;
    std::cerr << "stage=output dims=" << output.dims << " w=" << output.w << " h=" << output.h << " d=" << output.d << " c=" << output.c << " total=" << output.total() << "\n";
    if (output.total() != 4 * 32 * 96)
    {
        std::cerr << "unexpected output total=" << output.total() << "\n";
        return 7;
    }
    std::ofstream output_file(argv[5], std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(output.data), static_cast<std::streamsize>(output.total() * sizeof(float)));
    if (!output_file)
        return 8;
    std::cout << "case=" << argv[1] << " PASS dims=" << output.dims << " w=" << output.w << " h=" << output.h << " c=" << output.c << "\n";
    return 0;
}
