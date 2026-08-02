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

static bool write_f32(const char* path, const ncnn::Mat& mat, std::size_t count)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
        return false;
    file.write(reinterpret_cast<const char*>(mat.data), static_cast<std::streamsize>(count * sizeof(float)));
    return file.good();
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "usage: ncnn_attention_minimal <matmul|softmax|attention> <param> <bin> <input.bin> <output.bin>\n";
        return 2;
    }

    const std::string test_case = argv[1];
    const bool score_output = test_case == "matmul" || test_case == "softmax";
    if (!score_output && test_case != "attention")
        return 2;

    constexpr int HEADS = 4;
    constexpr int TOKENS = 32;
    constexpr int DIM = 8;
    std::vector<float> input_data;
    if (!read_f32(argv[4], input_data, HEADS * TOKENS * DIM))
    {
        std::cerr << "stage=read_input FAIL\n";
        return 3;
    }

    // Logical [heads,tokens,dim] maps to ncnn Mat(w=dim,h=tokens,c=heads).
    ncnn::Mat input(DIM, TOKENS, HEADS);
    std::memcpy(input.data, input_data.data(), input_data.size() * sizeof(float));

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 1;
    if (net.load_param(argv[2]) != 0 || net.load_model(argv[3]) != 0)
    {
        std::cerr << "stage=load_model FAIL\n";
        return 4;
    }
    std::cerr << "stage=load_model PASS\n";

    ncnn::Extractor ex = net.create_extractor();
    const int input_ret = ex.input("in0", input);
    std::cerr << "stage=input ret=" << input_ret << " dims=" << input.dims << " w=" << input.w << " h=" << input.h << " c=" << input.c << "\n";
    if (input_ret != 0)
        return 5;

    ncnn::Mat output;
    const int extract_ret = ex.extract("out0", output);
    std::cerr << "stage=extract ret=" << extract_ret << "\n";
    if (extract_ret != 0)
        return 6;

    const std::size_t output_count = score_output ? HEADS * TOKENS * TOKENS : HEADS * TOKENS * DIM;
    if (!output.data || output.total() != output_count)
    {
        std::cerr << "stage=validate_output FAIL total=" << output.total() << " expected=" << output_count << "\n";
        return 7;
    }
    if (!write_f32(argv[5], output, output_count))
        return 8;

    std::cout << "case=" << test_case << " PASS output_dims=" << output.dims << " w=" << output.w << " h=" << output.h << " c=" << output.c << "\n";
    return 0;
}
