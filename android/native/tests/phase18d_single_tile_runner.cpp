#include "minimal_video_pipeline.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        std::cerr << "usage: phase18d_single_tile_runner <model-root> <task-root> <rgb24-file> <width> <height> <request-json>\n";
        return 2;
    }
    const int width = std::stoi(argv[4]);
    const int height = std::stoi(argv[5]);
    std::ifstream input(argv[3], std::ios::binary);
    const std::vector<uint8_t> rgb((std::istreambuf_iterator<char>(input)), {});
    const std::string result = run_phase18d_single_photo_tile(argv[1], argv[2], rgb.data(), rgb.size(),
                                                                width, height, argv[6]);
    std::cout << result << '\n';
    return result.find("\"overall_pass\":true") != std::string::npos ? 0 : 1;
}
