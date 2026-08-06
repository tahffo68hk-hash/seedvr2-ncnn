#include "model_manifest.h"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: phase18d_model_bundle_test <model-root>\n";
        return 2;
    }
    const std::string result = validate_model_manifest(argv[1]);
    std::cout << result << '\n';
    return result.find("\"valid\":true") != std::string::npos ? 0 : 1;
}
