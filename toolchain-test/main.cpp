#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

int main()
{
    std::vector<int> values{1, 2, 3};
    std::string_view message = "MSVC C++17 toolchain works";
    std::filesystem::path path = "seedvr2-ncnn";

    std::cout << message << '\n';
    std::cout << "path=" << path.string() << '\n';
    std::cout << "size=" << values.size() << '\n';
    return 0;
}
