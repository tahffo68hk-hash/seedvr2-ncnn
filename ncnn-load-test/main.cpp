#include <iostream>
#include "net.h"

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    ncnn::Net net;
    int param_ret = net.load_param(argv[1]);
    int model_ret = net.load_model(argv[2]);
    std::cout << "param_ret=" << param_ret << " model_ret=" << model_ret << '\n';
    return (param_ret == 0 && model_ret == 0) ? 0 : 1;
}
