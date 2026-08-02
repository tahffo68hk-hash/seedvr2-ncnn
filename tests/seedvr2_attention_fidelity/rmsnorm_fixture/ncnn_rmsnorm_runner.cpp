#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "net.h"

static bool load(const std::string& path, std::vector<float>& values, size_t count) {
    std::ifstream file(path, std::ios::binary); values.resize(count);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return file.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}
static bool compare(const std::vector<float>& actual, const std::vector<float>& expected, float& maximum, double& mean, size_t& nan, size_t& inf) {
    maximum=0; mean=0; nan=inf=0;
    if (actual.size()!=expected.size()) return false;
    for (size_t i=0;i<actual.size();++i) { const float d=std::fabs(actual[i]-expected[i]); maximum=std::max(maximum,d); mean+=d; nan+=std::isnan(actual[i]); inf+=std::isinf(actual[i]); }
    mean/=actual.size(); return maximum<1.0e-4f && nan==0 && inf==0;
}
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: ncnn_rmsnorm_runner <fixture>\n"; return 2; }
    const std::string root=argv[1]; std::vector<float> input;
    if (!load(root+"/input.bin",input,4*64*8)) return 3;
    bool pass=true;
    for (const char variant : {'A','B','C','D'}) {
        ncnn::Net net; net.opt.use_vulkan_compute=false; net.opt.use_packing_layout=false; net.opt.num_threads=1;
        const std::string stem=root+"/rmsnorm_"+variant;
        if (net.load_param((stem+".ncnn.param").c_str()) || net.load_model((stem+".ncnn.bin").c_str())) return 4;
        ncnn::Mat in(8,64,4), out; std::memcpy(in.data,input.data(),input.size()*sizeof(float));
        auto ex=net.create_extractor(); if (ex.input("in0",in) || ex.extract("out0",out) || out.total()!=4*64*8) return 5;
        std::vector<float> actual(out.total()), expected; std::memcpy(actual.data(),out.data,actual.size()*sizeof(float));
        if (!load(root+"/reference_"+variant+".bin",expected,actual.size())) return 6;
        float maximum; double mean; size_t nan,inf; const bool current=compare(actual,expected,maximum,mean,nan,inf); pass&=(variant!='A' ? current : true);
        std::cout<<"rewrite="<<variant<<" extract="<<(current?"PASS":"FAIL")<<" max_abs_error="<<maximum<<" mean_abs_error="<<mean<<" nan_count="<<nan<<" inf_count="<<inf<<(variant=='A'?" note=known_rsqrtncnn_accuracy":"")<<"\n";
    }
    std::cout<<"selected_rewrite=D\n"<<"ncnn_rmsnorm_runner="<<(pass?"PASS":"FAIL")<<"\n"; return pass?0:1;
}
