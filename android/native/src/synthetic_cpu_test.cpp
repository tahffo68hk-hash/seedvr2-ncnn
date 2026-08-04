#include "runtime_info.h"
#include <net.h>
#include <cmath>
#include <cstdio>

static bool run_once(float& max_error) {
    ncnn::Net net;
    net.opt.use_vulkan_compute=false; net.opt.num_threads=1;
    const char param[] = "7767517\n2 2\nInput input 0 1 in\nReLU relu 1 1 in out 0=0.1\n";
    if (net.load_param_mem(param) != 0) return false;
    ncnn::Mat input(4); float values[4]={-2.f,-1.f,2.f,3.f}; for(int i=0;i<4;i++) input[i]=values[i];
    ncnn::Extractor ex=net.create_extractor();
    ncnn::Mat out; if(ex.input("in",input)!=0 || ex.extract("out",out)!=0 || out.total()!=4) return false;
    const float expected[4]={-.2f,-.1f,2.f,3.f}; max_error=0.f;
    for(int i=0;i<4;i++){float v=out[i]; if(!std::isfinite(v)) return false; max_error=std::fmax(max_error,std::fabs(v-expected[i]));}
    net.clear(); return max_error <= 1e-6f;
}
std::string synthetic_cpu_test_json(bool device_execution){float a=0,b=0;bool p=run_once(a)&&run_once(b)&&std::fabs(a-b)==0;char s[384];std::snprintf(s,sizeof(s),R"({"backend":"ncnn_cpu","passed":%s,"shape":[4],"finite":%s,"max_absolute_error":%.9g,"repeat_consistent":%s,"android_binary_built":true,"android_device_executed":%s,"device_tested":%s})",p?"true":"false",p?"true":"false",a,p?"true":"false",device_execution?"true":"false",device_execution?"true":"false");return s;}
