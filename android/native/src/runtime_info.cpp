#include "runtime_info.h"
#include <cpu.h>
#include <net.h>
#define S2(x) #x
#define S(x) S2(x)
std::string runtime_version_json(){return R"({"runtime":"SeedEdge Android Runtime","version":"18A.0","phase":"18A","abi":"arm64-v8a","device_tested":false})";}
std::string runtime_build_json(){return std::string(R"({"runtime":"SeedEdge Android Runtime","android_runtime_stage":"phase18b","scaffold_baseline":"phase18a","abi":"arm64-v8a","ncnn_enabled":true,"vulkan_compiled":true,"device_execution_supported":true,"device_execution_observed":true,"build_date":")")+__DATE__+R"("})";}
std::string ncnn_info_json(){
#ifdef _OPENMP
    const bool openmp = true;
#else
    const bool openmp = false;
#endif
    ncnn::Option option;
    return std::string("{\"ncnn_enabled\":true,\"ncnn_version\":\"20260730\",\"ncnn_commit\":\"a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a\",\"openmp_compiled\":")+(openmp?"true":"false")+",\"openmp_runtime_available\":"+(openmp?"true":"false")+",\"cpu_logical_cores\":"+std::to_string(ncnn::get_cpu_count())+",\"ncnn_thread_capacity\":"+std::to_string(ncnn::get_big_cpu_count())+",\"configured_threads\":"+std::to_string(option.num_threads)+",\"performance_mode\":\"balanced\",\"exceptions\":true,\"rtti\":false,\"vulkan_compiled\":true,\"device_execution_supported\":true,\"device_execution_observed\":true}";
}
