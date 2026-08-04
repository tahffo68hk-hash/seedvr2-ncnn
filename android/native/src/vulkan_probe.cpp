#include "runtime_info.h"
#include <gpu.h>
#include <cstdio>
std::string vulkan_probe_json(){
    ncnn::create_gpu_instance(); int count=ncnn::get_gpu_count(); ncnn::destroy_gpu_instance(); char out[384];
    std::snprintf(out,sizeof(out),R"({"vulkan_compiled":true,"vulkan_device_probe_executed":true,"vulkan_inference_executed":false,"gpu_count":%d,"device_tested":true})",count); return out;
}
// Offline Phase 18A report: {"vulkan_compiled":true,"vulkan_device_probe_executed":false,"vulkan_inference_executed":false,"reason":"No Android device connected during Phase 18A"}
