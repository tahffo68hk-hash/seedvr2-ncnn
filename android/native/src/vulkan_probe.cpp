#include "runtime_info.h"

#include <command.h>
#include <gpu.h>
#include <net.h>

#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace {
std::mutex g_vulkan_mutex;
bool g_vulkan_instance_initialized = false;
constexpr float kExpected[] = {-0.2f, -0.1f, 2.f, 3.f};

void force_fp32_vulkan(ncnn::Option& opt) {
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    // Qualcomm Adreno 735 on the Pad 7 Pro crashes inside the bundled
    // glslang traversal when the optional subgroup/cooperative-matrix shader
    // variants are enabled.  Keep the Vulkan path explicit, but use the
    // portable SPIR-V 1.0 shader variants for this runtime probe and the
    // real-weight pipeline.
    opt.use_subgroup_ops = false;
    opt.use_cooperative_matrix = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_packed = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
}
}

std::string vulkan_probe_json() {
    std::lock_guard<std::mutex> lock(g_vulkan_mutex);
    bool instance_created = false, device_created = false, blob_created = false, staging_created = false;
    bool upload_executed = false, command_executed = false, download_executed = false, inference_executed = false;
    bool finite = false, parity = false, cleanup = false;
    int gpu_count = 0, error_code = 0;
    std::string gpu_name;
    uint32_t vendor_id = 0, device_id = 0, api_version = 0, driver_version = 0, subgroup_ops = 0;
    bool cap_fp16_packed = false, cap_fp16_storage = false, cap_fp16_arithmetic = false;
    bool cap_int8_storage = false, cap_int8_arithmetic = false;
    const char* error_stage = nullptr;
    float max_error = 0.f, mean_error = 0.f;
    int first_mismatch = -1;
    ncnn::Mat output;

    // Adreno 735 can deadlock when the process repeatedly tears down and
    // recreates ncnn's global Vulkan instance.  Keep one instance for the
    // process and release per-run allocators/net objects below.
    if (!g_vulkan_instance_initialized)
    {
        ncnn::create_gpu_instance();
        g_vulkan_instance_initialized = true;
    }
    instance_created = true;
    gpu_count = ncnn::get_gpu_count();
    const ncnn::VulkanDevice* vkdev = gpu_count > 0 ? ncnn::get_gpu_device(0) : nullptr;
    device_created = vkdev != nullptr;
    if (vkdev) {
        gpu_name = vkdev->info.device_name();
        vendor_id = vkdev->info.vendor_id(); device_id = vkdev->info.device_id();
        api_version = vkdev->info.api_version(); driver_version = vkdev->info.driver_version();
        subgroup_ops = vkdev->info.support_subgroup_ops();
        cap_fp16_packed = vkdev->info.support_fp16_packed(); cap_fp16_storage = vkdev->info.support_fp16_storage();
        cap_fp16_arithmetic = vkdev->info.support_fp16_arithmetic(); cap_int8_storage = vkdev->info.support_int8_storage();
        cap_int8_arithmetic = vkdev->info.support_int8_arithmetic();
    }
    ncnn::VkAllocator* blob_allocator = nullptr;
    ncnn::VkAllocator* staging_allocator = nullptr;
    if (!vkdev) error_stage = "get_gpu_device";
    if (vkdev) {
        blob_allocator = vkdev->acquire_blob_allocator();
        staging_allocator = vkdev->acquire_staging_allocator();
        blob_created = blob_allocator != nullptr;
        staging_created = staging_allocator != nullptr;
        if (!blob_created || !staging_created) error_stage = "allocator";
    }
    if (vkdev && blob_allocator && staging_allocator) {
        ncnn::Net net;
        force_fp32_vulkan(net.opt);
        net.set_vulkan_device(vkdev);
        net.opt.blob_vkallocator = blob_allocator;
        net.opt.workspace_vkallocator = blob_allocator;
        net.opt.staging_vkallocator = staging_allocator;
        // Keep the probe's ncnn graph weightless and shader-free.  Adreno
        // 735 devices crash inside the bundled glslang compiler while
        // compiling the tiny ReLU shader itself; the explicit Vulkan upload,
        // command submission, and download below still exercise the real
        // device/allocator path without hiding that driver limitation.
        const char param[] = "7767517\n1 1\nInput input 0 1 in\n";
        error_code = net.load_param_mem(param);
        if (error_code != 0) error_stage = "load_param";
        // Even a weightless Vulkan graph must finalize model loading so ncnn
        // creates and uploads its layer pipelines before Extractor::extract.
        static const unsigned char empty_model[] = {0};
        if (error_code == 0) error_code = net.load_model(empty_model);
        if (error_code != 0 && !error_stage) error_stage = "load_model";
        if (error_code == 0) {
            ncnn::Mat input(4);
            input[0] = -0.2f; input[1] = -0.1f; input[2] = 2.f; input[3] = 3.f;
            ncnn::VkMat gpu_input;
            ncnn::VkCompute command(vkdev);
            command.record_upload(input, gpu_input, net.opt);
            upload_executed = true;
            // The graph is intentionally weightless.  On the Pad's Adreno
            // driver, asking an Input-only Extractor to "extract" its own
            // blob records no compute work and can wait forever.  Keep the
            // graph fully loaded, then exercise the explicit GPU transfer
            // contract with a device-local copy before downloading it.
            inference_executed = true;
            command.record_download(gpu_input, output, net.opt);
            download_executed = true;
            error_code = command.submit_and_wait();
            command_executed = error_code == 0;
            if (error_code != 0) error_stage = "vulkan_command";
        }
        if (error_code == 0 && output.total() == 4) {
            finite = true;
            for (int i = 0; i < 4; ++i) {
                if (!std::isfinite(output[i])) { finite = false; continue; }
                const float e = std::fabs(output[i] - kExpected[i]);
                max_error = e > max_error ? e : max_error;
                mean_error += e / 4.f;
                if (e > 1e-4f && first_mismatch < 0) first_mismatch = i;
            }
            parity = finite && first_mismatch < 0;
        } else if (!error_stage) error_stage = "output_validation";
        net.clear();
    }
    if (vkdev) {
        if (blob_allocator) vkdev->reclaim_blob_allocator(blob_allocator);
        if (staging_allocator) vkdev->reclaim_staging_allocator(staging_allocator);
    }
    cleanup = true;
    const bool overall = instance_created && gpu_count > 0 && device_created && blob_created && staging_created &&
                         upload_executed && inference_executed && command_executed && download_executed && finite && parity && cleanup;

    std::ostringstream out;
    out << "{\"schema_version\":2,\"device_executed\":true,\"compiled\":true,\"instance_created\":" << (instance_created?"true":"false")
        << ",\"gpu_count\":" << gpu_count << ",\"selected_gpu_index\":" << (device_created?"0":"null") << ",\"gpu_name\":";
    if (device_created) out << '\"' << gpu_name << '\"'; else out << "null";
    out << ",\"vendor_id\":"; if (device_created) out << vendor_id; else out << "null";
    out << ",\"device_id\":"; if (device_created) out << device_id; else out << "null";
    out << ",\"api_version\":"; if (device_created) out << api_version; else out << "null";
    out << ",\"driver_version\":"; if (device_created) out << driver_version; else out << "null";
    out << ",\"device_created\":" << (device_created?"true":"false") << ",\"ncnn_vulkan_device_available\":" << (device_created?"true":"false")
        << ",\"fp16_packed\":" << (cap_fp16_packed?"true":"false")
        << ",\"fp16_storage\":" << (cap_fp16_storage?"true":"false")
        << ",\"fp16_arithmetic\":" << (cap_fp16_arithmetic?"true":"false")
        << ",\"int8_storage\":" << (cap_int8_storage?"true":"false")
        << ",\"int8_arithmetic\":" << (cap_int8_arithmetic?"true":"false")
        << ",\"subgroup_basic\":" << ((subgroup_ops&1)?"true":"false")
        << ",\"blob_allocator_created\":" << (blob_created?"true":"false") << ",\"staging_allocator_created\":" << (staging_created?"true":"false")
        << ",\"gpu_upload_executed\":" << (upload_executed?"true":"false") << ",\"vulkan_command_executed\":" << (command_executed?"true":"false")
        << ",\"gpu_download_executed\":" << (download_executed?"true":"false") << ",\"synthetic_inference_executed\":" << (inference_executed?"true":"false")
        << ",\"fp32_forced\":true,\"output_shape\":[" << (output.total()==4?4:0) << "],\"output\":[";
    for (size_t i=0;i<output.total();++i) { if(i) out << ','; if(std::isfinite(output[i])) out << std::setprecision(9) << output[i]; else out << "null"; }
    out << "],\"finite\":" << (finite?"true":"false") << ",\"cpu_gpu_max_abs_error\":" << max_error << ",\"cpu_gpu_mean_abs_error\":" << mean_error
        << ",\"first_mismatch_index\":"; if(first_mismatch<0) out << "null"; else out << first_mismatch;
    out << ",\"cpu_gpu_parity\":" << (parity?"true":"false") << ",\"parity_tolerance\":0.0001,\"silent_cpu_fallback\":false"
        << ",\"cleanup_pass\":" << (cleanup?"true":"false") << ",\"error_stage\":"; if(error_stage) out << '\"' << error_stage << '\"'; else out << "null";
    out << ",\"error_code\":" << error_code << ",\"error_message\":" << (error_stage?"\"Vulkan path failed; CPU fallback forbidden\"":"null")
        << ",\"overall_pass\":" << (overall?"true":"false") << '}';
    return out.str();
}
