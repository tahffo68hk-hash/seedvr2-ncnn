#pragma once
#include <string>
std::string runtime_version_json();
std::string runtime_build_json();
std::string ncnn_info_json();
std::string synthetic_cpu_test_json(bool device_execution);
std::string vulkan_probe_json();
