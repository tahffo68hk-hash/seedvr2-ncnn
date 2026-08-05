#include "runtime_info.h"
#include <iostream>

int main() {
    const std::string json = synthetic_cpu_test_json(false);
    std::cout << json << '\n';
    const char* required[] = {"\"schema_version\":2", "\"runs\":3", "\"threads\":1", "\"fp32_forced\":true",
                              "\"finite\":true", "\"expected_parity\":true", "\"numeric_repeat_consistent\":true",
                              "\"device_executed\":false", "\"overall_pass\":true"};
    for (const char* token : required) if (json.find(token) == std::string::npos) return 1;
    if (json.find("NaN") != std::string::npos || json.find("Infinity") != std::string::npos) return 2;
    return 0;
}
