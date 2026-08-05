#include "runtime_info.h"

#include <net.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace {
constexpr float kExpected[] = {-0.2f, -0.1f, 2.f, 3.f};
constexpr float kExpectedTolerance = 1e-4f;
constexpr float kRepeatTolerance = 1e-6f;
constexpr int kRunCount = 3;

struct RunResult {
    int return_code = -1;
    std::vector<float> output;
    bool shape_pass = false;
    bool finite = false;
    int first_non_finite_index = -1;
    std::string first_non_finite_value;
    float max_abs_error = 0.f;
    int first_mismatch_index = -1;
    double duration_ms = 0.;
};

void force_fp32_single_thread(ncnn::Option& opt) {
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
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

std::string non_finite_name(float value) {
    if (std::isnan(value)) return "NaN";
    return std::signbit(value) ? "-Infinity" : "+Infinity";
}

RunResult run_once() {
    RunResult result;
    const auto start = std::chrono::steady_clock::now();
    ncnn::Net net;
    force_fp32_single_thread(net.opt);
    const char param[] = "7767517\n2 2\nInput input 0 1 in\nReLU relu 1 1 in out 0=0.1\n";
    result.return_code = net.load_param_mem(param);
    if (result.return_code == 0) {
        ncnn::Mat input(4);
        const float values[] = {-2.f, -1.f, 2.f, 3.f};
        for (int i = 0; i < 4; ++i) input[i] = values[i];
        ncnn::Extractor extractor = net.create_extractor();
        result.return_code = extractor.input("in", input);
        ncnn::Mat output;
        if (result.return_code == 0) result.return_code = extractor.extract("out", output);
        result.shape_pass = result.return_code == 0 && output.dims == 1 && output.w == 4 && output.total() == 4;
        if (result.return_code == 0) {
            result.output.resize(output.total());
            for (size_t i = 0; i < output.total(); ++i) result.output[i] = output[i];
        }
    }
    result.finite = result.shape_pass;
    if (result.shape_pass) {
        for (int i = 0; i < 4; ++i) {
            const float actual = result.output[i];
            const float expected = kExpected[i];
            if (!std::isfinite(expected) || !std::isfinite(actual)) {
                result.finite = false;
                if (result.first_non_finite_index < 0) {
                    result.first_non_finite_index = i;
                    result.first_non_finite_value = non_finite_name(actual);
                }
                continue;
            }
            const float error = std::fabs(actual - expected);
            if (error > result.max_abs_error) result.max_abs_error = error;
            if (error > kExpectedTolerance && result.first_mismatch_index < 0) result.first_mismatch_index = i;
        }
    }
    result.duration_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

void float_value(std::ostringstream& out, float value) {
    if (std::isfinite(value)) out << std::setprecision(9) << value;
    else out << '\"' << non_finite_name(value) << '\"';
}
}

std::string synthetic_cpu_test_json(bool device_execution) {
    std::vector<RunResult> runs;
    for (int i = 0; i < kRunCount; ++i) runs.push_back(run_once());
    bool shape_pass = true;
    bool finite = true;
    bool expected_parity = true;
    int non_finite_count = 0;
    int first_non_finite_run = -1;
    int first_non_finite_index = -1;
    std::string first_non_finite_value;
    float max_abs_error = 0.f;
    int first_mismatch = -1;
    for (int r = 0; r < kRunCount; ++r) {
        shape_pass &= runs[r].shape_pass;
        finite &= runs[r].finite;
        expected_parity &= runs[r].shape_pass && runs[r].finite && runs[r].first_mismatch_index < 0;
        if (runs[r].max_abs_error > max_abs_error) max_abs_error = runs[r].max_abs_error;
        if (first_mismatch < 0 && runs[r].first_mismatch_index >= 0) first_mismatch = runs[r].first_mismatch_index;
        if (!runs[r].finite) {
            ++non_finite_count;
            if (first_non_finite_run < 0) {
                first_non_finite_run = r + 1;
                first_non_finite_index = runs[r].first_non_finite_index;
                first_non_finite_value = runs[r].first_non_finite_value;
            }
        }
    }
    bool bitwise_equal = shape_pass;
    bool numeric_repeat = shape_pass && finite;
    float repeat_max_abs_error = 0.f;
    if (shape_pass) {
        for (int r = 1; r < kRunCount; ++r) {
            bitwise_equal &= std::memcmp(runs[0].output.data(), runs[r].output.data(), 4 * sizeof(float)) == 0;
            for (int i = 0; i < 4; ++i) {
                const float error = std::fabs(runs[0].output[i] - runs[r].output[i]);
                if (error > repeat_max_abs_error) repeat_max_abs_error = error;
                if (!std::isfinite(error) || error > kRepeatTolerance) numeric_repeat = false;
            }
        }
    }
    bool return_codes_ok = true;
    for (const RunResult& run : runs) return_codes_ok &= run.return_code == 0;
    const bool overall = return_codes_ok && shape_pass && finite && expected_parity && numeric_repeat;

    std::ostringstream out;
    out << "{\"schema_version\":2,\"backend\":\"cpu\",\"device_executed\":" << (device_execution ? "true" : "false")
        << ",\"threads\":1,\"runs\":3,\"fp32_forced\":true,\"use_vulkan_compute\":false,\"use_packing_layout\":false"
        << ",\"use_fp16_packed\":false,\"use_fp16_storage\":false,\"use_fp16_arithmetic\":false"
        << ",\"use_int8_packed\":false,\"use_int8_storage\":false,\"use_int8_arithmetic\":false"
        << ",\"shape\":[4],\"expected\":[-0.2,-0.1,2,3],\"outputs\":[";
    for (int r = 0; r < kRunCount; ++r) {
        if (r) out << ',';
        out << '[';
        for (size_t i = 0; i < runs[r].output.size(); ++i) { if (i) out << ','; float_value(out, runs[r].output[i]); }
        out << ']';
    }
    out << "],\"run_details\":[";
    for (int r = 0; r < kRunCount; ++r) {
        if (r) out << ',';
        out << "{\"run\":" << r + 1 << ",\"output_shape\":[" << (runs[r].shape_pass ? 4 : 0) << "],\"element_count\":" << runs[r].output.size()
            << ",\"finite\":" << (runs[r].finite ? "true" : "false") << ",\"first_non_finite_index\":";
        if (runs[r].first_non_finite_index < 0) out << "null"; else out << runs[r].first_non_finite_index;
        out << ",\"first_non_finite_value\":";
        if (runs[r].first_non_finite_index < 0) out << "null"; else out << '\"' << runs[r].first_non_finite_value << '\"';
        out << ",\"max_abs_error_vs_expected\":" << runs[r].max_abs_error << ",\"first_mismatch_index\":";
        if (runs[r].first_mismatch_index < 0) out << "null"; else out << runs[r].first_mismatch_index;
        out << ",\"duration_ms\":" << runs[r].duration_ms << ",\"ncnn_return_code\":" << runs[r].return_code << '}';
    }
    out << "],\"ncnn_return_codes\":[" << runs[0].return_code << ',' << runs[1].return_code << ',' << runs[2].return_code << ']'
        << ",\"shape_pass\":" << (shape_pass ? "true" : "false") << ",\"finite\":" << (finite ? "true" : "false")
        << ",\"non_finite_count\":" << non_finite_count << ",\"first_non_finite_run\":";
    if (first_non_finite_run < 0) out << "null"; else out << first_non_finite_run;
    out << ",\"first_non_finite_index\":"; if (first_non_finite_index < 0) out << "null"; else out << first_non_finite_index;
    out << ",\"first_non_finite_value\":"; if (first_non_finite_run < 0) out << "null"; else out << '\"' << first_non_finite_value << '\"';
    out << ",\"max_abs_error\":" << max_abs_error << ",\"first_expected_mismatch_index\":"; if (first_mismatch < 0) out << "null"; else out << first_mismatch;
    out << ",\"expected_tolerance\":" << kExpectedTolerance << ",\"expected_parity\":" << (expected_parity ? "true" : "false")
        << ",\"bitwise_equal\":" << (bitwise_equal ? "true" : "false") << ",\"numeric_repeat_consistent\":" << (numeric_repeat ? "true" : "false")
        << ",\"repeat_max_abs_error\":" << repeat_max_abs_error << ",\"repeat_tolerance\":" << kRepeatTolerance
        << ",\"silent_fallback\":false,\"overall_pass\":" << (overall ? "true" : "false") << '}';
    return out.str();
}
