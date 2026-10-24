#include "hybrid_attention_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace attention_dispatcher;

struct Stats
{
    float maximum = 0;
    double mean = 0;
    double cosine = 1;
    size_t nan = 0;
    size_t inf = 0;
};

struct Case
{
    const char* name;
    int t, h, w, gt, gh, gw;
    const char* artifact;
};

static bool read(const fs::path& path, std::vector<float>& values, size_t count)
{
    std::ifstream file(path, std::ios::binary);
    values.resize(count);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return file.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}

static Stats compare(const std::vector<float>& actual, const fs::path& reference_path)
{
    std::vector<float> reference;
    Stats stats;
    if (!read(reference_path, reference, actual.size()))
    {
        stats.maximum = INFINITY;
        return stats;
    }

    double dot = 0, actual_norm_squared = 0, reference_norm_squared = 0;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const float difference = std::fabs(actual[i] - reference[i]);
        stats.maximum = std::max(stats.maximum, difference);
        stats.mean += difference;
        dot += static_cast<double>(actual[i]) * reference[i];
        actual_norm_squared += static_cast<double>(actual[i]) * actual[i];
        reference_norm_squared += static_cast<double>(reference[i]) * reference[i];
        stats.nan += std::isnan(actual[i]);
        stats.inf += std::isinf(actual[i]);
    }
    stats.mean /= actual.size();
    const double norm_product = std::sqrt(actual_norm_squared * reference_norm_squared);
    stats.cosine = norm_product > 0 ? dot / norm_product
                                    : (actual_norm_squared == 0 && reference_norm_squared == 0 ? 1 : 0);
    return stats;
}

static void stats_json(std::ostream& output, const Stats& stats, size_t count)
{
    output << "{\"element_count\":" << count << ",\"max_abs_error\":" << stats.maximum
           << ",\"mean_abs_error\":" << stats.mean << ",\"cosine_similarity\":" << stats.cosine
           << ",\"nan_count\":" << stats.nan << ",\"inf_count\":" << stats.inf << "}";
}

static long long read_process_vmhwm_kb()
{
#if defined(__ANDROID__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
    {
        if (line.rfind("VmHWM:", 0) == 0)
        {
            std::istringstream value(line.substr(6));
            long long kilobytes = -1;
            value >> kilobytes;
            return kilobytes;
        }
    }
#endif
    return -1;
}

static bool select_cases(const std::string& profile, std::vector<int>& selected)
{
    if (profile == "all")
    {
        selected = {0, 1, 2};
        return true;
    }
    if (profile == "A") selected = {0};
    else if (profile == "B") selected = {1};
    else if (profile == "C") selected = {2};
    else return false;
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 5)
    {
        std::cerr << "usage: dynamic_multi_shape_validation <baseline_root> <fidelity_root> [report_path] [A|B|C|all]\n";
        return 2;
    }

    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(std::vector<fs::path>{argv[1], argv[2]}, error))
    {
        std::cerr << error << '\n';
        return 3;
    }

    const Case cases[] = {
        {"dynamic_a", 4, 8, 8, 1, 2, 2, "hybrid_dynamic_rms_rope444_grid_1x2x2"},
        {"dynamic_b", 4, 9, 10, 1, 3, 3, "hybrid_dynamic_rms_rope444_grid_1x3x3"},
        {"dynamic_c", 5, 9, 10, 2, 3, 3, "hybrid_dynamic_rms_rope444_grid_2x3x3"},
    };
    const std::string profile = argc == 5 ? argv[4] : "all";
    std::vector<int> selected;
    if (!select_cases(profile, selected))
    {
        std::cerr << "invalid profile: " << profile << " (expected A, B, C, or all)\n";
        return 2;
    }

    const fs::path default_report = fs::path(__FILE__).parent_path().parent_path().parent_path()
                                    / "reports/dynamic_rms_multi_shape_runtime_validation.json";
    const fs::path report = argc >= 4 ? fs::path(argv[3]) : default_report;
    std::ofstream json(report);
    if (!json)
    {
        std::cerr << "cannot open report path: " << report.string() << '\n';
        return 7;
    }

    NcnnModelCache cache;
    bool ok = true;
    bool first_case = true;
    json << std::setprecision(10)
         << "{\n\"phase\":\"9.5\",\n\"variant\":\"dynamic_rms_rope3d_hybrid\",\n\"cases\":[";

    for (const int case_index : selected)
    {
        const Case& test_case = cases[case_index];
        const fs::path artifact_reference = fs::path(argv[2]) / test_case.artifact / "reference";
        std::vector<float> input;
        const size_t input_count = static_cast<size_t>(test_case.t) * test_case.h * test_case.w * 32;
        if (!read(artifact_reference / "input.bin", input, input_count))
        {
            std::cerr << "cannot read input reference for " << test_case.name << '\n';
            return 4;
        }

        const AttentionRequest request{1, test_case.t, test_case.h, test_case.w, 32, "fp32",
                                       "dynamic_rms_rope3d_hybrid"};
        std::vector<double> samples;
        HybridRuntimeResult measured;
        for (int iteration = 0; iteration < 6; ++iteration)
        {
            auto result = run_hybrid_attention(request, input, registry, cache);
            if (!result.success)
            {
                std::cerr << test_case.name << " iteration=" << iteration << " error=" << result.error_message << '\n';
                return 5;
            }
            if (result.selected_artifact != test_case.artifact || result.metadata.grid_t != test_case.gt
                || result.metadata.grid_h != test_case.gh || result.metadata.grid_w != test_case.gw)
            {
                std::cerr << test_case.name << " selected artifact/grid mismatch\n";
                return 6;
            }
            if (iteration >= 3) samples.push_back(result.timings.total_ms);
            if (iteration == 5) measured = std::move(result);
        }

        std::vector<float> validity(measured.validity.begin(), measured.validity.end());
        const Stats validity_stats = compare(validity, artifact_reference / "validity.bin");
        const Stats mask_stats = compare(measured.mask_input, artifact_reference / "mask.bin");
        const Stats rope_cos_stats = compare(measured.rope_cos, artifact_reference / "rope_cos.bin");
        const Stats rope_sin_stats = compare(measured.rope_sin, artifact_reference / "rope_sin.bin");
        const Stats bthwc_stats = compare(measured.bthwc_output, artifact_reference / "bthwc_output.bin");
        const Stats merged_stats = compare(measured.merged_output, artifact_reference / "merged.bin");
        const Stats projection_stats = compare(measured.projection_output, artifact_reference / "projection_2d.bin");
        const Stats windows_stats = compare(measured.windows_output, artifact_reference / "windows_output.bin");
        std::array<Stats, 4> head_stats;
        for (int head = 0; head < 4; ++head)
        {
            head_stats[head] = compare(measured.head_outputs[head],
                                       artifact_reference / ("head" + std::to_string(head) + ".bin"));
        }

        size_t valid_tokens = 0, masked_entries = 0;
        for (const unsigned char value : measured.validity) valid_tokens += value != 0;
        for (const float value : measured.mask_input) masked_entries += value < 0;
        ok &= validity_stats.maximum == 0 && mask_stats.maximum == 0 && rope_cos_stats.maximum < 1e-6f
              && rope_sin_stats.maximum < 1e-6f && bthwc_stats.maximum < 1e-4f && !bthwc_stats.nan
              && !bthwc_stats.inf && merged_stats.maximum < 1e-4f && projection_stats.maximum < 1e-4f
              && windows_stats.maximum < 1e-4f;
        for (const Stats& head : head_stats) ok &= head.maximum < 1e-4f && !head.nan && !head.inf;

        if (!first_case) json << ',';
        first_case = false;
        json << "\n{\"name\":\"" << test_case.name << "\",\"request\":{\"input_shape\":[1,"
             << test_case.t << ',' << test_case.h << ',' << test_case.w << ",32]},\"window_metadata\":{\"window\":[4,4,4],\"grid\":["
             << test_case.gt << ',' << test_case.gh << ',' << test_case.gw << "]},\"validity\":{\"valid_tokens_total\":"
             << valid_tokens << "},\"mask\":{\"masked_key_entries\":" << masked_entries << "},\"accuracy\":{\"validity\":";
        stats_json(json, validity_stats, validity.size());
        json << ",\"mask\":";
        stats_json(json, mask_stats, measured.mask_input.size());
        json << ",\"rope_cos\":";
        stats_json(json, rope_cos_stats, measured.rope_cos.size());
        json << ",\"rope_sin\":";
        stats_json(json, rope_sin_stats, measured.rope_sin.size());
        for (int head = 0; head < 4; ++head)
        {
            json << ",\"head" << head << "\":";
            stats_json(json, head_stats[head], measured.head_outputs[head].size());
        }
        json << ",\"merged\":";
        stats_json(json, merged_stats, measured.merged_output.size());
        json << ",\"projection\":";
        stats_json(json, projection_stats, measured.projection_output.size());
        json << ",\"windows\":";
        stats_json(json, windows_stats, measured.windows_output.size());
        json << ",\"bthwc\":";
        stats_json(json, bthwc_stats, measured.bthwc_output.size());
        json << "},\"latency_ms\":{\"warmups\":3,\"iterations\":3,\"mean\":"
             << std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()
             << ",\"min\":" << *std::min_element(samples.begin(), samples.end())
             << ",\"max\":" << *std::max_element(samples.begin(), samples.end()) << "},\"status\":\""
             << (ok ? "PASS" : "FAIL") << "\"}";

        std::cout << test_case.name << " input=[1," << test_case.t << ',' << test_case.h << ',' << test_case.w
                  << ",32] bthwc_cosine=" << bthwc_stats.cosine << " bthwc_max_abs_error=" << bthwc_stats.maximum
                  << " warmups=3 iterations=3 mean_ms="
                  << std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()
                  << " min_ms=" << *std::min_element(samples.begin(), samples.end())
                  << " max_ms=" << *std::max_element(samples.begin(), samples.end()) << '\n';

        if (case_index == 0)
        {
            auto repeated = run_hybrid_attention(request, input, registry, cache);
            ok &= repeated.success && measured.bthwc_output == repeated.bthwc_output;
        }
    }

    const std::vector<float> bad_input(1);
    const auto rejected = run_hybrid_attention({1, 4, 9, 10, 32, "fp32", "dynamic_rms_rope3d_hybrid"},
                                               bad_input, registry, cache);
    ok &= !rejected.success;

    const long long vmhwm_kb = read_process_vmhwm_kb();
    json << "\n],\n\"regressions\":{\"invalid_input_rejected\":" << (!rejected.success ? "true" : "false")
         << "},\n\"cache_size_after\":" << cache.size() << ",\n\"process_vmhwm_kb\":";
    if (vmhwm_kb >= 0) json << vmhwm_kb;
    else json << "null";
    json << ",\n\"overall_status\":\"" << (ok ? "PASS" : "FAIL") << "\"\n}\n";

    std::cout << "process_vmhwm_kb=";
    if (vmhwm_kb >= 0) std::cout << vmhwm_kb;
    else std::cout << "unmeasured";
    std::cout << " overall_status=" << (ok ? "PASS" : "FAIL") << " report=" << report.string() << '\n';
    return ok ? 0 : 8;
}
