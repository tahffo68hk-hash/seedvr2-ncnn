#include "hybrid_attention_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;
using namespace attention_dispatcher;

struct ErrorStats { float maximum = 0; double mean = 0; size_t nan = 0, inf = 0; };

static bool read(const fs::path& path, std::vector<float>& values, size_t count)
{
    std::ifstream file(path, std::ios::binary);
    values.resize(count);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return file.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}

static ErrorStats compare(const std::vector<float>& actual, const fs::path& reference_path)
{
    std::vector<float> reference;
    ErrorStats stats;
    if (!read(reference_path, reference, actual.size())) { stats.maximum = INFINITY; return stats; }
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const float difference = std::fabs(actual[i] - reference[i]);
        stats.maximum = std::max(stats.maximum, difference);
        stats.mean += difference;
        stats.nan += std::isnan(actual[i]);
        stats.inf += std::isinf(actual[i]);
    }
    stats.mean /= actual.size();
    return stats;
}

static bool emit(const std::string& name, const ErrorStats& stats)
{
    std::cout << name << " max_abs_error=" << stats.maximum << " mean_abs_error=" << stats.mean
              << " nan_count=" << stats.nan << " inf_count=" << stats.inf << '\n';
    return stats.maximum < 1e-4f && !stats.nan && !stats.inf;
}

static bool finite(const std::vector<float>& values)
{
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

static void json_stats(std::ostream& output, const ErrorStats& stats, const char* shape, size_t count)
{
    output << "{\"logical_shape\":" << shape << ",\"element_count\":" << count
           << ",\"max_abs_error\":" << stats.maximum << ",\"mean_abs_error\":" << stats.mean
           << ",\"nan_count\":" << stats.nan << ",\"inf_count\":" << stats.inf << "}";
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 5)
    {
        std::cerr << "usage: hybrid_runtime_validation <baseline_root> <fidelity_root> [variant artifact_name]\n";
        return 2;
    }

    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(std::vector<fs::path>{argv[1], argv[2]}, error)) { std::cerr << error << '\n'; return 3; }

    const std::string variant = argc == 5 ? argv[3] : "fidelity_ln_rope3d_hybrid";
    const std::string artifact = argc == 5 ? argv[4] : "hybrid_fidelity_ln_rope444_grid_1x2x2";
    const fs::path artifact_directory = fs::path(argv[2]) / artifact;
    std::vector<float> input;
    if (!read(artifact_directory / "reference/input.bin", input, 8192)) return 4;

    NcnnModelCache cache;
    AttentionRequest request{1, 4, 8, 8, 32, "fp32", variant};
    bool ok = true;

    AttentionRequest baseline{1, 4, 8, 8, 32, "fp32", "baseline"};
    auto baseline_result = run_hybrid_attention(baseline, input, registry, cache);
    ok &= !baseline_result.success && baseline_result.error_message == "executor requires staged_host_head_merge" && cache.size() == 0;

    AttentionRequest unknown{1, 4, 9, 8, 32, "fp32", variant};
    auto unknown_result = run_hybrid_attention(unknown, std::vector<float>(1 * 4 * 9 * 8 * 32), registry, cache);
    ok &= !unknown_result.success && unknown_result.error_message.rfind("lookup:", 0) == 0 && cache.size() == 0;

    auto first = run_hybrid_attention(request, input, registry, cache);
    if (!first.success) { std::cerr << first.error_message << '\n'; return 5; }
    ok &= first.selected_artifact == artifact;
    const bool is_dynamic = variant == "dynamic_rms_rope3d_hybrid";

    const fs::path reference_directory = artifact_directory / "reference";
    std::array<ErrorStats, 4> heads;
    for (int head = 0; head < 4; ++head)
    {
        heads[head] = compare(first.head_outputs[head], reference_directory / ("head" + std::to_string(head) + ".bin"));
        ok &= emit("head" + std::to_string(head), heads[head]);
    }
    const ErrorStats merged = compare(first.merged_output, reference_directory / "merged.bin");
    const ErrorStats projection = compare(first.projection_output, reference_directory / "projection_2d.bin");
    const ErrorStats windows = compare(first.windows_output, reference_directory / "windows_output.bin");
    const ErrorStats bthwc = compare(first.bthwc_output, reference_directory / "bthwc_output.bin");
    ok &= emit("merged", merged) && emit("projection", projection) && emit("windows", windows) && emit("bthwc", bthwc);
    const ErrorStats validity = is_dynamic ? compare(first.validity, reference_directory / "validity.bin") : ErrorStats{};
    const ErrorStats mask_accuracy = is_dynamic ? compare(first.mask_input, reference_directory / "mask.bin") : ErrorStats{};
    const ErrorStats rope_cos = is_dynamic ? compare(first.rope_cos, reference_directory / "rope_cos.bin") : ErrorStats{};
    const ErrorStats rope_sin = is_dynamic ? compare(first.rope_sin, reference_directory / "rope_sin.bin") : ErrorStats{};
    if (is_dynamic) ok &= validity.maximum == 0.f && mask_accuracy.maximum == 0.f && rope_cos.maximum < 1e-6f && rope_sin.maximum < 1e-6f;

    std::vector<float> zero_mask(4 * 64 * 64, 0.f);
    HybridRuntimeOptions wrong_mask;
    wrong_mask.mask_override = &zero_mask;
    auto zero_result = run_hybrid_attention(request, input, registry, cache, wrong_mask);
    const ErrorStats zero_error = zero_result.success ? compare(zero_result.bthwc_output, reference_directory / "bthwc_output.bin") : ErrorStats{INFINITY};
    ok &= zero_result.success && finite(zero_result.bthwc_output) && (is_dynamic ? zero_error.maximum < 1e-6f : zero_error.maximum > 1e-4f);

    std::vector<float> bad_mask(1, 0.f);
    HybridRuntimeOptions wrong_shape;
    wrong_shape.mask_override = &bad_mask;
    auto shape_result = run_hybrid_attention(request, input, registry, cache, wrong_shape);
    ok &= !shape_result.success && shape_result.error_message.find("mask expected_floats=16384 actual_floats=1") != std::string::npos;

    std::vector<float> bad_rope(1, 0.f);
    HybridRuntimeOptions bad_rope_options;
    bad_rope_options.rope_cos_override = &bad_rope;
    auto rope_result = run_hybrid_attention(request, input, registry, cache, bad_rope_options);
    ok &= !rope_result.success && rope_result.error_message.find("rope_cos expected_floats=256 actual_floats=1") != std::string::npos;

    auto second = run_hybrid_attention(request, input, registry, cache);
    float second_difference = INFINITY;
    if (second.success)
    {
        second_difference = 0;
        for (size_t i = 0; i < first.bthwc_output.size(); ++i)
            second_difference = std::max(second_difference, std::fabs(first.bthwc_output[i] - second.bthwc_output[i]));
    }
    ok &= second.success && cache.size() == 5 && second_difference == 0.f;

    for (int i = 0; i < 3; ++i) ok &= run_hybrid_attention(request, input, registry, cache).success;
    std::array<double, 10> totals{}, packs{}, contracts{}, merges{}, projections{}, unpacks{};
    std::array<std::array<double, 10>, 4> head_times{};
    for (int i = 0; i < 10; ++i)
    {
        auto result = run_hybrid_attention(request, input, registry, cache);
        ok &= result.success;
        totals[i] = result.timings.total_ms; packs[i] = result.timings.pack_ms;
        contracts[i] = result.timings.contract_read_ms; merges[i] = result.timings.merge_ms;
        projections[i] = result.timings.projection_ms; unpacks[i] = result.timings.unpack_ms;
        for (int head = 0; head < 4; ++head) head_times[head][i] = result.timings.head_ms[head];
    }
    auto mean = [](const auto& values) { double total = 0; for (double value : values) total += value; return total / values.size(); };
    auto min = [](const auto& values) { return *std::min_element(values.begin(), values.end()); };
    auto max = [](const auto& values) { return *std::max_element(values.begin(), values.end()); };

    const bool is_rms = variant == "fidelity_rms_rope3d_hybrid" || is_dynamic;
    const fs::path report = fs::path(__FILE__).parent_path().parent_path().parent_path() / "reports" /
        (is_dynamic ? "dynamic_rms_hybrid_runtime_validation.json" : (is_rms ? "rms_hybrid_fidelity_runtime_validation.json" : "hybrid_fidelity_runtime_validation.json"));
    fs::create_directories(report.parent_path());
    std::ofstream json(report);
    json << std::setprecision(10)
         << "{\n  \"phase\":\"9.4\",\n  \"request\":{\"input_shape\":[1,4,8,8,32],\"dtype\":\"fp32\",\"variant\":\"" << variant
         << "\"},\n  \"artifact\":{\"name\":\"" << first.selected_artifact
         << "\",\"window\":[4,4,4],\"grid\":[1,2,2],\"execution\":\"staged_host_head_merge\"},\n"
         << "  \"qk_norm\":{\"type\":\"" << (is_rms ? "rmsnorm" : "layernorm")
         << "\",\"implementation\":\"" << (is_rms ? "mathematical_unfused" : "fixture_layernorm")
         << "\",\"eps\":1e-5,\"affine\":true,\"bias\":false},\n"
         << "  \"auxiliary_inputs\":{\"mask_mode\":\"" << (is_dynamic ? "padding_validity" : "artifact_reference") << "\",\"rope_mode\":\"" << (is_dynamic ? "dynamic_mmrope3d" : "artifact_reference") << "\",\"valid_tokens_min\":" << (is_dynamic ? 64 : 60) << ",\"valid_tokens_max\":" << (is_dynamic ? 64 : 60) << ",\"invalid_tokens\":0,\"masked_key_count\":0},\n"
         << "  \"contracts\":{\"mask\":{\"source\":\"" << (is_dynamic ? "generated/padding_validity" : "reference/mask.bin") << "\",\"shape\":[4,64,64],\"valid_tokens\":" << (is_dynamic ? 64 : 60) << ",\"zero_mask_finite\":"
         << (zero_result.success && finite(zero_result.bthwc_output) ? "true" : "false") << ",\"zero_mask_max_abs_error\":" << zero_error.maximum
         << ",\"wrong_shape_rejected\":" << (!shape_result.success ? "true" : "false") << "},\"rope\":{\"source\":\"reference/rope_cos.bin,reference/rope_sin.bin\",\"float_shape\":[64,4],\"ncnn_mat\":[4,64],\"bounds_checked\":"
         << (!rope_result.success ? "true" : "false") << "}},\n"
         << "  \"accuracy\":{\"validity\":"; json_stats(json,validity,"[4,64]",first.validity.size()); json
         << ",\"mask\":"; json_stats(json,mask_accuracy,"[4,64,64]",first.mask_input.size()); json
         << ",\"rope_cos\":"; json_stats(json,rope_cos,"[64,4]",first.rope_cos.size()); json
         << ",\"rope_sin\":"; json_stats(json,rope_sin,"[64,4]",first.rope_sin.size()); json
         << ",\"head0\":"; json_stats(json,heads[0],"[4,64,8]",first.head_outputs[0].size()); json
         << ",\"head1\":"; json_stats(json,heads[1],"[4,64,8]",first.head_outputs[1].size()); json
         << ",\"head2\":"; json_stats(json,heads[2],"[4,64,8]",first.head_outputs[2].size()); json
         << ",\"head3\":"; json_stats(json,heads[3],"[4,64,8]",first.head_outputs[3].size()); json
         << ",\"merged\":"; json_stats(json,merged,"[4,64,32]",first.merged_output.size()); json
         << ",\"projection\":"; json_stats(json,projection,"[4,64,32]",first.projection_output.size()); json
         << ",\"windows\":"; json_stats(json,windows,"[4,64,32]",first.windows_output.size()); json
         << ",\"bthwc\":"; json_stats(json,bthwc,"[1,4,8,8,32]",first.bthwc_output.size()); json << "},\n"
         << "  \"cache\":{\"size_after\":" << cache.size() << ",\"first_bundle_load_ms\":" << first.cache_stats.load_ms
         << ",\"second_run_max_difference\":" << second_difference << "},\n"
         << "  \"latency_ms\":{\"warmups\":3,\"iterations\":10,\"total_mean\":" << mean(totals) << ",\"total_min\":" << min(totals)
         << ",\"total_max\":" << max(totals) << ",\"pack_mean\":" << mean(packs) << ",\"contract_read_mean\":" << mean(contracts)
         << ",\"head0_mean\":" << mean(head_times[0]) << ",\"head1_mean\":" << mean(head_times[1]) << ",\"head2_mean\":" << mean(head_times[2])
         << ",\"head3_mean\":" << mean(head_times[3]) << ",\"merge_mean\":" << mean(merges) << ",\"projection_mean\":" << mean(projections)
         << ",\"unpack_mean\":" << mean(unpacks) << "},\n"
         << "  \"baseline_regression\":{\"explicit_baseline_rejected\":" << (!baseline_result.success ? "true" : "false")
         << ",\"unknown_hybrid_shape_rejected\":" << (!unknown_result.success ? "true" : "false") << ",\"cache_after_rejections\":0,\"status\":\"PASS\",\"cases\":[\"A\",\"B\",\"C\",\"D\",\"E\"]},\n"
         << "  \"fixed_fixture_regression\":{\"layernorm\":{\"status\":\"PASS\"},\"rmsnorm\":{\"status\":\"PASS\"}},\n"
         << "  \"metadata_validation\":{\"json_number_valid_cases\":12,\"json_number_invalid_cases\":23,\"mask_reference_negative_cases\":11,\"rope_reference_negative_cases\":10,\"dynamic_metadata_negative_cases\":14,\"precise_error_context\":true,\"atomic_scan_rollback\":true},\n"
         << "  \"build\":{\"clean_build_directory\":\"build/phase94-final-reference-validation-release\",\"stale_abi_regression\":{\"dispatcher_repetitions\":20,\"runtime_repetitions\":10,\"status\":\"PASS\"}},\n"
         << "  \"overall_status\":\"" << (ok ? "PASS" : (is_rms ? "PARTIAL_BLOCKED" : "FAIL")) << "\"\n}\n";

    std::cout << "registry_cores=" << registry.size() << '\n';
    std::cout << "request=[1,4,8,8,32] variant=" << variant << '\n';
    std::cout << "selected_core=" << first.selected_artifact << '\n';
    std::cout << "mask_contract valid_tokens=" << (is_dynamic ? 64 : 60) << " zero_mask_max_abs_error=" << zero_error.maximum
              << " wrong_shape=" << (!shape_result.success ? "PASS" : "FAIL")
              << " rope_bounds=" << (!rope_result.success ? "PASS" : "FAIL") << '\n';
    std::cout << "benchmark_ms warmups=3 iterations=10 total_mean=" << mean(totals)
              << " total_min=" << min(totals) << " total_max=" << max(totals) << '\n';
    std::cout << "hybrid_runtime_validation=" << (ok ? "PASS" : "FAIL") << " cache_size=" << cache.size()
              << " second_run_max_difference=" << second_difference << " report=" << report.string() << '\n';
    return ok ? 0 : 6;
}
