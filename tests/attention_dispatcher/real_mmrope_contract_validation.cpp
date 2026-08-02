// Phase 10.1: real SeedVR2 MM-RoPE3D contract head_dim=128 RMS hybrid attention.
//
// Validates the exact rotary-embedding-torch==0.5.3 contract SeedVR2 uses
// (SeedVR2 commit e4de8c24441a67e1b7df56abea10645059bb1185): configured
// rope_dim=128 -> RotaryEmbedding(dim=42, freqs_for="lang", theta=10000) per
// axis, three axial 42-dim blocks -> effective_rotary_dim=126, 2-dim identity
// tail, axis-major pairs T/H/W, W-fastest token flatten.  This standalone
// validator constructs DynamicAttentionContract / CoreKey values directly and
// reuses the fully parameterized primitives (window pack/unpack, padding
// validity + additive key mask, dynamic MM-RoPE3D, ncnn Mat contracts, hybrid
// bundle loading and host head merge).  It intentionally does not touch
// run_hybrid_attention, which remains the Phase 9.5 C=32 runtime.
#include "hybrid_model_bundle.h"
#include "hybrid_window_adapter.h"
#include "dynamic_padding_mask.h"
#include "dynamic_rope3d.h"
#include "ncnn_mat_contract.h"
#include "../seedvr2_attention_fidelity/host_head_merge.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

namespace fs = std::filesystem;
using namespace attention_dispatcher;

namespace phase10_detail
{
constexpr int kChannels = 512;
constexpr int kHeads = 4;
constexpr int kHeadDim = 128;
constexpr int kRopeDim = 126;
constexpr int kTokens = 64;
constexpr int kWindowT = 4;
constexpr int kWindowH = 4;
constexpr int kWindowW = 4;
constexpr char kVariant[] = "seedvr2_hd128_mmrope3d_hybrid";

struct Case
{
    const char* name;
    int t, h, w;
    int gt, gh, gw;
    int pad_t, pad_h, pad_w;
    int num_windows;
    const char* artifact;
};

const Case kCases[] = {
    {"hd128_single", 4, 4, 4, 1, 1, 1, 0, 0, 0, 1, "hybrid_seedvr2_hd128_mmrope444_grid_1x1x1"},
    {"hd128_multi", 4, 8, 8, 1, 2, 2, 0, 0, 0, 4, "hybrid_seedvr2_hd128_mmrope444_grid_1x2x2"},
    {"hd128_padded", 5, 9, 10, 2, 3, 3, 3, 3, 2, 18, "hybrid_seedvr2_hd128_mmrope444_grid_2x3x3"},
};

struct Stats
{
    double maximum = 0.0;
    double mean = 0.0;
    size_t nan = 0;
    size_t inf = 0;
    bool file_ok = true;
};

struct StageTimings
{
    double total_ms = 0.0;
    double pack_ms = 0.0;
    double auxiliary_ms = 0.0;
    double rope_ms = 0.0;
    double heads_ms = 0.0;
    double merge_ms = 0.0;
    double projection_ms = 0.0;
    double unpack_ms = 0.0;
};

struct CaseOutput
{
    bool ok = false;
    std::string error;
    std::string selected_artifact;
    std::vector<float> input, windows, mask, rope_cos, rope_sin, merged, projection, bthwc;
    std::array<std::vector<float>, 4> heads;
    std::vector<unsigned char> validity;
    int valid_tokens = 0;
    int invalid_tokens = 0;
    int masked_key_entries = 0;
    int valid_min = 0;
    int valid_max = 0;
    size_t cache_size_before = 0;
    size_t cache_size_after = 0;
    bool all_cache_hits = false;
    StageTimings timings;
};

struct CaseReport
{
    std::string name;
    std::string artifact;
    int t = 0, h = 0, w = 0;
    int gt = 0, gh = 0, gw = 0;
    int pad_t = 0, pad_h = 0, pad_w = 0;
    int num_windows = 0;
    bool success = false;
    Stats validity, mask, rope_cos, rope_sin, windows_bin, merged, projection, windows_out, bthwc;
    std::array<Stats, 4> heads;
    int valid_tokens = 0, invalid_tokens = 0, masked_key_entries = 0, valid_min = 0, valid_max = 0;
    size_t cache_before = 0, cache_after = 0;
    bool all_cache_hits = false;
    StageTimings timings;
};

bool read_floats(const fs::path& path, std::vector<float>& out, size_t count)
{
    std::ifstream file(path, std::ios::binary);
    out.assign(count, 0.f);
    if (!file) return false;
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return file.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}

Stats compare_floats(const std::vector<float>& actual, const fs::path& ref)
{
    Stats s;
    std::vector<float> expected;
    if (!read_floats(ref, expected, actual.size()))
    {
        s.file_ok = false;
        s.maximum = std::numeric_limits<double>::infinity();
        return s;
    }
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const double d = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        s.maximum = std::max(s.maximum, d);
        s.mean += d;
        if (std::isnan(actual[i])) ++s.nan;
        if (std::isinf(actual[i])) ++s.inf;
    }
    s.mean = actual.empty() ? 0.0 : s.mean / static_cast<double>(actual.size());
    return s;
}

bool compare_exact_u8(const std::vector<unsigned char>& actual, const fs::path& ref)
{
    std::vector<unsigned char> expected(actual.size());
    std::ifstream file(ref, std::ios::binary);
    if (!file) return false;
    file.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(expected.size()));
    return file.gcount() == static_cast<std::streamsize>(expected.size()) && expected == actual;
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

DynamicAttentionContract make_contract(const Case& c)
{
    DynamicAttentionContract dc{};
    dc.batch = 1; dc.time = c.t; dc.height = c.h; dc.width = c.w;
    dc.channels = kChannels; dc.window_t = kWindowT; dc.window_h = kWindowH; dc.window_w = kWindowW;
    dc.grid_t = c.gt; dc.grid_h = c.gh; dc.grid_w = c.gw;
    dc.pad_t = c.pad_t; dc.pad_h = c.pad_h; dc.pad_w = c.pad_w;
    dc.num_windows = c.num_windows; dc.tokens_per_window = kTokens;
    dc.heads = kHeads; dc.head_dim = kHeadDim; dc.rope_dim = kRopeDim;
    return dc;
}

bool run_hd128_case(const Case& c, const fs::path& ref_root, const ArtifactRegistry& registry,
                    NcnnModelCache& cache, const DynamicMMRoPE3DConfig& rope_config, CaseOutput& out)
{
    using Clock = std::chrono::steady_clock;
    const Clock::time_point total_start = Clock::now();
    const DynamicAttentionContract dc = make_contract(c);
    std::string e;
    if (!validate_dynamic_attention_contract(dc, e)) { out.error = "validate: " + e; return false; }

    const CoreKey key{kWindowT, kWindowH, kWindowW, c.gt, c.gh, c.gw, kVariant};
    const CoreInfo* core = registry.lookup(key);
    if (!core) { out.error = "lookup: variant=" + std::string(kVariant); return false; }
    out.selected_artifact = core->name;
    const AuxiliaryInputConfig* aux = core->auxiliary();
    if (!aux || aux->mask_mode != MaskMode::PaddingValidity || aux->rope_mode != RopeMode::DynamicMMRoPE3D)
    {
        out.error = "contract: staged artifact auxiliary_inputs mismatch";
        return false;
    }

    const size_t input_count = static_cast<size_t>(dc.batch) * dc.time * dc.height * dc.width * dc.channels;
    if (!read_floats(ref_root / "input.bin", out.input, input_count)) { out.error = "read: input.bin"; return false; }

    out.cache_size_before = cache.size();
    HybridModelBundle bundle;
    HybridBundleLoadStats stats;
    Clock::time_point timer = Clock::now();
    if (!acquire_hybrid_bundle(*core, cache, bundle, stats, e)) { out.error = "bundle: " + e; return false; }
    out.cache_size_after = cache.size();
    out.all_cache_hits = stats.heads[0].cache_hit && stats.heads[1].cache_hit && stats.heads[2].cache_hit &&
                         stats.heads[3].cache_hit && stats.projection.cache_hit;

    const HybridWindowContract hc{dc.batch, dc.time, dc.height, dc.width, dc.channels, dc.window_t, dc.window_h,
                                  dc.window_w, dc.grid_t, dc.grid_h, dc.grid_w, dc.pad_t, dc.pad_h, dc.pad_w};
    timer = Clock::now();
    if (!pack_hybrid_windows(out.input, hc, out.windows, e)) { out.error = "pack: " + e; return false; }
    out.timings.pack_ms = elapsed_ms(timer);

    timer = Clock::now();
    if (!generate_padding_validity(dc, out.validity, e)) { out.error = "validity: " + e; return false; }
    if (!generate_additive_key_mask(out.validity, dc.num_windows, dc.tokens_per_window, out.mask, e)) { out.error = "mask: " + e; return false; }
    out.timings.auxiliary_ms = elapsed_ms(timer);

    timer = Clock::now();
    Rope3DOutput rope;
    const Rope3DConfig rc{rope_config.theta, rope_config.axis_dims, rope_config.axis_scales, rope_config.freq_mode};
    if (!generate_dynamic_rope3d(dc, rc, rope, e)) { out.error = "rope: " + e; return false; }
    out.rope_cos = rope.cos;
    out.rope_sin = rope.sin;
    out.timings.rope_ms = elapsed_ms(timer);

    ncnn::Mat mask_mat, cos_mat, sin_mat;
    if (!make_mask_mat(out.mask, dc.num_windows, dc.tokens_per_window, mask_mat, e) ||
        !make_rope_mat(out.rope_cos, dc.tokens_per_window, rope.pairs, cos_mat, e) ||
        !make_rope_mat(out.rope_sin, dc.tokens_per_window, rope.pairs, sin_mat, e))
    {
        out.error = "mat: " + e;
        return false;
    }

    timer = Clock::now();
    for (int h = 0; h < kHeads; ++h)
    {
        out.heads[h].assign(static_cast<size_t>(dc.num_windows) * kTokens * kHeadDim, 0.f);
        for (int w = 0; w < dc.num_windows; ++w)
        {
            ncnn::Mat x(kChannels, kTokens);
            std::memcpy(x.data, out.windows.data() + static_cast<size_t>(w) * kTokens * kChannels,
                        sizeof(float) * kTokens * kChannels);
            ncnn::Mat m = mask_mat.channel(w);
            ncnn::Mat co = cos_mat;
            ncnn::Mat si = sin_mat;
            ncnn::Mat head_out;
            ncnn::Extractor ex = bundle.heads[h]->create_extractor();
            if (ex.input("in0", x) || ex.input("in1", m) || ex.input("in2", co) || ex.input("in3", si) ||
                ex.extract("out0", head_out) || head_out.total() != static_cast<size_t>(kTokens) * kHeadDim)
            {
                out.error = "head inference head=" + std::to_string(h) + " window=" + std::to_string(w);
                return false;
            }
            std::memcpy(out.heads[h].data() + static_cast<size_t>(w) * kTokens * kHeadDim, head_out.data,
                        sizeof(float) * kTokens * kHeadDim);
        }
    }
    out.timings.heads_ms = elapsed_ms(timer);

    timer = Clock::now();
    std::vector<std::vector<float>> heads_vec(4);
    for (int h = 0; h < 4; ++h) heads_vec[h] = out.heads[h];
    if (!seedvr2::merge_attention_heads(heads_vec, dc.num_windows, kTokens, kHeadDim, out.merged, e)) { out.error = "merge: " + e; return false; }
    out.timings.merge_ms = elapsed_ms(timer);

    timer = Clock::now();
    ncnn::Mat pi(kChannels, dc.num_windows * kTokens), po;
    std::memcpy(pi.data, out.merged.data(), out.merged.size() * sizeof(float));
    ncnn::Extractor pex = bundle.output_projection->create_extractor();
    if (pex.input("in0", pi) || pex.extract("out0", po) || po.total() != out.merged.size())
    {
        out.error = "projection inference";
        return false;
    }
    out.projection.assign(static_cast<const float*>(po.data), static_cast<const float*>(po.data) + po.total());
    out.timings.projection_ms = elapsed_ms(timer);

    timer = Clock::now();
    if (!unpack_hybrid_windows(out.projection, hc, out.bthwc, e)) { out.error = "unpack: " + e; return false; }
    out.timings.unpack_ms = elapsed_ms(timer);
    out.timings.total_ms = elapsed_ms(total_start);

    out.valid_tokens = 0;
    out.valid_min = kTokens;
    out.valid_max = 0;
    for (int w = 0; w < dc.num_windows; ++w)
    {
        const int count = count_valid_tokens(out.validity, w, dc.tokens_per_window);
        out.valid_tokens += count;
        out.valid_min = std::min(out.valid_min, count);
        out.valid_max = std::max(out.valid_max, count);
    }
    out.invalid_tokens = dc.num_windows * kTokens - out.valid_tokens;
    out.masked_key_entries = 0;
    for (const float value : out.mask) out.masked_key_entries += value < 0.f ? 1 : 0;
    out.ok = true;
    return true;
}

void stats_json(std::ostream& o, const char* name, const Stats& s, size_t count, const char* logical_shape)
{
    o << "\"" << name << "\":{\"logical_shape\":\"" << logical_shape << "\",\"element_count\":" << count
      << ",\"max_abs_error\":" << std::setprecision(10) << s.maximum
      << ",\"mean_abs_error\":" << s.mean
      << ",\"nan_count\":" << s.nan << ",\"inf_count\":" << s.inf << "}";
}

bool has_context(const std::string& error)
{
    const char* contexts[] = {"validate", "lookup", "bundle", "pack", "validity", "mask", "rope", "mat", "axis",
                              "head", "merge", "projection", "unpack", "read", "input", "cache", "freq"};
    for (const char* context : contexts)
        if (error.find(context) != std::string::npos) return true;
    return false;
}

struct NegativeResult
{
    int total = 0;
    int passed = 0;
    std::vector<std::string> failures;
};

bool run_negative_tests(const ArtifactRegistry& registry, NcnnModelCache& cache, const fs::path& ref_root,
                        const DynamicMMRoPE3DConfig& rope_config, NegativeResult& result)
{
    const size_t cache_before = cache.size();
    const auto check = [&](bool condition, const std::string& error, const char* label) {
        ++result.total;
        if (condition && has_context(error)) ++result.passed;
        else result.failures.push_back(std::string(label) + ": " + (condition ? "missing stage context" : error));
    };

    const DynamicAttentionContract base = make_contract(kCases[2]);
    std::string e;

    DynamicAttentionContract bad = base; bad.channels = 256;
    check(!validate_dynamic_attention_contract(bad, e), e, "channels != 512");
    bad = base; bad.heads = 8;
    check(!validate_dynamic_attention_contract(bad, e), e, "heads != 4");
    bad = base; bad.head_dim = 64;
    check(!validate_dynamic_attention_contract(bad, e), e, "head_dim != 128");
    bad = base; bad.rope_dim = 256;
    check(!validate_dynamic_attention_contract(bad, e), e, "rope_dim > head_dim");
    bad = base; bad.rope_dim = 64;
    {
        Rope3DOutput out;
        const Rope3DConfig rc{rope_config.theta, rope_config.axis_dims, rope_config.axis_scales, rope_config.freq_mode};
        check(!generate_dynamic_rope3d(bad, rc, out, e) && out.cos.empty(), e, "rope_dim != metadata");
    }
    {
        Rope3DConfig rc{rope_config.theta, {{41, 43, 42}}, rope_config.axis_scales, rope_config.freq_mode};
        Rope3DOutput out;
        check(!generate_dynamic_rope3d(base, rc, out, e), e, "odd axis_dims");
    }
    {
        Rope3DConfig rc{rope_config.theta, {{42, 42, 40}}, rope_config.axis_scales, rope_config.freq_mode};
        Rope3DOutput out;
        check(!generate_dynamic_rope3d(base, rc, out, e), e, "axis_dims sum mismatch");
    }
    {
        Rope3DConfig rc{rope_config.theta, rope_config.axis_dims, rope_config.axis_scales, "bogus_mode"};
        Rope3DOutput out;
        check(!generate_dynamic_rope3d(base, rc, out, e), e, "invalid freq_mode");
    }
    {
        DynamicAttentionContract bad = base; bad.rope_dim = 130;
        Rope3DOutput out;
        check(!generate_dynamic_rope3d(bad, {rope_config.theta, rope_config.axis_dims, rope_config.axis_scales, rope_config.freq_mode}, out, e), e, "effective rope_dim > head_dim");
    }
    {
        Rope3DConfig rc{rope_config.theta, {{44, 42, 42}}, {1.0, 0.01, 0.0}};
        Rope3DOutput out;
        check(!generate_dynamic_rope3d(base, rc, out, e), e, "invalid axis_scales");
    }
    bad = base; bad.grid_t = 3;
    check(!validate_dynamic_attention_contract(bad, e), e, "grid vs padded shape mismatch");
    bad = base; bad.pad_t = 0;
    check(!validate_dynamic_attention_contract(bad, e), e, "padding metadata wrong");

    const CoreKey unknown_key{kWindowT, kWindowH, kWindowW, 2, 2, 2, kVariant};
    check(registry.lookup(unknown_key) == nullptr, "lookup returned core", "unknown HD128 grid");
    const CoreKey phase95_key{kWindowT, kWindowH, kWindowW, 1, 1, 1, "dynamic_rms_rope3d_hybrid"};
    check(registry.lookup(phase95_key) == nullptr, "lookup returned core", "phase 9.5 variant for HD128 shape");

    const CoreKey c32_key{kWindowT, kWindowH, kWindowW, 1, 2, 2, kVariant};
    check(registry.lookup(c32_key) != nullptr, "lookup failed", "HD128 variant core presence");
    {
        std::vector<float> c32_input(static_cast<size_t>(1) * 4 * 8 * 8 * 32, 1.f);
        std::vector<float> packed;
        const HybridWindowContract hc128{1, 4, 8, 8, kChannels, 4, 4, 4, 1, 2, 2, 0, 0, 0};
        check(!pack_hybrid_windows(c32_input, hc128, packed, e), e, "HD128 variant requested with C=32");
    }

    const HybridWindowContract hc{1, 5, 9, 10, kChannels, 4, 4, 4, 2, 3, 3, 3, 3, 2};
    std::vector<float> too_few(static_cast<size_t>(1) * 5 * 9 * 10 * kChannels - 1, 1.f);
    std::vector<float> too_many(static_cast<size_t>(1) * 5 * 9 * 10 * kChannels + 1, 1.f);
    std::vector<float> packed;
    check(!pack_hybrid_windows(too_few, hc, packed, e), e, "input elements -1");
    check(!pack_hybrid_windows(too_many, hc, packed, e), e, "input elements +1");
    {
        std::vector<float> wrong_windows(100, 0.f);
        check(!unpack_hybrid_windows(wrong_windows, hc, packed, e), e, "reverse/crop out of bounds");
    }
    {
        ncnn::Mat mat;
        check(!make_mask_mat(std::vector<float>(18 * 64 * 64 - 1), 18, 64, mat, e), e, "mask Mat shape error");
        check(!make_rope_mat(std::vector<float>(64 * 64 - 1), 64, 64, mat, e), e, "rope Mat shape error");
    }

    check(cache.size() == cache_before, "cache polluted by negatives", "cache non-pollution");
    return result.failures.empty();
}

struct LatencyReport
{
    std::string name;
    double total_mean_ms = 0.0, total_min_ms = 0.0, total_max_ms = 0.0;
    StageTimings mean;
};

bool measure_latency(const Case& c, const fs::path& ref_root, const ArtifactRegistry& registry,
                     NcnnModelCache& cache, const DynamicMMRoPE3DConfig& rope_config, LatencyReport& report)
{
    std::vector<CaseOutput> runs(13);
    for (int i = 0; i < 13; ++i)
    {
        if (!run_hd128_case(c, ref_root, registry, cache, rope_config, runs[i])) return false;
    }
    report.name = c.name;
    std::vector<double> totals;
    StageTimings sum{};
    for (int i = 3; i < 13; ++i)
    {
        totals.push_back(runs[i].timings.total_ms);
        sum.total_ms += runs[i].timings.total_ms;
        sum.pack_ms += runs[i].timings.pack_ms;
        sum.auxiliary_ms += runs[i].timings.auxiliary_ms;
        sum.rope_ms += runs[i].timings.rope_ms;
        sum.heads_ms += runs[i].timings.heads_ms;
        sum.merge_ms += runs[i].timings.merge_ms;
        sum.projection_ms += runs[i].timings.projection_ms;
        sum.unpack_ms += runs[i].timings.unpack_ms;
    }
    const double n = 10.0;
    report.mean.total_ms = sum.total_ms / n;
    report.total_mean_ms = report.mean.total_ms;
    report.mean.pack_ms = sum.pack_ms / n;
    report.mean.auxiliary_ms = sum.auxiliary_ms / n;
    report.mean.rope_ms = sum.rope_ms / n;
    report.mean.heads_ms = sum.heads_ms / n;
    report.mean.merge_ms = sum.merge_ms / n;
    report.mean.projection_ms = sum.projection_ms / n;
    report.mean.unpack_ms = sum.unpack_ms / n;
    const auto minmax = std::minmax_element(totals.begin(), totals.end());
    report.total_min_ms = *minmax.first;
    report.total_max_ms = *minmax.second;
    return true;
}

void write_regressions_from_input(std::ostream& json, const fs::path& input_path, bool& all_regressions_pass)
{
    std::ifstream file(input_path);
    json << "\"regressions\":{";
    if (!file)
    {
        json << "\"external_regressions\":\"not_run\"}";
        return;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string content = buffer.str();
    size_t pos = 0;
    bool first = true;
    all_regressions_pass = true;
    while (pos < content.size())
    {
        const size_t end = content.find('\n', pos);
        std::string line = content.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        const size_t eq = line.find('=');
        if (eq != std::string::npos)
        {
            if (!first) json << ",";
            first = false;
            const std::string key = line.substr(0, eq);
            const std::string value = line.substr(eq + 1);
            json << "\"" << key << "\":\"" << value << "\"";
            if (value != "PASS") all_regressions_pass = false;
        }
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    json << "}";
}

} // namespace phase10_detail

using namespace phase10_detail;

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3)
    {
        std::cerr << "usage: real_mmrope_contract_validation <fidelity_root> [regression_inputs.txt]\n";
        return 2;
    }
    const fs::path fidelity_root = argv[1];
    const fs::path regression_input = argc == 3 ? fs::path(argv[2]) : fs::path();
    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(fidelity_root, error)) { std::cerr << "registry scan failed: " << error << '\n'; return 3; }

    const CoreKey probe{4, 4, 4, 1, 1, 1, kVariant};
    const CoreInfo* probe_core = registry.lookup(probe);
    if (!probe_core) { std::cerr << "HD128-S core not found\n"; return 4; }
    const AuxiliaryInputConfig* aux = probe_core->auxiliary();
    if (!aux || aux->rope_mode != RopeMode::DynamicMMRoPE3D)
    {
        std::cerr << "HD128-S auxiliary rope contract missing\n";
        return 4;
    }
    const DynamicMMRoPE3DConfig rope_config = std::get<DynamicMMRoPE3DConfig>(aux->rope);
    if (rope_config.rope_dim != kRopeDim || rope_config.axis_dims[0] + rope_config.axis_dims[1] + rope_config.axis_dims[2] != kRopeDim ||
        rope_config.freq_mode != "lang_rope")
    {
        std::cerr << "HD128-S real rope contract mismatch\n";
        return 4;
    }

    NcnnModelCache cache;
    std::vector<CaseReport> reports;
    std::vector<float> first_s_bthwc, first_s_merged, first_s_projection, first_s_windows;
    bool second_s_cache_hits = false;
    std::array<std::vector<float>, 4> first_s_heads;
    std::vector<unsigned char> first_s_validity;
    bool determinism_ok = true;
    const int run_order[4] = {0, 1, 2, 0};
    for (int run = 0; run < 4; ++run)
    {
        const Case& c = kCases[run_order[run]];
        const fs::path ref = fidelity_root / c.artifact / "reference";
        CaseOutput out;
        const bool ran = run_hd128_case(c, ref, registry, cache, rope_config, out);
        if (run < 3)
        {
            CaseReport report;
            report.name = c.name;
            report.artifact = c.artifact;
            report.t = c.t; report.h = c.h; report.w = c.w;
            report.gt = c.gt; report.gh = c.gh; report.gw = c.gw;
            report.pad_t = c.pad_t; report.pad_h = c.pad_h; report.pad_w = c.pad_w;
            report.num_windows = c.num_windows;
            report.success = ran && out.ok;
            if (ran && out.ok)
            {
                report.validity = compare_floats(std::vector<float>(out.validity.begin(), out.validity.end()), ref / "validity.bin");
                report.mask = compare_floats(out.mask, ref / "mask.bin");
                report.rope_cos = compare_floats(out.rope_cos, ref / "rope_cos.bin");
                report.rope_sin = compare_floats(out.rope_sin, ref / "rope_sin.bin");
                report.windows_bin = compare_floats(out.windows, ref / "windows.bin");
                for (int h = 0; h < 4; ++h) report.heads[h] = compare_floats(out.heads[h], ref / ("head" + std::to_string(h) + ".bin"));
                report.merged = compare_floats(out.merged, ref / "merged.bin");
                report.projection = compare_floats(out.projection, ref / "projection.bin");
                report.windows_out = compare_floats(out.projection, ref / "windows_output.bin");
                report.bthwc = compare_floats(out.bthwc, ref / "bthwc_output.bin");
                report.valid_tokens = out.valid_tokens;
                report.invalid_tokens = out.invalid_tokens;
                report.masked_key_entries = out.masked_key_entries;
                report.valid_min = out.valid_min;
                report.valid_max = out.valid_max;
                report.cache_before = out.cache_size_before;
                report.cache_after = out.cache_size_after;
                report.all_cache_hits = out.all_cache_hits;
                report.timings = out.timings;
            }
            reports.push_back(std::move(report));
        }
        if (run == 0)
        {
            first_s_bthwc = out.bthwc;
            first_s_merged = out.merged;
            first_s_projection = out.projection;
            first_s_windows = out.windows;
            first_s_heads = out.heads;
            first_s_validity = out.validity;
        }
        if (run == 3)
        {
            second_s_cache_hits = out.all_cache_hits;
            determinism_ok = out.bthwc == first_s_bthwc && out.merged == first_s_merged &&
                             out.projection == first_s_projection && out.windows == first_s_windows &&
                             out.heads == first_s_heads && out.validity == first_s_validity;
        }
        if (!ran || !out.ok) std::cerr << "case " << c.name << " failed: " << out.error << '\n';
    }

    bool all_cases_ok = true;
    for (const CaseReport& report : reports)
    {
        if (!report.success) all_cases_ok = false;
        if (report.validity.file_ok && report.validity.maximum != 0.0) all_cases_ok = false;
        if (report.mask.file_ok && report.mask.maximum != 0.0) all_cases_ok = false;
        if (report.rope_cos.maximum >= 1e-6 || report.rope_sin.maximum >= 1e-6) all_cases_ok = false;
        if (report.windows_bin.maximum != 0.0) all_cases_ok = false;
        if (report.merged.maximum >= 1e-4 || report.projection.maximum >= 1e-4 || report.windows_out.maximum >= 1e-4 ||
            report.bthwc.maximum >= 1e-4)
            all_cases_ok = false;
        for (const Stats& head : report.heads)
            if (head.maximum >= 1e-4 || head.nan || head.inf) all_cases_ok = false;
        if (report.bthwc.nan || report.bthwc.inf || report.merged.nan || report.merged.inf) all_cases_ok = false;
    }
    if (reports.size() < 3 || !reports[2].success || reports[2].valid_tokens != 450 || reports[2].masked_key_entries != 44928 ||
        reports[2].invalid_tokens != 702 || reports[2].valid_min != 2 || reports[2].valid_max != 64)
        all_cases_ok = false;

    std::vector<LatencyReport> latency;
    bool latency_ok = true;
    NcnnModelCache timing_cache;
    for (const Case& c : kCases)
    {
        LatencyReport report;
        if (!measure_latency(c, fidelity_root / c.artifact / "reference", registry, timing_cache, rope_config, report))
            latency_ok = false;
        latency.push_back(report);
    }

    NegativeResult negative;
    bool negative_ok = run_negative_tests(registry, cache, fidelity_root / kCases[0].artifact / "reference", rope_config, negative);
    if (!negative_ok) std::cerr << "negative tests failed\n";

    size_t ncnn_bytes = 0;
    const fs::path source_dir = fidelity_root / kCases[0].artifact;
    for (int h = 0; h < 4; ++h)
    {
        ncnn_bytes += fs::file_size(source_dir / "attention_heads" / ("head" + std::to_string(h)) / "model.ncnn.param");
        ncnn_bytes += fs::file_size(source_dir / "attention_heads" / ("head" + std::to_string(h)) / "model.ncnn.bin");
    }
    ncnn_bytes += fs::file_size(source_dir / "output_projection" / "model.ncnn.param");
    ncnn_bytes += fs::file_size(source_dir / "output_projection" / "model.ncnn.bin");

    PROCESS_MEMORY_COUNTERS memory{};
    memory.cb = sizeof(memory);
    const bool memory_ok = GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)) != 0;

    size_t buffer_bytes = 0;
    for (const CaseReport& report : reports)
    {
        const size_t windows = static_cast<size_t>(report.num_windows);
        buffer_bytes += (windows * kTokens * kChannels) * sizeof(float) * 2;
        buffer_bytes += (windows * kTokens * kTokens) * sizeof(float);
        buffer_bytes += (kTokens * kTokens) * sizeof(float) * 2;
        buffer_bytes += (windows * kTokens * kHeadDim) * sizeof(float) * 4;
        buffer_bytes += (windows * kTokens * kChannels) * sizeof(float);
        buffer_bytes += (static_cast<size_t>(report.t) * report.h * report.w * kChannels) * sizeof(float);
    }

    const fs::path report_path = fs::path(__FILE__).parent_path().parent_path().parent_path() / "reports" / "real_seedvr2_mmrope_contract_validation.json";
    std::ofstream json(report_path);
    json << std::setprecision(10);
    json << "{\n\"phase\":\"10.1\",\n\"scope\":\"real SeedVR2 MM-RoPE3D contract fixture (rotary-embedding-torch==0.5.3, SeedVR2 e4de8c2)\",\n"
         << "\"real_seedvr2_weights\":false,\n"
         << "\"configuration\":{\"heads\":" << kHeads << ",\"head_dim\":" << kHeadDim << ",\"channels\":" << kChannels
         << ",\"configured_rope_dim\":128,\"rope_dim\":" << kRopeDim << ",\"identity_tail_dim\":2,\"window\":[4,4,4],\"tokens_per_window\":" << kTokens
         << ",\"dtype\":\"fp32\",\"pnnx_fp16\":false},\n"
         << "\"rope_contract\":{\"mode\":\"dynamic_mmrope3d\",\"freq_mode\":\"lang_rope\",\"rope_dim\":" << rope_config.rope_dim
         << ",\"theta\":" << rope_config.theta
         << ",\"axis_dims\":[" << rope_config.axis_dims[0] << "," << rope_config.axis_dims[1] << ","
         << rope_config.axis_dims[2] << "],\"axis_scales\":[" << rope_config.axis_scales[0] << ","
         << rope_config.axis_scales[1] << "," << rope_config.axis_scales[2]
         << "],\"source\":\"exact upstream contract: RotaryEmbedding(dim=128//3=42, freqs_for=lang, theta=10000); per-axis local pair inv_freq=theta^(-k/21); T,H,W axis-major concat; identity tail 2 dims\"},\n"
         << "\"cases\":[\n";
    const char* sep = "";
    for (const CaseReport& report : reports)
    {
        json << sep;
        sep = ",\n";
        json << "{\n\"name\":\"" << report.name << "\",\n"
             << "\"request\":{\"input_shape\":[1," << report.t << "," << report.h << "," << report.w << "," << kChannels
             << "],\"variant\":\"" << kVariant << "\",\"window\":[4,4,4],\"grid\":[" << report.gt << "," << report.gh
             << "," << report.gw << "],\"padding\":[" << report.pad_t << "," << report.pad_h << "," << report.pad_w
             << "],\"num_windows\":" << report.num_windows << "},\n"
             << "\"selected_core\":{\"key\":\"window=[4,4,4],grid=[" << report.gt << "," << report.gh << "," << report.gw
             << "],variant=" << kVariant << "\",\"artifact\":\"" << report.artifact << "\"},\n"
             << "\"validity\":{\"valid_tokens_total\":" << report.valid_tokens << ",\"invalid_tokens\":" << report.invalid_tokens
             << ",\"valid_tokens_min\":" << report.valid_min << ",\"valid_tokens_max\":" << report.valid_max
             << ",\"total_window_tokens\":" << (report.num_windows * kTokens) << "},\n"
             << "\"mask\":{\"masked_key_entries\":" << report.masked_key_entries << "},\n"
             << "\"rope\":{\"logical_shape\":\"[64,64]\",\"ncnn_mat\":\"Mat(w=64,h=64)\","
             << "\"cos_max_abs_error\":" << report.rope_cos.maximum << ",\"sin_max_abs_error\":" << report.rope_sin.maximum << "},\n"
             << "\"accuracy\":{";
        stats_json(json, "validity", report.validity, static_cast<size_t>(report.num_windows) * kTokens, "uint8 per window token");
        json << ",";
        stats_json(json, "mask", report.mask, static_cast<size_t>(report.num_windows) * kTokens * kTokens, "[W,64,64]");
        json << ",";
        stats_json(json, "rope_cos", report.rope_cos, static_cast<size_t>(kTokens) * kTokens, "[64,64]");
        json << ",";
        stats_json(json, "rope_sin", report.rope_sin, static_cast<size_t>(kTokens) * kTokens, "[64,64]");
        for (int h = 0; h < 4; ++h)
        {
            json << ",";
            stats_json(json, ("head" + std::to_string(h)).c_str(), report.heads[h], static_cast<size_t>(report.num_windows) * kTokens * kHeadDim, "[W,64,128]");
        }
        json << ",";
        stats_json(json, "merged", report.merged, static_cast<size_t>(report.num_windows) * kTokens * kChannels, "[W,64,512]");
        json << ",";
        stats_json(json, "projection", report.projection, static_cast<size_t>(report.num_windows) * kTokens * kChannels, "[W,64,512]");
        json << ",";
        stats_json(json, "windows", report.windows_out, static_cast<size_t>(report.num_windows) * kTokens * kChannels, "[W,64,512]");
        json << ",";
        stats_json(json, "bthwc", report.bthwc, static_cast<size_t>(report.t) * report.h * report.w * kChannels, "[1,T,H,W,512]");
        json << "},\n"
             << "\"cache\":{\"size_before\":" << report.cache_before << ",\"size_after\":" << report.cache_after
             << ",\"all_cache_hits\":" << (report.all_cache_hits ? "true" : "false") << "},\n"
             << "\"latency_ms\":{\"total\":" << report.timings.total_ms << "},\n"
             << "\"status\":\"" << (report.success ? "PASS" : "FAIL") << "\"\n}";
    }
    json << "\n],\n\"determinism\":{\"s_m_p_s_bit_identical\":" << (determinism_ok ? "true" : "false")
         << ",\"second_s_run_all_cache_hits\":" << (second_s_cache_hits ? "true" : "false") << "},\n"
         << "\"negative_tests\":{\"total\":" << negative.total << ",\"passed\":" << negative.passed << ",\"failures\":[";
    for (size_t i = 0; i < negative.failures.size(); ++i)
        json << (i ? ",\"" : "\"") << negative.failures[i] << "\"";
    json << "]},\n\"latency\":{\"warmups\":3,\"cached_iterations\":10,\"profiles\":[";
    sep = "";
    for (const LatencyReport& report : latency)
    {
        json << sep << "{\"name\":\"" << report.name << "\",\"total_ms_mean\":" << report.total_mean_ms
             << ",\"total_ms_min\":" << report.total_min_ms << ",\"total_ms_max\":" << report.total_max_ms
             << ",\"window_pack_ms\":" << report.mean.pack_ms << ",\"validity_mask_ms\":" << report.mean.auxiliary_ms
             << ",\"rope_ms\":" << report.mean.rope_ms << ",\"heads_total_ms\":" << report.mean.heads_ms
             << ",\"host_merge_ms\":" << report.mean.merge_ms << ",\"projection_ms\":" << report.mean.projection_ms
             << ",\"window_unpack_ms\":" << report.mean.unpack_ms << "}";
        sep = ",";
    }
    json << "]},\n\"memory\":{\"ncnn_model_files_bytes\":" << ncnn_bytes << ",\"runtime_main_buffers_bytes\":" << buffer_bytes;
    if (memory_ok)
        json << ",\"process_working_set_bytes\":" << memory.WorkingSetSize << ",\"process_peak_working_set_bytes\":" << memory.PeakWorkingSetSize;
    else
        json << ",\"process_working_set_bytes\":\"not measured\",\"process_peak_working_set_bytes\":\"not measured\"";
    json << "},\n\"cache_size_after\":" << cache.size() << ",\n";
    bool regressions_pass = false;
    write_regressions_from_input(json, regression_input, regressions_pass);
    const bool overall = all_cases_ok && determinism_ok && negative.failures.empty() && latency_ok &&
                         (regression_input.empty() || regressions_pass);
    json << ",\n\"overall_status\":\"" << (overall ? "PASS" : "FAIL") << "\"\n}\n";
    json.close();

    std::cout << "real_mmrope_contract_validation=" << (overall ? "PASS" : "FAIL") << " cache=" << cache.size()
              << " cases_ok=" << (all_cases_ok ? "PASS" : "FAIL") << " determinism=" << (determinism_ok ? "PASS" : "FAIL")
              << " negatives=" << negative.passed << "/" << negative.total << '\n';
    return overall ? 0 : 6;
}
