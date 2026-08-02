// Phase 13: real-width (C=2560/H=20/HD=128) SeedVR2 DiT block staged ncnn
// end-to-end validation for the minimal RW-A profile.
//
// Runs RW-A -> RW-A inside one process (first complete forward + second
// forward), checks per-stage parity against the generated PyTorch reference
// with the Phase 13 thresholds, verifies bit-identical determinism, cache
// stability, first-load vs cached latency and process working set, and writes
// a stdlib-parseable JSON report (no NaN/Infinity) per process run.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include "artifact_registry.h"
#include "hybrid_dit_block_runtime.h"
#include "phase12_test_common.h"

using namespace attention_dispatcher;
namespace fs = std::filesystem;
using seedvr2_dit_block::DiTBlockInput;
using seedvr2_dit_block::DiTBlockOutput;
using seedvr2_dit_block::DiTBlockRunStats;
using seedvr2_dit_block::DiTBlockTrace;
using seedvr2_dit_block::HybridDiTBlockRuntime;
using seedvr2_dit_block::VideoShape;
using phase12_test::ErrorStats;
using phase12_test::JVal;

namespace
{
const char* kVariant = "realwidth_seedvr2_dit_block_c2560_h20_hd128_random";

struct ProfileRef
{
    fs::path dir;
    std::vector<VideoShape> vid_shapes;
    std::vector<int> txt_lengths;
    std::vector<float> vid, txt, emb;
    std::vector<int> ref_window_count, ref_window_sizes, ref_partition, ref_reverse, ref_segments, ref_cu;
    int ref_total_packed = 0;
};

JVal number(double v)
{
    JVal j;
    j.kind = JVal::Number;
    j.num = v;
    return j;
}
JVal boolean(bool v)
{
    JVal j;
    j.kind = JVal::Bool;
    j.b = v;
    return j;
}
JVal text(const std::string& v)
{
    JVal j;
    j.kind = JVal::String;
    j.str = v;
    return j;
}

double working_set_mb(bool peak)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    {
        const std::size_t bytes = peak ? pmc.PeakWorkingSetSize : pmc.WorkingSetSize;
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }
#endif
    return -1.0;
}

bool load_profile(const fs::path& dir, ProfileRef& out, std::string& error)
{
    out.dir = dir;
    std::string text_content;
    if (!phase12_test::read_text(dir / "manifest.json", text_content))
    {
        error = "manifest missing " + (dir / "manifest.json").string();
        return false;
    }
    const JVal manifest = phase12_test::parse_json(text_content);
    const JVal& inputs = manifest.obj.at("inputs");
    const JVal& windows = manifest.obj.at("windows");
    const auto shapes = phase12_test::nested_int_array(inputs.obj.at("vid_shape"));
    const auto tshapes = phase12_test::nested_int_array(inputs.obj.at("txt_shape"));
    for (const auto& s : shapes) out.vid_shapes.push_back({s[0], s[1], s[2]});
    for (const auto& s : tshapes) out.txt_lengths.push_back(s[0]);
    if (!phase12_test::read_bin(dir / "input_vid.bin", out.vid) ||
        !phase12_test::read_bin(dir / "input_txt.bin", out.txt) ||
        !phase12_test::read_bin(dir / "emb.bin", out.emb))
    {
        error = "input bin missing in " + dir.string();
        return false;
    }
    out.ref_window_count = phase12_test::int_array(windows.obj.at("window_count"));
    out.ref_window_sizes = phase12_test::int_array(windows.obj.at("window_sizes"));
    out.ref_partition = phase12_test::int_array(windows.obj.at("partition_index"));
    out.ref_reverse = phase12_test::int_array(windows.obj.at("reverse_index"));
    out.ref_segments = phase12_test::int_array(windows.obj.at("segment_lengths"));
    out.ref_cu = phase12_test::int_array(windows.obj.at("cu_seqlens_q"));
    out.ref_total_packed = static_cast<int>(windows.obj.at("total_packed").num);
    error.clear();
    return true;
}

DiTBlockInput make_input(const ProfileRef& p, int dim)
{
    DiTBlockInput in;
    in.vid = ncnn::Mat(dim, static_cast<int>(p.vid.size() / dim));
    std::memcpy(in.vid.data, p.vid.data(), p.vid.size() * sizeof(float));
    in.txt = ncnn::Mat(dim, static_cast<int>(p.txt.size() / dim));
    std::memcpy(in.txt.data, p.txt.data(), p.txt.size() * sizeof(float));
    in.emb = ncnn::Mat(6 * dim, static_cast<int>(p.emb.size() / (6 * dim)));
    std::memcpy(in.emb.data, p.emb.data(), p.emb.size() * sizeof(float));
    in.vid_shapes = p.vid_shapes;
    in.txt_lengths = p.txt_lengths;
    return in;
}

struct StageSpec
{
    const char* name;
    std::vector<float> DiTBlockTrace::*field;
    float threshold;
};

// Phase 13 thresholds (task section 14): RMSNorm/Ada-in <1e-5, Linear/QKV and
// rotated Q/K <2e-5, RoPE tables <1e-6, packed attention/projection <1e-4,
// SwiGLU product <2e-4, residuals and finals <2e-4.  Default expectation stays
// near the Phase 12 1e-5-level results; final >2e-4 would not be READY.
const StageSpec kStages[] = {
    {"attn_norm_vid", &DiTBlockTrace::attn_norm_vid, 1e-5f},
    {"attn_norm_txt", &DiTBlockTrace::attn_norm_txt, 1e-5f},
    {"ada_attn_in_vid", &DiTBlockTrace::ada_attn_in_vid, 1e-5f},
    {"ada_attn_in_txt", &DiTBlockTrace::ada_attn_in_txt, 1e-5f},
    {"qkv_vid", &DiTBlockTrace::qkv_vid, 2e-5f},
    {"qkv_txt", &DiTBlockTrace::qkv_txt, 2e-5f},
    {"q_vid", &DiTBlockTrace::q_vid, 2e-5f},
    {"k_vid", &DiTBlockTrace::k_vid, 2e-5f},
    {"v_vid", &DiTBlockTrace::v_vid, 2e-5f},
    {"q_txt", &DiTBlockTrace::q_txt, 2e-5f},
    {"k_txt", &DiTBlockTrace::k_txt, 2e-5f},
    {"v_txt", &DiTBlockTrace::v_txt, 2e-5f},
    {"q_norm_vid", &DiTBlockTrace::q_norm_vid, 1e-5f},
    {"k_norm_vid", &DiTBlockTrace::k_norm_vid, 1e-5f},
    {"q_norm_txt", &DiTBlockTrace::q_norm_txt, 1e-5f},
    {"k_norm_txt", &DiTBlockTrace::k_norm_txt, 1e-5f},
    {"rope_before_q_vid", &DiTBlockTrace::rope_before_q_vid, 2e-5f},
    {"rope_before_k_vid", &DiTBlockTrace::rope_before_k_vid, 2e-5f},
    {"rope_after_q_vid", &DiTBlockTrace::rope_after_q_vid, 2e-5f},
    {"rope_after_k_vid", &DiTBlockTrace::rope_after_k_vid, 2e-5f},
    {"rope_after_q_txt", &DiTBlockTrace::rope_after_q_txt, 2e-5f},
    {"rope_after_k_txt", &DiTBlockTrace::rope_after_k_txt, 2e-5f},
    {"packed_q", &DiTBlockTrace::packed_q, 1e-4f},
    {"packed_k", &DiTBlockTrace::packed_k, 1e-4f},
    {"packed_v", &DiTBlockTrace::packed_v, 1e-4f},
    {"attention_packed_output", &DiTBlockTrace::attention_packed_output, 1e-4f},
    {"attention_vid_windows", &DiTBlockTrace::attention_vid_windows, 1e-4f},
    {"attention_txt_windows", &DiTBlockTrace::attention_txt_windows, 1e-4f},
    {"attention_vid_merged", &DiTBlockTrace::attention_vid_merged, 1e-4f},
    {"attention_txt_mean_merged", &DiTBlockTrace::attention_txt_mean_merged, 1e-4f},
    {"attention_proj_vid", &DiTBlockTrace::attention_proj_vid, 1e-4f},
    {"attention_proj_txt", &DiTBlockTrace::attention_proj_txt, 1e-4f},
    {"ada_attn_gate_vid", &DiTBlockTrace::ada_attn_gate_vid, 1e-4f},
    {"ada_attn_gate_txt", &DiTBlockTrace::ada_attn_gate_txt, 1e-4f},
    {"attention_residual_vid", &DiTBlockTrace::attention_residual_vid, 2e-4f},
    {"attention_residual_txt", &DiTBlockTrace::attention_residual_txt, 2e-4f},
    {"mlp_norm_vid", &DiTBlockTrace::mlp_norm_vid, 1e-5f},
    {"mlp_norm_txt", &DiTBlockTrace::mlp_norm_txt, 1e-5f},
    {"ada_mlp_in_vid", &DiTBlockTrace::ada_mlp_in_vid, 1e-5f},
    {"ada_mlp_in_txt", &DiTBlockTrace::ada_mlp_in_txt, 1e-5f},
    {"swiglu_gate_projection_vid", &DiTBlockTrace::swiglu_gate_vid, 1e-4f},
    {"swiglu_up_projection_vid", &DiTBlockTrace::swiglu_up_vid, 1e-4f},
    {"swiglu_product_vid", &DiTBlockTrace::swiglu_product_vid, 2e-4f},
    {"swiglu_output_vid", &DiTBlockTrace::swiglu_output_vid, 1e-4f},
    {"swiglu_gate_projection_txt", &DiTBlockTrace::swiglu_gate_txt, 1e-4f},
    {"swiglu_up_projection_txt", &DiTBlockTrace::swiglu_up_txt, 1e-4f},
    {"swiglu_product_txt", &DiTBlockTrace::swiglu_product_txt, 2e-4f},
    {"swiglu_output_txt", &DiTBlockTrace::swiglu_output_txt, 1e-4f},
    {"ada_mlp_gate_vid", &DiTBlockTrace::ada_mlp_gate_vid, 1e-4f},
    {"ada_mlp_gate_txt", &DiTBlockTrace::ada_mlp_gate_txt, 1e-4f},
    {"final_vid", &DiTBlockTrace::final_vid, 2e-4f},
    {"final_txt", &DiTBlockTrace::final_txt, 2e-4f},
};

bool compare_stage(const std::string& name, const std::vector<float>& actual,
                   const ProfileRef& p, float threshold, bool& ok, JVal& json_out)
{
    std::vector<float> reference;
    if (!phase12_test::read_bin(p.dir / (name + ".bin"), reference) || reference.size() != actual.size())
    {
        std::cout << "  " << name << " reference missing/size mismatch FAIL\n";
        ok = false;
        json_out.obj[name] = JVal{};
        return false;
    }
    const ErrorStats stats = phase12_test::compare(actual, reference);
    const bool pass = phase12_test::emit(name, stats, threshold);
    ok = ok && pass;
    JVal entry;
    entry.kind = JVal::Object;
    entry.obj["max_abs_error"] = number(stats.maximum);
    entry.obj["mean_abs_error"] = number(stats.mean);
    entry.obj["nan_count"] = number(static_cast<double>(stats.nan));
    entry.obj["inf_count"] = number(static_cast<double>(stats.inf));
    entry.obj["pass"] = boolean(pass);
    json_out.obj[name] = entry;
    return pass;
}

bool check_windows(const DiTBlockRunStats& stats, const ProfileRef& p, bool& ok, JVal& json_out)
{
    JVal entry;
    entry.kind = JVal::Object;
    const auto put = [&entry](const std::string& n, bool v) { entry.obj[n] = boolean(v); };
    bool good = true;
    put("window_counts", stats.windows.window_counts == p.ref_window_count);
    good &= stats.windows.window_counts == p.ref_window_count;
    put("window_sizes", stats.windows.window_sizes == p.ref_window_sizes);
    good &= stats.windows.window_sizes == p.ref_window_sizes;
    put("partition_index", stats.windows.partition_index == p.ref_partition);
    good &= stats.windows.partition_index == p.ref_partition;
    put("reverse_index", stats.windows.reverse_index == p.ref_reverse);
    good &= stats.windows.reverse_index == p.ref_reverse;
    put("segment_lengths", stats.segment_lengths == p.ref_segments);
    good &= stats.segment_lengths == p.ref_segments;
    put("cu_seqlens", stats.cu_seqlens == p.ref_cu);
    good &= stats.cu_seqlens == p.ref_cu;
    put("total_packed", stats.total_packed == p.ref_total_packed);
    good &= stats.total_packed == p.ref_total_packed;
    bool monotonic = true;
    for (std::size_t i = 1; i < stats.cu_seqlens.size(); ++i)
        monotonic = monotonic && stats.cu_seqlens[static_cast<std::size_t>(i)] > stats.cu_seqlens[i - 1];
    put("cu_seqlens_strictly_monotonic", monotonic);
    good &= monotonic;
    std::cout << "  window_counts=" << phase12_test::join(stats.windows.window_counts)
              << " window_sizes=" << phase12_test::join(stats.windows.window_sizes)
              << " total_packed=" << stats.total_packed
              << " cu_monotonic=" << (monotonic ? "PASS" : "FAIL") << '\n';
    ok = ok && good;
    json_out.obj["windows"] = entry;
    return good;
}

bool check_rope_tables(const std::vector<float>& cos, const std::vector<float>& sin,
                       const ProfileRef& p, const std::string& prefix, bool& ok, JVal& json_out)
{
    std::vector<float> freqs;
    if (!phase12_test::read_bin(p.dir / (prefix + ".bin"), freqs)) return false;
    const int rows = static_cast<int>(freqs.size() / 126);
    const int pairs = 64;
    std::vector<float> exp_cos(static_cast<std::size_t>(rows) * pairs, 1.f);
    std::vector<float> exp_sin(static_cast<std::size_t>(rows) * pairs, 0.f);
    for (int r = 0; r < rows; ++r)
        for (int p2 = 0; p2 < 63; ++p2)
        {
            const float angle = freqs[static_cast<std::size_t>(r) * 126 + static_cast<std::size_t>(p2) * 2];
            exp_cos[static_cast<std::size_t>(r) * pairs + p2] = std::cos(angle);
            exp_sin[static_cast<std::size_t>(r) * pairs + p2] = std::sin(angle);
        }
    const ErrorStats c = phase12_test::compare(cos, exp_cos);
    const ErrorStats s = phase12_test::compare(sin, exp_sin);
    const bool cp = phase12_test::emit(prefix + "_cos", c, 1e-6f);
    const bool sp = phase12_test::emit(prefix + "_sin", s, 1e-6f);
    ok = ok && cp && sp;
    JVal entry;
    entry.kind = JVal::Object;
    entry.obj["cos_max_abs_error"] = number(c.maximum);
    entry.obj["sin_max_abs_error"] = number(s.maximum);
    entry.obj["pass"] = boolean(cp && sp);
    json_out.obj[prefix + "_cos_sin"] = entry;
    return true;
}

bool run_once(HybridDiTBlockRuntime& runtime, const ProfileRef& p, DiTBlockTrace& trace,
              JVal& out_json, bool verbose)
{
    DiTBlockInput input = make_input(p, runtime.block().hidden_dim);
    DiTBlockOutput output;
    DiTBlockRunStats stats;
    if (!runtime.run(input, output, &trace, stats))
    {
        std::cerr << "runtime error: " << stats.error_message << '\n';
        return false;
    }
    bool ok = true;
    check_windows(stats, p, ok, out_json);
    check_rope_tables(trace.rope_cos_vid, trace.rope_sin_vid, p, "rope_freqs_vid", ok, out_json);
    check_rope_tables(trace.rope_cos_txt, trace.rope_sin_txt, p, "rope_freqs_txt", ok, out_json);
    JVal stages;
    stages.kind = JVal::Object;
    for (const auto& s : kStages) compare_stage(s.name, trace.*(s.field), p, s.threshold, ok, stages);
    out_json.obj["stages"] = stages;
    out_json.obj["latency_ms"] = number(stats.total_ms);
    out_json.obj["cache_before"] = number(static_cast<double>(stats.cache_size_before));
    out_json.obj["cache_after"] = number(static_cast<double>(stats.cache_size_after));
    out_json.obj["pass"] = boolean(ok);
    if (verbose) std::cout << "  runtime.ok=" << ok << " latency_ms=" << stats.total_ms
                           << " cache=" << stats.cache_size_before << "->" << stats.cache_size_after << '\n';
    return ok;
}

int write_json_file(const fs::path& path, const JVal& value)
{
    std::ofstream file(path);
    if (!file) { std::cerr << "cannot write " << path.string() << '\n'; return 1; }
    file << phase12_test::to_json(value);
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    // registry scans direct children of the root for metadata.json
    const fs::path artifacts_root = argc > 1 ? fs::path(argv[1]) : fs::path(".phase13_local_artifacts");
    const fs::path out_json = argc > 2 ? fs::path(argv[2]) : fs::path();

    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(artifacts_root, error))
    {
        std::cerr << "registry scan failed: " << error << '\n';
        return 1;
    }
    const CoreKey key{4, 3, 3, 1, 1, 1, kVariant};
    const CoreInfo* core = registry.lookup(key);
    if (!core)
    {
        std::cerr << "variant not found: " << key.variant << '\n';
        return 1;
    }
    NcnnModelCache cache;
    HybridDiTBlockRuntime runtime(cache);

    const double ws_before_mb = working_set_mb(false);
    const auto t_init0 = std::chrono::steady_clock::now();
    if (!runtime.initialize(*core, error))
    {
        std::cerr << "runtime init failed: " << error << '\n';
        return 1;
    }
    const auto t_init1 = std::chrono::steady_clock::now();
    const double init_ms = std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();
    const std::size_t cache_after_init = cache.size();
    const double ws_after_load_mb = working_set_mb(false);
    std::cout << "variant=" << core->name << " initialized in " << init_ms << " ms, cache="
              << cache_after_init << ", ws_before=" << ws_before_mb << " MB, ws_after_load="
              << ws_after_load_mb << " MB\n";

    ProfileRef profile;
    if (!load_profile(core->directory / "reference" / "A", profile, error))
    {
        std::cerr << "load profile A: " << error << '\n';
        return 1;
    }
    std::cout << "profile A vid_shapes=[" << profile.vid_shapes[0].t << "," << profile.vid_shapes[0].h
              << "," << profile.vid_shapes[0].w << "] txt_lengths="
              << phase12_test::join(profile.txt_lengths) << '\n';

    // RW-A -> RW-A (two complete forwards in one process)
    DiTBlockTrace trace1, trace2;
    JVal run1, run2;
    run1.kind = JVal::Object;
    run2.kind = JVal::Object;
    const bool ok1 = run_once(runtime, profile, trace1, run1, true);
    const bool ok2 = run_once(runtime, profile, trace2, run2, true);
    const bool det = ok1 && ok2 && trace1.final_vid == trace2.final_vid && trace1.final_txt == trace2.final_txt;
    std::cout << "determinism RW-A->RW-A vid=" << (det ? "bit-identical" : "FAIL")
              << " txt=" << (det ? "bit-identical" : "FAIL") << '\n';
    const double ws_peak_mb = working_set_mb(true);
    const double ws_after_runs_mb = working_set_mb(false);

    JVal report;
    report.kind = JVal::Object;
    report.obj["variant"] = text(core->name);
    report.obj["sequence"] = text("RW-A,RW-A");
    report.obj["run1"] = run1;
    report.obj["run2"] = run2;
    report.obj["determinism"] = boolean(det);
    report.obj["init_ms_first_load"] = number(init_ms);
    report.obj["cache_size_after_init"] = number(static_cast<double>(cache_after_init));
    report.obj["working_set_mb_before"] = number(ws_before_mb);
    report.obj["working_set_mb_after_load"] = number(ws_after_load_mb);
    report.obj["peak_working_set_mb"] = number(ws_peak_mb);
    report.obj["working_set_mb_after_runs"] = number(ws_after_runs_mb);
    report.obj["real_seedvr2_weights"] = boolean(false);
    report.obj["realwidth_block"] = boolean(true);
    report.obj["full_32_block_dit"] = boolean(false);
    report.obj["android_vulkan"] = boolean(false);
    report.obj["large_artifacts_committed"] = boolean(false);
    report.obj["overall_pass"] = boolean(ok1 && ok2 && det);
    report.obj["overall_status"] = text(ok1 && ok2 && det ? "PASS" : "FAIL");

    if (!out_json.empty())
    {
        if (write_json_file(out_json, report)) return 1;
        std::cout << "report written: " << out_json << '\n';
    }
    return (ok1 && ok2 && det) ? 0 : 1;
}
