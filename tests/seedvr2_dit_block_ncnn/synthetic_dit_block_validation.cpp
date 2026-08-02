// Phase 12: per-stage parity validation of the synthetic SeedVR2 DiT block
// ncnn hybrid runtime against the generated PyTorch reference.
//
// Runs the full A -> B -> C -> D -> A profile sequence 5 times, prints per
// stage shape / max-abs-error / NaN / Inf / cache state / latency and writes a
// stdlib-parseable JSON report (no NaN/Infinity) when an output path is given.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    const char* name;                          // reference bin basename
    std::vector<float> DiTBlockTrace::*field;  // trace member
    float threshold;
};

const StageSpec kStages[] = {
    {"attn_norm_vid", &DiTBlockTrace::attn_norm_vid, 1e-5f},
    {"attn_norm_txt", &DiTBlockTrace::attn_norm_txt, 1e-5f},
    // AdaSingle "in" stages are exact elementwise (hid*(scaleA+scaleB)+(shiftA+shiftB));
    // their deviation is the upstream ncnn RMSNorm FP32 noise (~5e-7) amplified by the
    // modulation scale; measured max across A/B/C/D/A is 6.7e-6, so 1e-5 is used.
    {"ada_attn_in_vid", &DiTBlockTrace::ada_attn_in_vid, 1e-5f},
    {"ada_attn_in_txt", &DiTBlockTrace::ada_attn_in_txt, 1e-5f},
    {"qkv_vid", &DiTBlockTrace::qkv_vid, 1e-5f},
    {"qkv_txt", &DiTBlockTrace::qkv_txt, 1e-5f},
    {"q_vid", &DiTBlockTrace::q_vid, 1e-5f},
    {"k_vid", &DiTBlockTrace::k_vid, 1e-5f},
    {"v_vid", &DiTBlockTrace::v_vid, 1e-5f},
    {"q_txt", &DiTBlockTrace::q_txt, 1e-5f},
    {"k_txt", &DiTBlockTrace::k_txt, 1e-5f},
    {"v_txt", &DiTBlockTrace::v_txt, 1e-5f},
    {"q_norm_vid", &DiTBlockTrace::q_norm_vid, 1e-5f},
    {"k_norm_vid", &DiTBlockTrace::k_norm_vid, 1e-5f},
    {"q_norm_txt", &DiTBlockTrace::q_norm_txt, 1e-5f},
    {"k_norm_txt", &DiTBlockTrace::k_norm_txt, 1e-5f},
    {"rope_before_q_vid", &DiTBlockTrace::rope_before_q_vid, 1e-5f},
    {"rope_before_k_vid", &DiTBlockTrace::rope_before_k_vid, 1e-5f},
    {"rope_after_q_vid", &DiTBlockTrace::rope_after_q_vid, 1e-5f},
    {"rope_after_k_vid", &DiTBlockTrace::rope_after_k_vid, 1e-5f},
    {"rope_after_q_txt", &DiTBlockTrace::rope_after_q_txt, 1e-5f},
    {"rope_after_k_txt", &DiTBlockTrace::rope_after_k_txt, 1e-5f},
    {"packed_q", &DiTBlockTrace::packed_q, 1e-5f},
    {"packed_k", &DiTBlockTrace::packed_k, 1e-5f},
    {"packed_v", &DiTBlockTrace::packed_v, 1e-5f},
    {"attention_packed_output", &DiTBlockTrace::attention_packed_output, 1e-4f},
    {"attention_vid_windows", &DiTBlockTrace::attention_vid_windows, 1e-4f},
    {"attention_txt_windows", &DiTBlockTrace::attention_txt_windows, 1e-4f},
    {"attention_vid_merged", &DiTBlockTrace::attention_vid_merged, 1e-4f},
    {"attention_txt_mean_merged", &DiTBlockTrace::attention_txt_mean_merged, 1e-4f},
    {"attention_proj_vid", &DiTBlockTrace::attention_proj_vid, 1e-4f},
    {"attention_proj_txt", &DiTBlockTrace::attention_proj_txt, 1e-4f},
    {"ada_attn_gate_vid", &DiTBlockTrace::ada_attn_gate_vid, 1e-4f},
    {"ada_attn_gate_txt", &DiTBlockTrace::ada_attn_gate_txt, 1e-4f},
    {"attention_residual_vid", &DiTBlockTrace::attention_residual_vid, 1e-4f},
    {"attention_residual_txt", &DiTBlockTrace::attention_residual_txt, 1e-4f},
    {"mlp_norm_vid", &DiTBlockTrace::mlp_norm_vid, 1e-5f},
    {"mlp_norm_txt", &DiTBlockTrace::mlp_norm_txt, 1e-5f},
    {"ada_mlp_in_vid", &DiTBlockTrace::ada_mlp_in_vid, 1e-5f},
    {"ada_mlp_in_txt", &DiTBlockTrace::ada_mlp_in_txt, 1e-5f},
    {"swiglu_gate_projection_vid", &DiTBlockTrace::swiglu_gate_vid, 1e-4f},
    {"swiglu_up_projection_vid", &DiTBlockTrace::swiglu_up_vid, 1e-4f},
    {"swiglu_product_vid", &DiTBlockTrace::swiglu_product_vid, 1e-4f},
    {"swiglu_output_vid", &DiTBlockTrace::swiglu_output_vid, 1e-4f},
    {"swiglu_gate_projection_txt", &DiTBlockTrace::swiglu_gate_txt, 1e-4f},
    {"swiglu_up_projection_txt", &DiTBlockTrace::swiglu_up_txt, 1e-4f},
    {"swiglu_product_txt", &DiTBlockTrace::swiglu_product_txt, 1e-4f},
    {"swiglu_output_txt", &DiTBlockTrace::swiglu_output_txt, 1e-4f},
    {"ada_mlp_gate_vid", &DiTBlockTrace::ada_mlp_gate_vid, 1e-4f},
    {"ada_mlp_gate_txt", &DiTBlockTrace::ada_mlp_gate_txt, 1e-4f},
    {"final_vid", &DiTBlockTrace::final_vid, 1e-4f},
    {"final_txt", &DiTBlockTrace::final_txt, 1e-4f},
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
    // freqs: [rows, 126] angles; expected cos/sin: [rows, 64] pairs (pair 63 identity)
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
    const fs::path artifacts_root = argc > 1 ? fs::path(argv[1]) : fs::path("tests/seedvr2_dit_block_ncnn/artifacts");
    const fs::path out_json = argc > 2 ? fs::path(argv[2]) : fs::path();
    const char* profile_names[] = {"A", "B", "C", "D", "A"};

    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(artifacts_root, error))
    {
        std::cerr << "registry scan failed: " << error << '\n';
        return 1;
    }
    const CoreKey key{4, 3, 3, 1, 1, 1, "synthetic_seedvr2_dit_block_c512_h4_hd128"};
    const CoreInfo* core = registry.lookup(key);
    if (!core)
    {
        std::cerr << "variant not found: " << key.variant << '\n';
        return 1;
    }
    NcnnModelCache cache;
    HybridDiTBlockRuntime runtime(cache);
    if (!runtime.initialize(*core, error))
    {
        std::cerr << "runtime init failed: " << error << '\n';
        return 1;
    }
    const std::size_t cache_after_init = cache.size();
    std::cout << "variant=" << core->name << " initialized, cache size after init=" << cache_after_init << '\n';

    ProfileRef profiles[4];
    for (int i = 0; i < 4; ++i)
    {
        const fs::path dir = core->directory / "reference" / profile_names[i];
        if (!load_profile(dir, profiles[i], error))
        {
            std::cerr << "load profile " << profile_names[i] << ": " << error << '\n';
            return 1;
        }
        std::cout << "profile " << profile_names[i] << " vid_shapes=";
        for (const auto& s : profiles[i].vid_shapes) std::cout << "[" << s.t << "," << s.h << "," << s.w << "]";
        std::cout << " txt_lengths=" << phase12_test::join(profiles[i].txt_lengths) << '\n';
    }

    JVal report;
    report.kind = JVal::Object;
    report.obj["variant"] = text(core->name);
    report.obj["repetitions"] = number(5);
    report.obj["sequence"] = text("A,B,C,D,A");
    report.obj["cache_size_after_init"] = number(static_cast<double>(cache_after_init));
    // Provenance flags: this fixture uses deterministic synthetic weights and the
    // documented FP32 CPU-compat attention path, never real SeedVR2 weights, real
    // flash-attn/apex CUDA kernels, a full 32-block DiT, or an Android/Vulkan build.
    report.obj["real_seedvr2_weights"] = boolean(false);
    report.obj["real_flash_attn_cuda_kernel"] = boolean(false);
    report.obj["real_apex_cuda_kernel"] = boolean(false);
    report.obj["full_32_block_dit"] = boolean(false);
    report.obj["android_vulkan"] = boolean(false);
    JVal runs;
    runs.kind = JVal::Array;
    bool overall = true;
    std::vector<float> first_a_vid, first_a_txt;
    bool first_a_captured = false;
    bool det_vid = false, det_txt = false;

    for (int rep = 0; rep < 5; ++rep)
    {
        std::cout << "=== repetition " << (rep + 1) << "/5 ===\n";
        for (int i = 0; i < 5; ++i)
        {
            const int pidx = (i == 4) ? 0 : i;
            const char* name = profile_names[i];
            DiTBlockTrace trace;
            JVal run_json;
            run_json.kind = JVal::Object;
            run_json.obj["profile"] = text(name);
            run_json.obj["repetition"] = number(static_cast<double>(rep + 1));
            const bool ok = run_once(runtime, profiles[pidx], trace, run_json, true);
            overall = overall && ok;
            if (i == 0 && !first_a_captured)
            {
                first_a_vid = trace.final_vid;
                first_a_txt = trace.final_txt;
                first_a_captured = true;
            }
            if (i == 4)
            {
                det_vid = first_a_vid.size() == trace.final_vid.size() &&
                          std::memcmp(first_a_vid.data(), trace.final_vid.data(), first_a_vid.size() * sizeof(float)) == 0;
                det_txt = first_a_txt.size() == trace.final_txt.size() &&
                          std::memcmp(first_a_txt.data(), trace.final_txt.data(), first_a_txt.size() * sizeof(float)) == 0;
                std::cout << "  determinism A(first) vs A(last): vid "
                          << (det_vid ? "BIT-IDENTICAL" : "DIFF") << " txt "
                          << (det_txt ? "BIT-IDENTICAL" : "DIFF") << '\n';
                JVal det;
                det.kind = JVal::Object;
                det.obj["vid_bit_identical"] = boolean(det_vid);
                det.obj["txt_bit_identical"] = boolean(det_txt);
                run_json.obj["determinism"] = det;
                overall = overall && det_vid && det_txt;
            }
            runs.arr.push_back(run_json);
        }
    }
    report.obj["runs"] = runs;
    JVal det;
    det.kind = JVal::Object;
    det.obj["vid_bit_identical"] = boolean(det_vid);
    det.obj["txt_bit_identical"] = boolean(det_txt);
    report.obj["determinism"] = det;
    report.obj["overall_status"] = text(overall ? "PASS" : "FAIL");
    report.obj["overall_pass"] = boolean(overall);

    std::cout << "\n=== overall: " << (overall ? "PASS" : "FAIL") << " ===\n";
    if (!out_json.empty()) return write_json_file(out_json, report);
    return overall ? 0 : 2;
}
