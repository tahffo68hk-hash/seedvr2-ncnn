// Phase 12: last-layer contract validation for the synthetic SeedVR2 DiT block.
// Verifies the real vid_only semantics: mlp_norm/ada/mlp are identity on the
// txt branch, so final_txt == 2 * attention_residual_txt, while the vid branch
// runs the full staged MLP.  Runs profiles A and B three times each.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
using phase12_test::ErrorStats;
using phase12_test::JVal;

namespace
{
struct LLProfile
{
    fs::path dir;
    std::vector<seedvr2_dit_block::VideoShape> vid_shapes;
    std::vector<int> txt_lengths;
    std::vector<float> vid, txt, emb;
};

bool load_profile(const fs::path& dir, LLProfile& out, std::string& error)
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
    error.clear();
    return true;
}

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

bool compare_stage(const std::string& name, const std::vector<float>& actual,
                   const LLProfile& p, float threshold, bool& ok, JVal& json_out)
{
    std::vector<float> reference;
    const fs::path bin = p.dir / (name + ".bin");
    if (!fs::exists(bin)) return true; // stage not captured for last layer
    if (!phase12_test::read_bin(bin, reference) || reference.size() != actual.size())
    {
        std::cout << "  " << name << " reference missing/size mismatch FAIL\n";
        ok = false;
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

// last-layer txt semantics: final_txt == 2 * attention_residual_txt
bool check_last_layer_txt(const DiTBlockTrace& trace, const LLProfile& p, bool& ok, JVal& json_out)
{
    std::vector<float> residual;
    if (!phase12_test::read_bin(p.dir / "attention_residual_txt.bin", residual) || residual.size() != trace.final_txt.size())
    {
        std::cerr << "last-layer txt residual reference missing\n";
        ok = false;
        return false;
    }
    std::vector<float> twice(residual.size());
    for (std::size_t i = 0; i < twice.size(); ++i) twice[i] = 2.f * residual[i];
    const ErrorStats stats = phase12_test::compare(trace.final_txt, twice);
    const bool pass = phase12_test::emit("final_txt==2*attention_residual_txt", stats, 1e-4f);
    ok = ok && pass;
    JVal entry;
    entry.kind = JVal::Object;
    entry.obj["max_abs_error"] = number(stats.maximum);
    entry.obj["pass"] = boolean(pass);
    json_out.obj["last_layer_txt_semantics"] = entry;
    return pass;
}

struct StageSpec
{
    const char* name;
    std::vector<float> DiTBlockTrace::*field;
    float threshold;
};

const StageSpec kStages[] = {
    {"attn_norm_vid", &DiTBlockTrace::attn_norm_vid, 1e-5f},
    {"attn_norm_txt", &DiTBlockTrace::attn_norm_txt, 1e-5f},
    // AdaSingle "in" deviation is upstream RMSNorm FP32 noise amplified by scale
    // (see synthetic_dit_block_validation.cpp); 1e-5 with measured max 3.8e-6.
    {"ada_attn_in_vid", &DiTBlockTrace::ada_attn_in_vid, 1e-5f},
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
    {"attention_residual_vid", &DiTBlockTrace::attention_residual_vid, 1e-4f},
    {"attention_residual_txt", &DiTBlockTrace::attention_residual_txt, 1e-4f},
    {"mlp_norm_vid", &DiTBlockTrace::mlp_norm_vid, 1e-5f},
    {"ada_mlp_in_vid", &DiTBlockTrace::ada_mlp_in_vid, 1e-5f},
    {"swiglu_gate_projection_vid", &DiTBlockTrace::swiglu_gate_vid, 1e-4f},
    {"swiglu_up_projection_vid", &DiTBlockTrace::swiglu_up_vid, 1e-4f},
    {"swiglu_product_vid", &DiTBlockTrace::swiglu_product_vid, 1e-4f},
    {"swiglu_output_vid", &DiTBlockTrace::swiglu_output_vid, 1e-4f},
    {"ada_mlp_gate_vid", &DiTBlockTrace::ada_mlp_gate_vid, 1e-4f},
    {"final_vid", &DiTBlockTrace::final_vid, 1e-4f},
    {"final_txt", &DiTBlockTrace::final_txt, 1e-4f},
};

bool run_once(HybridDiTBlockRuntime& runtime, const LLProfile& p, DiTBlockTrace& trace,
              JVal& out_json, bool verbose)
{
    DiTBlockInput input;
    const int dim = runtime.block().hidden_dim;
    input.vid = ncnn::Mat(dim, static_cast<int>(p.vid.size() / dim));
    std::memcpy(input.vid.data, p.vid.data(), p.vid.size() * sizeof(float));
    input.txt = ncnn::Mat(dim, static_cast<int>(p.txt.size() / dim));
    std::memcpy(input.txt.data, p.txt.data(), p.txt.size() * sizeof(float));
    input.emb = ncnn::Mat(6 * dim, static_cast<int>(p.emb.size() / (6 * dim)));
    std::memcpy(input.emb.data, p.emb.data(), p.emb.size() * sizeof(float));
    input.vid_shapes = p.vid_shapes;
    input.txt_lengths = p.txt_lengths;
    DiTBlockOutput output;
    DiTBlockRunStats stats;
    if (!runtime.run(input, output, &trace, stats))
    {
        std::cerr << "runtime error: " << stats.error_message << '\n';
        return false;
    }
    bool ok = true;
    JVal stages;
    stages.kind = JVal::Object;
    for (const auto& s : kStages) compare_stage(s.name, trace.*(s.field), p, s.threshold, ok, stages);
    check_last_layer_txt(trace, p, ok, stages);
    out_json.obj["stages"] = stages;
    out_json.obj["latency_ms"] = number(stats.total_ms);
    out_json.obj["cache_before"] = number(static_cast<double>(stats.cache_size_before));
    out_json.obj["cache_after"] = number(static_cast<double>(stats.cache_size_after));
    out_json.obj["pass"] = boolean(ok);
    if (verbose) std::cout << "  runtime.ok=" << ok << " latency_ms=" << stats.total_ms << '\n';
    return ok;
}
} // namespace

int main(int argc, char** argv)
{
    const fs::path artifacts_root = argc > 1 ? fs::path(argv[1]) : fs::path("tests/seedvr2_dit_block_ncnn/artifacts");
    const fs::path out_json = argc > 2 ? fs::path(argv[2]) : fs::path();

    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(artifacts_root, error))
    {
        std::cerr << "registry scan failed: " << error << '\n';
        return 1;
    }
    const CoreKey key{4, 3, 3, 1, 1, 1, "synthetic_seedvr2_dit_block_c512_h4_hd128_last_layer"};
    const CoreInfo* core = registry.lookup(key);
    if (!core)
    {
        std::cerr << "last-layer variant not found\n";
        return 1;
    }
    if (!core->dit_block() || !core->dit_block()->block.is_last_layer)
    {
        std::cerr << "last-layer core metadata mismatch\n";
        return 1;
    }
    NcnnModelCache cache;
    HybridDiTBlockRuntime runtime(cache);
    if (!runtime.initialize(*core, error))
    {
        std::cerr << "runtime init failed: " << error << '\n';
        return 1;
    }
    std::cout << "last-layer variant=" << core->name << " initialized, cache=" << cache.size() << '\n';

    JVal report;
    report.kind = JVal::Object;
    report.obj["variant"] = phase12_test::JVal{};
    report.obj["variant"] = JVal{JVal::String, false, 0, core->name, {}, {}};
    JVal runs;
    runs.kind = JVal::Array;
    bool overall = true;
    std::vector<float> first_a_vid, first_a_txt;
    bool first_captured = false;
    bool det_vid = false, det_txt = false;

    const char* names[] = {"A", "B"};
    for (int rep = 0; rep < 3; ++rep)
    {
        std::cout << "=== last-layer repetition " << (rep + 1) << "/3 ===\n";
        for (const char* name : names)
        {
            LLProfile profile;
            if (!load_profile(core->directory / "reference" / name, profile, error))
            {
                std::cerr << "load profile " << name << ": " << error << '\n';
                return 1;
            }
            DiTBlockTrace trace;
            JVal run_json;
            run_json.kind = JVal::Object;
            run_json.obj["profile"] = JVal{JVal::String, false, 0, name, {}, {}};
            run_json.obj["repetition"] = number(static_cast<double>(rep + 1));
            const bool ok = run_once(runtime, profile, trace, run_json, true);
            overall = overall && ok;
            if (!first_captured)
            {
                first_a_vid = trace.final_vid;
                first_a_txt = trace.final_txt;
                first_captured = true;
            }
            else if (std::string(name) == "A")
            {
                det_vid = first_a_vid.size() == trace.final_vid.size() &&
                          std::memcmp(first_a_vid.data(), trace.final_vid.data(), first_a_vid.size() * sizeof(float)) == 0;
                det_txt = first_a_txt.size() == trace.final_txt.size() &&
                          std::memcmp(first_a_txt.data(), trace.final_txt.data(), first_a_txt.size() * sizeof(float)) == 0;
                std::cout << "  determinism A(1) vs A(3): vid "
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
    report.obj["overall_status"] = JVal{JVal::String, false, 0, overall ? "PASS" : "FAIL", {}, {}};
    report.obj["overall_pass"] = boolean(overall);

    std::cout << "\n=== last-layer overall: " << (overall ? "PASS" : "FAIL") << " ===\n";
    if (!out_json.empty())
    {
        std::ofstream file(out_json);
        if (!file) { std::cerr << "cannot write " << out_json.string() << '\n'; return 1; }
        file << phase12_test::to_json(report);
    }
    return overall ? 0 : 2;
}
