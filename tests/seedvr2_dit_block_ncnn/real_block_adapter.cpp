// Phase 14: real-block ncnn adapter executable.
//
// Runs the real-weight SeedVR2 block-16 HybridDiTBlockRuntime on captured
// official-context inputs (input_vid.bin, input_txt.bin, emb.bin,
// vid_shapes.json, txt_lengths.json) and writes output bins + a JSON report.
// This executable IS the ncnn path: there is no PyTorch fallback, so
// original_block_call_count is always 0 by construction and the adapter prints
// `HYBRID BLOCK EXECUTED` with content hashes so the Python driver can verify
// the hybrid path was taken (no silent fallback).
#include <chrono>
#include <cstdint>
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
using phase12_test::JVal;
using phase12_test::parse_json;

namespace
{
JVal number(double v) { JVal j; j.kind = JVal::Number; j.num = v; return j; }
JVal boolean(bool v) { JVal j; j.kind = JVal::Bool; j.b = v; return j; }
JVal text(const std::string& v) { JVal j; j.kind = JVal::String; j.str = v; return j; }

uint64_t fnv1a64(const void* data, std::size_t n)
{
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

std::string hash_hex(const std::vector<float>& v)
{
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(fnv1a64(v.data(), v.size() * sizeof(float))));
    return buf;
}

bool read_json_ints(const fs::path& p, std::vector<int>& out)
{
    std::string s;
    if (!phase12_test::read_text(p, s)) return false;
    out = phase12_test::int_array(parse_json(s));
    return true;
}

bool write_bin(const fs::path& p, const std::vector<float>& v)
{
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return bool(f);
}

int usage()
{
    std::cerr << "usage: real_block_adapter --root <artifacts_root> --variant <name>"
                 " --input-dir <dir> --output-dir <dir> [--report <json>] [--trace-dir <dir>]\n";
    return 2;
}
} // namespace

int main(int argc, char** argv)
{
    fs::path root, input_dir, output_dir, report_path, trace_dir;
    std::string variant;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&](const char* n) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << n << '\n'; std::exit(2); }
            return argv[++i];
        };
        if (a == "--root") root = next("--root");
        else if (a == "--variant") variant = next("--variant");
        else if (a == "--input-dir") input_dir = next("--input-dir");
        else if (a == "--output-dir") output_dir = next("--output-dir");
        else if (a == "--report") report_path = next("--report");
        else if (a == "--trace-dir") trace_dir = next("--trace-dir");
        else { std::cerr << "unknown arg " << a << '\n'; return usage(); }
    }
    if (root.empty() || variant.empty() || input_dir.empty() || output_dir.empty()) return usage();

    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(root, error)) { std::cerr << "registry scan failed: " << error << '\n'; return 1; }
    const CoreKey key{4, 3, 3, 1, 1, 1, variant};
    const CoreInfo* core = registry.lookup(key);
    if (!core) { std::cerr << "variant not found: " << variant << '\n'; return 1; }

    NcnnModelCache cache;
    HybridDiTBlockRuntime runtime(cache);
    if (!runtime.initialize(*core, error)) { std::cerr << "runtime init failed: " << error << '\n'; return 1; }

    DiTBlockInput in;
    std::vector<float> vid, txt, emb;
    if (!phase12_test::read_bin(input_dir / "input_vid.bin", vid) ||
        !phase12_test::read_bin(input_dir / "input_txt.bin", txt) ||
        !phase12_test::read_bin(input_dir / "emb.bin", emb))
    {
        std::cerr << "input bins missing in " << input_dir << '\n';
        return 1;
    }
    std::vector<int> shapes_flat;
    if (!read_json_ints(input_dir / "vid_shapes.json", shapes_flat) || shapes_flat.empty() || shapes_flat.size() % 3 != 0)
    {
        std::cerr << "vid_shapes.json invalid\n";
        return 1;
    }
    std::vector<int> txt_lengths;
    if (!read_json_ints(input_dir / "txt_lengths.json", txt_lengths) || txt_lengths.empty())
    {
        std::cerr << "txt_lengths.json invalid\n";
        return 1;
    }
    const int dim = runtime.block().hidden_dim;
    for (std::size_t i = 0; i < shapes_flat.size(); i += 3)
        in.vid_shapes.push_back({shapes_flat[i], shapes_flat[i + 1], shapes_flat[i + 2]});
    in.txt_lengths = txt_lengths;
    in.vid = ncnn::Mat(dim, static_cast<int>(vid.size() / dim));
    std::memcpy(in.vid.data, vid.data(), vid.size() * sizeof(float));
    in.txt = ncnn::Mat(dim, static_cast<int>(txt.size() / dim));
    std::memcpy(in.txt.data, txt.data(), txt.size() * sizeof(float));
    in.emb = ncnn::Mat(6 * dim, static_cast<int>(emb.size() / (6 * dim)));
    std::memcpy(in.emb.data, emb.data(), emb.size() * sizeof(float));

    DiTBlockTrace trace;
    DiTBlockOutput out;
    DiTBlockRunStats stats;
    const auto t0 = std::chrono::steady_clock::now();
    if (!runtime.run(in, out, &trace, stats))
    {
        std::cerr << "runtime error: " << stats.error_message << '\n';
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    fs::create_directories(output_dir);
    const std::vector<float> out_vid(static_cast<const float*>(out.vid.data),
                                     static_cast<const float*>(out.vid.data) + out.vid.total());
    const std::vector<float> out_txt(static_cast<const float*>(out.txt.data),
                                     static_cast<const float*>(out.txt.data) + out.txt.total());
    if (!write_bin(output_dir / "output_vid.bin", out_vid) || !write_bin(output_dir / "output_txt.bin", out_txt))
    {
        std::cerr << "cannot write output bins\n";
        return 1;
    }

    JVal report;
    report.kind = JVal::Object;
    report.obj["variant"] = text(variant);
    report.obj["block_index"] = number(16);
    report.obj["success"] = boolean(true);
    report.obj["latency_ms"] = number(ms);
    report.obj["cache_size"] = number(static_cast<double>(cache.size()));
    report.obj["total_packed"] = number(static_cast<double>(stats.total_packed));
    report.obj["vid_hash"] = text(hash_hex(out_vid));
    report.obj["txt_hash"] = text(hash_hex(out_txt));
    report.obj["ncnn_adapter_call_count"] = number(1);
    report.obj["original_block_call_count"] = number(0);
    report.obj["silent_fallback"] = boolean(false);

    if (!trace_dir.empty())
    {
        fs::create_directories(trace_dir);
        std::vector<std::pair<const char*, const std::vector<float>*>> stages = {
            {"attn_norm_vid", &trace.attn_norm_vid}, {"attn_norm_txt", &trace.attn_norm_txt},
            {"ada_attn_in_vid", &trace.ada_attn_in_vid}, {"ada_attn_in_txt", &trace.ada_attn_in_txt},
            {"qkv_vid", &trace.qkv_vid}, {"qkv_txt", &trace.qkv_txt},
            {"q_vid", &trace.q_vid}, {"k_vid", &trace.k_vid}, {"v_vid", &trace.v_vid},
            {"q_txt", &trace.q_txt}, {"k_txt", &trace.k_txt}, {"v_txt", &trace.v_txt},
            {"q_norm_vid", &trace.q_norm_vid}, {"k_norm_vid", &trace.k_norm_vid},
            {"q_norm_txt", &trace.q_norm_txt}, {"k_norm_txt", &trace.k_norm_txt},
            {"rope_after_q_vid", &trace.rope_after_q_vid}, {"rope_after_k_vid", &trace.rope_after_k_vid},
            {"rope_after_q_txt", &trace.rope_after_q_txt}, {"rope_after_k_txt", &trace.rope_after_k_txt},
            {"attention_packed_output", &trace.attention_packed_output},
            {"attention_vid_windows", &trace.attention_vid_windows},
            {"attention_txt_windows", &trace.attention_txt_windows},
            {"attention_vid_merged", &trace.attention_vid_merged},
            {"attention_txt_mean_merged", &trace.attention_txt_mean_merged},
            {"attention_proj_vid", &trace.attention_proj_vid}, {"attention_proj_txt", &trace.attention_proj_txt},
            {"ada_attn_gate_vid", &trace.ada_attn_gate_vid}, {"ada_attn_gate_txt", &trace.ada_attn_gate_txt},
            {"attention_residual_vid", &trace.attention_residual_vid}, {"attention_residual_txt", &trace.attention_residual_txt},
            {"mlp_norm_vid", &trace.mlp_norm_vid}, {"mlp_norm_txt", &trace.mlp_norm_txt},
            {"ada_mlp_in_vid", &trace.ada_mlp_in_vid}, {"ada_mlp_in_txt", &trace.ada_mlp_in_txt},
            {"swiglu_gate_vid", &trace.swiglu_gate_vid}, {"swiglu_up_vid", &trace.swiglu_up_vid},
            {"swiglu_product_vid", &trace.swiglu_product_vid}, {"swiglu_output_vid", &trace.swiglu_output_vid},
            {"swiglu_gate_txt", &trace.swiglu_gate_txt}, {"swiglu_up_txt", &trace.swiglu_up_txt},
            {"swiglu_product_txt", &trace.swiglu_product_txt}, {"swiglu_output_txt", &trace.swiglu_output_txt},
            {"ada_mlp_gate_vid", &trace.ada_mlp_gate_vid}, {"ada_mlp_gate_txt", &trace.ada_mlp_gate_txt},
            {"final_vid", &trace.final_vid}, {"final_txt", &trace.final_txt},
        };
        for (const auto& s : stages)
            write_bin(trace_dir / (std::string(s.first) + ".bin"), *s.second);
        report.obj["trace_stage_count"] = number(static_cast<double>(stages.size()));
    }

    if (!report_path.empty())
    {
        std::ofstream f(report_path);
        if (f) f << phase12_test::to_json(report);
    }
    std::cout << "HYBRID BLOCK EXECUTED block_index=16 runtime=" << variant
              << " vid_hash=" << report.obj.at("vid_hash").str
              << " txt_hash=" << report.obj.at("txt_hash").str
              << " ncnn_adapter_call_count=1 original_block_call_count=0"
              << " silent_fallback=false latency_ms=" << ms << '\n';
    return 0;
}
