// Phase 12: SwiGLU contract tests against the staged ncnn SwiGLU artifacts.
// Verifies gate/up/product/output parity with the PyTorch reference across
// dynamic token counts (profiles B/C/D), plus empty-input / wrong-dim / NaN
// error handling and deterministic repeatability.
#include <cmath>
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
using seedvr2_dit_block::HybridDiTBlockRuntime;
using seedvr2_dit_block::SwigluStageOutput;

namespace
{
int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what)
{
    ++g_checks;
    std::cout << (ok ? "  PASS " : "  FAIL ") << what << '\n';
    if (!ok) ++g_failures;
}

bool read_bin(const fs::path& path, std::vector<float>& out)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto bytes = file.tellg();
    if (bytes % 4 != 0 || bytes <= 0) return false;
    out.resize(static_cast<std::size_t>(bytes / 4));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(out.data()), bytes);
    return bool(file);
}

float maxdiff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return INFINITY;
    float m = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

bool compare_stage(const std::string& name, const std::vector<float>& actual,
                   const fs::path& ref_dir, float threshold)
{
    std::vector<float> reference;
    if (!read_bin(ref_dir / (name + ".bin"), reference) || reference.size() != actual.size())
    {
        std::cout << "  " << name << " reference missing/size mismatch FAIL\n";
        return false;
    }
    const phase12_test::ErrorStats stats = phase12_test::compare(actual, reference);
    return phase12_test::emit(name, stats, threshold);
}

bool profile_swiglu_parity(HybridDiTBlockRuntime& runtime, const fs::path& ref_dir,
                           const std::string& side, float threshold)
{
    std::vector<float> mlp_in;
    if (!read_bin(ref_dir / ("ada_mlp_in_" + side + ".bin"), mlp_in)) return false;
    const int dim = runtime.block().hidden_dim;
    ncnn::Mat input(dim, static_cast<int>(mlp_in.size() / dim));
    std::memcpy(input.data, mlp_in.data(), mlp_in.size() * sizeof(float));
    SwigluStageOutput out;
    std::string error;
    if (!runtime.swiglu_forward(side, input, out, error))
    {
        std::cout << "  swiglu_forward(" << side << ") error: " << error << '\n';
        return false;
    }
    bool ok = true;
    ok &= compare_stage("swiglu_gate_projection_" + side, out.gate, ref_dir, threshold);
    ok &= compare_stage("swiglu_up_projection_" + side, out.up, ref_dir, threshold);
    ok &= compare_stage("swiglu_product_" + side, out.product, ref_dir, threshold);
    ok &= compare_stage("swiglu_output_" + side, out.output, ref_dir, threshold);
    return ok;
}
} // namespace

int main(int argc, char** argv)
{
    const fs::path artifacts_root = argc > 1 ? fs::path(argv[1]) : fs::path("tests/seedvr2_dit_block_ncnn/artifacts");
    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(artifacts_root, error))
    {
        std::cerr << "registry scan failed: " << error << '\n';
        return 2;
    }
    const CoreInfo* base = registry.lookup(CoreKey{4, 3, 3, 1, 1, 1, "synthetic_seedvr2_dit_block_c512_h4_hd128"});
    if (!base)
    {
        std::cerr << "base variant missing\n";
        return 2;
    }
    NcnnModelCache cache;
    HybridDiTBlockRuntime runtime(cache);
    if (!runtime.initialize(*base, error))
    {
        std::cerr << "runtime init failed: " << error << '\n';
        return 2;
    }
    const int dim = runtime.block().hidden_dim;
    const int mlp = runtime.block().mlp_hidden_dim;
    std::cout << "mlp_hidden_dim=" << mlp << " (256*ceil(2*512*4/3/256)=1536 derived by real helper)\n";

    std::cout << "== SwiGLU stage parity (dynamic token counts) ==\n";
    const char* profiles[] = {"A", "B", "C", "D"};
    bool all_ok = true;
    for (const char* name : profiles)
    {
        const fs::path ref_dir = base->directory / "reference" / name;
        std::cout << "  -- profile " << name << " --\n";
        all_ok &= profile_swiglu_parity(runtime, ref_dir, "vid", 1e-4f);
        all_ok &= profile_swiglu_parity(runtime, ref_dir, "txt", 1e-4f);
    }
    check(all_ok, "swiglu gate/up/product/output parity A/B/C/D (vid+txt)");

    std::cout << "== SwiGLU deterministic repeat ==\n";
    {
        std::vector<float> mlp_in;
        read_bin(base->directory / "reference" / "D" / "ada_mlp_in_vid.bin", mlp_in);
        ncnn::Mat input(dim, static_cast<int>(mlp_in.size() / dim));
        std::memcpy(input.data, mlp_in.data(), mlp_in.size() * sizeof(float));
        SwigluStageOutput o1, o2;
        std::string e1, e2;
        check(runtime.swiglu_forward("vid", input, o1, e1) && runtime.swiglu_forward("vid", input, o2, e2),
              "swiglu_forward repeatable");
        check(o1.output == o2.output && o1.product == o2.product, "swiglu repeat bit-identical");
    }

    std::cout << "== SwiGLU negative tests ==\n";
    {
        // empty tokens
        ncnn::Mat empty(dim, 0);
        SwigluStageOutput out;
        std::string err;
        check(!runtime.swiglu_forward("vid", empty, out, err) && !err.empty(), "empty input rejected");

        // wrong hidden dim
        ncnn::Mat wrong_dim(1024, 8);
        check(!runtime.swiglu_forward("vid", wrong_dim, out, err) && !err.empty(), "wrong hidden dim rejected");

        // NaN input
        ncnn::Mat nan_in(dim, 4);
        std::vector<float> nan_flat(static_cast<std::size_t>(dim * 4), 0.f);
        nan_flat[0] = std::nanf("");
        std::memcpy(nan_in.data, nan_flat.data(), nan_flat.size() * sizeof(float));
        check(!runtime.swiglu_forward("vid", nan_in, out, err) && !err.empty(), "NaN input rejected");

        // last-layer runtime rejects txt SwiGLU
        const CoreInfo* ll = registry.lookup(CoreKey{4, 3, 3, 1, 1, 1, "synthetic_seedvr2_dit_block_c512_h4_hd128_last_layer"});
        if (ll)
        {
            NcnnModelCache ll_cache;
            HybridDiTBlockRuntime ll_runtime(ll_cache);
            if (ll_runtime.initialize(*ll, error))
            {
                ncnn::Mat in(dim, 2);
                std::vector<float> flat(static_cast<std::size_t>(dim * 2), 1.f);
                std::memcpy(in.data, flat.data(), flat.size() * sizeof(float));
                check(!ll_runtime.swiglu_forward("txt", in, out, err) && !err.empty(),
                      "last-layer txt SwiGLU rejected (vid_only)");
            }
            else check(false, "last-layer runtime init");
        }
        else check(false, "last-layer variant missing");
    }

    std::cout << "\n=== swiglu contract tests: " << (g_failures == 0 ? "PASS" : "FAIL")
              << " (" << g_checks << " checks, " << g_failures << " failures) ===\n";
    return g_failures == 0 ? 0 : 1;
}
