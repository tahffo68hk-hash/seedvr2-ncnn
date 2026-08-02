// Phase 12: AdaSingle modulation contract tests (real SeedVR2 AdaSingle math).
//
// Pure-function tests cover the exact math, slice order, broadcasting, shape
// and finiteness guards.  Runtime-level tests (against the real artifact) prove
// the six-slice [attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale,
// mlp_gate] order, attn/mlp branch selection, gate position (no sigmoid/tanh),
// vid/txt learned-parameter separation and batch/emb mismatch errors.
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
using seedvr2_dit_block::DiTBlockInput;
using seedvr2_dit_block::DiTBlockOutput;
using seedvr2_dit_block::DiTBlockRunStats;
using seedvr2_dit_block::DiTBlockTrace;
using seedvr2_dit_block::HybridDiTBlockRuntime;
using seedvr2_dit_block::VideoShape;

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

std::vector<float> linspace(int n, float start, float step)
{
    std::vector<float> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = start + step * static_cast<float>(i);
    return v;
}

float maxdiff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return INFINITY;
    float m = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
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
} // namespace

int main(int argc, char** argv)
{
    const fs::path artifacts_root = argc > 1 ? fs::path(argv[1]) : fs::path("tests/seedvr2_dit_block_ncnn/artifacts");
    const int dim = 512;
    const int six_c = 6 * dim;

    std::cout << "== AdaSingle pure-function math ==\n";
    {
        const std::vector<int> counts{2, 4};
        const std::vector<float> hid = linspace(6 * 512, 0.5f, 0.1f);
        // emb_shift / emb_scale / emb_gate [B, dim]: rows b0 = all 1, b1 = all 2
        std::vector<float> emb_shift(static_cast<std::size_t>(2 * dim));
        std::vector<float> emb_scale(static_cast<std::size_t>(2 * dim));
        std::vector<float> emb_gate(static_cast<std::size_t>(2 * dim));
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < dim; ++c)
            {
                emb_shift[static_cast<std::size_t>(b * dim + c)] = static_cast<float>(b + 1);
                emb_scale[static_cast<std::size_t>(b * dim + c)] = static_cast<float>(b + 1);
                emb_gate[static_cast<std::size_t>(b * dim + c)] = static_cast<float>(b + 1);
            }
        const std::vector<float> shift_b = linspace(dim, 0.25f, 0.01f);
        const std::vector<float> scale_b = linspace(dim, 0.5f, 0.02f);
        const std::vector<float> gate_b = linspace(dim, 0.75f, 0.03f);

        std::vector<float> out;
        std::string error;
        check(seedvr2_dit_block::ada_single_in(hid, dim, emb_shift, emb_scale, counts, shift_b, scale_b, out, error),
              "ada_single_in ok");
        std::vector<float> expect_in;
        std::size_t token = 0;
        for (int b = 0; b < 2; ++b)
            for (int t = 0; t < counts[static_cast<std::size_t>(b)]; ++t, ++token)
                for (int c = 0; c < dim; ++c)
                    expect_in.push_back(hid[token * static_cast<std::size_t>(dim) + c] *
                                            (emb_scale[static_cast<std::size_t>(b * dim + c)] + scale_b[static_cast<std::size_t>(c)]) +
                                        (emb_shift[static_cast<std::size_t>(b * dim + c)] + shift_b[static_cast<std::size_t>(c)]));
        check(maxdiff(out, expect_in) < 1e-6f, "ada_single_in math (no 1+scale) matches");

        check(seedvr2_dit_block::ada_single_gate(hid, dim, emb_gate, counts, gate_b, out, error),
              "ada_single_gate ok");
        std::vector<float> expect_gate;
        token = 0;
        for (int b = 0; b < 2; ++b)
            for (int t = 0; t < counts[static_cast<std::size_t>(b)]; ++t, ++token)
                for (int c = 0; c < dim; ++c)
                    expect_gate.push_back(hid[token * static_cast<std::size_t>(dim) + c] *
                                          (emb_gate[static_cast<std::size_t>(b * dim + c)] + gate_b[static_cast<std::size_t>(c)]));
        check(maxdiff(out, expect_gate) < 1e-6f, "ada_single_gate is linear hid*(gate+gate_b) (no sigmoid/tanh)");

        // wrong shapes
        const std::vector<float> bad_hid(static_cast<std::size_t>(dim * 3) + 1, 0.f);
        const std::vector<float> bad_emb(static_cast<std::size_t>(dim * 2) - 1, 0.f);
        const std::vector<float> bad_param(static_cast<std::size_t>(dim) - 1, 0.f);
        check(!seedvr2_dit_block::ada_single_in(bad_hid, dim, emb_shift, emb_scale, counts, shift_b, scale_b, out, error) && !error.empty(),
              "ada_single_in wrong hid token*dim rejected");
        check(!seedvr2_dit_block::ada_single_in(hid, dim, bad_emb, emb_scale, counts, shift_b, scale_b, out, error) && !error.empty(),
              "ada_single_in wrong emb [B,dim] rejected");
        check(!seedvr2_dit_block::ada_single_in(hid, dim, emb_shift, emb_scale, counts, bad_param, scale_b, out, error) && !error.empty(),
              "ada_single_in wrong learned param size rejected");
        check(!seedvr2_dit_block::ada_single_gate(hid, dim, emb_gate, counts, bad_param, out, error) && !error.empty(),
              "ada_single_gate wrong gate param size rejected");
        check(!seedvr2_dit_block::ada_single_in(hid, dim, emb_shift, emb_scale, std::vector<int>{2, -3}, shift_b, scale_b, out, error) && !error.empty(),
              "negative token count rejected");

        // NaN / Inf
        std::vector<float> nan_emb = emb_shift;
        nan_emb[0] = std::nanf("");
        check(!seedvr2_dit_block::ada_single_in(hid, dim, nan_emb, emb_scale, counts, shift_b, scale_b, out, error) && !error.empty(),
              "ada_single_in emb NaN rejected");
        std::vector<float> inf_hid = hid;
        inf_hid[1] = INFINITY;
        check(!seedvr2_dit_block::ada_single_in(inf_hid, dim, emb_shift, emb_scale, counts, shift_b, scale_b, out, error) && !error.empty(),
              "ada_single_in hid Inf rejected");
        std::vector<float> nan_gate_emb = emb_gate;
        nan_gate_emb[static_cast<std::size_t>(dim) + 5] = std::nanf("");
        check(!seedvr2_dit_block::ada_single_gate(hid, dim, nan_gate_emb, counts, gate_b, out, error) && !error.empty(),
              "ada_single_gate emb NaN rejected");
    }

    // Runtime-level contract checks against the real artifact.
    std::cout << "== AdaSingle runtime contract (real artifact) ==\n";
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
    const DiTBlockArtifact* art = base->dit_block();
    const fs::path params_dir = art->params_dir;
    auto load_param = [&](const std::string& name) {
        std::vector<float> v;
        read_bin(params_dir / name, v);
        return v;
    };
    const std::vector<float> vid_attn_shift = load_param("ada_vid_attn_shift.bin");
    const std::vector<float> vid_attn_scale = load_param("ada_vid_attn_scale.bin");
    const std::vector<float> vid_attn_gate = load_param("ada_vid_attn_gate.bin");
    const std::vector<float> vid_mlp_shift = load_param("ada_vid_mlp_shift.bin");
    const std::vector<float> vid_mlp_scale = load_param("ada_vid_mlp_scale.bin");
    const std::vector<float> vid_mlp_gate = load_param("ada_vid_mlp_gate.bin");
    const std::vector<float> txt_attn_shift = load_param("ada_txt_attn_shift.bin");
    const std::vector<float> txt_attn_scale = load_param("ada_txt_attn_scale.bin");
    const std::vector<float> txt_attn_gate = load_param("ada_txt_attn_gate.bin");
    check(vid_attn_shift.size() == static_cast<std::size_t>(dim) && txt_attn_shift.size() == static_cast<std::size_t>(dim),
          "learned ada params loaded");

    // Load profile A inputs (16 vid tokens, 2 txt tokens, batch 1).
    const fs::path ref_a = base->directory / "reference" / "A";
    std::vector<float> input_vid, input_txt;
    std::vector<int> txt_lengths{2};
    std::vector<VideoShape> vid_shapes{{1, 4, 4}};
    if (!read_bin(ref_a / "input_vid.bin", input_vid) || !read_bin(ref_a / "input_txt.bin", input_txt))
    {
        std::cerr << "profile A inputs missing\n";
        return 2;
    }

    NcnnModelCache cache;
    HybridDiTBlockRuntime runtime(cache);
    if (!runtime.initialize(*base, error))
    {
        std::cerr << "runtime init failed: " << error << '\n';
        return 2;
    }

    const auto run_with_emb = [&](const std::vector<float>& emb_flat, DiTBlockTrace& trace) {
        DiTBlockInput in;
        in.vid = ncnn::Mat(dim, static_cast<int>(input_vid.size() / dim));
        std::memcpy(in.vid.data, input_vid.data(), input_vid.size() * sizeof(float));
        in.txt = ncnn::Mat(dim, static_cast<int>(input_txt.size() / dim));
        std::memcpy(in.txt.data, input_txt.data(), input_txt.size() * sizeof(float));
        in.emb = ncnn::Mat(six_c, static_cast<int>(emb_flat.size() / six_c));
        std::memcpy(in.emb.data, emb_flat.data(), emb_flat.size() * sizeof(float));
        in.vid_shapes = vid_shapes;
        in.txt_lengths = txt_lengths;
        DiTBlockOutput out;
        DiTBlockRunStats stats;
        const bool ok = runtime.run(in, out, &trace, stats);
        if (!ok) std::cerr << "  crafted-emb run error: " << stats.error_message << '\n';
        return ok;
    };

    // Crafted emb in the REAL interleaved layout (rearrange "b (d l g) -> b d l g",
    // l=2, g=3): per dim c the six values are
    // [attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate], each
    // varying with c so the per-dim interleaving is unambiguously verified:
    //   value(g, c) = (g + 1) * 0.1 + c * 0.001
    std::vector<float> emb(six_c, 0.f);
    for (int c = 0; c < dim; ++c)
        for (int g = 0; g < 6; ++g)
            emb[static_cast<std::size_t>(6 * c + g)] = static_cast<float>((g + 1) * 0.1 + c * 0.001);
    DiTBlockTrace trace;
    check(run_with_emb(emb, trace), "runtime forward with crafted emb ok");

    {
        const auto attn_shift_c = [&](std::size_t c) { return 0.1f + static_cast<float>(c) * 0.001f; };
        const auto attn_scale_c = [&](std::size_t c) { return 0.2f + static_cast<float>(c) * 0.001f; };
        const auto attn_gate_c = [&](std::size_t c) { return 0.3f + static_cast<float>(c) * 0.001f; };
        const auto mlp_shift_c = [&](std::size_t c) { return 0.4f + static_cast<float>(c) * 0.001f; };
        const auto mlp_scale_c = [&](std::size_t c) { return 0.5f + static_cast<float>(c) * 0.001f; };

        // attn in: hid*(attn_scale + scale_b) + (attn_shift + shift_b), per dim
        std::vector<float> expect(trace.attn_norm_vid.size());
        for (std::size_t i = 0; i < expect.size(); ++i)
        {
            const std::size_t c = i % static_cast<std::size_t>(dim);
            expect[i] = trace.attn_norm_vid[i] * (attn_scale_c(c) + vid_attn_scale[c]) + (attn_shift_c(c) + vid_attn_shift[c]);
        }
        check(maxdiff(trace.ada_attn_in_vid, expect) < 1e-5f, "interleaved layout: attn_in uses (shift,scale) per dim");

        // attn gate: out*(attn_gate + gate_b), per dim
        std::vector<float> expect_gate(trace.attention_proj_vid.size());
        for (std::size_t i = 0; i < expect_gate.size(); ++i)
        {
            const std::size_t c = i % static_cast<std::size_t>(dim);
            expect_gate[i] = trace.attention_proj_vid[i] * (attn_gate_c(c) + vid_attn_gate[c]);
        }
        check(maxdiff(trace.ada_attn_gate_vid, expect_gate) < 1e-4f, "attn gate applied after attention projection (no sigmoid)");

        // mlp branch: per dim mlp shift/scale (values 4/5, distinct from attn)
        std::vector<float> expect_mlp(trace.mlp_norm_vid.size());
        for (std::size_t i = 0; i < expect_mlp.size(); ++i)
        {
            const std::size_t c = i % static_cast<std::size_t>(dim);
            expect_mlp[i] = trace.mlp_norm_vid[i] * (mlp_scale_c(c) + vid_mlp_scale[c]) + (mlp_shift_c(c) + vid_mlp_shift[c]);
        }
        check(maxdiff(trace.ada_mlp_in_vid, expect_mlp) < 1e-5f, "mlp branch uses per-dim (shift,scale)");

        // txt branch: same emb values but txt learned params
        std::vector<float> expect_txt(trace.attn_norm_txt.size());
        for (std::size_t i = 0; i < expect_txt.size(); ++i)
        {
            const std::size_t c = i % static_cast<std::size_t>(dim);
            expect_txt[i] = trace.attn_norm_txt[i] * (attn_scale_c(c) + txt_attn_scale[c]) + (attn_shift_c(c) + txt_attn_shift[c]);
        }
        check(maxdiff(trace.ada_attn_in_txt, expect_txt) < 1e-5f, "txt branch uses txt learned params");
    }

    std::cout << "== AdaSingle runtime negative tests ==\n";
    {
        DiTBlockTrace t;
        // batch mismatch: 2 vid shapes, 1 txt length
        DiTBlockInput in;
        in.vid = ncnn::Mat(dim, static_cast<int>(input_vid.size() / dim));
        std::memcpy(in.vid.data, input_vid.data(), input_vid.size() * sizeof(float));
        in.txt = ncnn::Mat(dim, static_cast<int>(input_txt.size() / dim));
        std::memcpy(in.txt.data, input_txt.data(), input_txt.size() * sizeof(float));
        in.emb = ncnn::Mat(six_c, 1);
        std::memcpy(in.emb.data, emb.data(), emb.size() * sizeof(float));
        in.vid_shapes = {{1, 4, 4}, {1, 4, 4}};
        in.txt_lengths = {2};
        DiTBlockOutput out;
        DiTBlockRunStats stats;
        check(!runtime.run(in, out, &t, stats) && !stats.error_message.empty(), "batch mismatch rejected");

        // emb wrong dimension
        in.vid_shapes = {{1, 4, 4}};
        in.txt_lengths = {2};
        in.emb = ncnn::Mat(six_c - 1, 1);
        std::memcpy(in.emb.data, emb.data(), (emb.size() - 1) * sizeof(float));
        check(!runtime.run(in, out, &t, stats) && !stats.error_message.empty(), "emb not [B,6*C] rejected");

        // emb NaN
        std::vector<float> nan_emb = emb;
        nan_emb[0] = std::nanf("");
        in.emb = ncnn::Mat(six_c, 1);
        std::memcpy(in.emb.data, nan_emb.data(), nan_emb.size() * sizeof(float));
        check(!runtime.run(in, out, &t, stats) && !stats.error_message.empty(), "emb NaN rejected");

        // vid NaN
        std::vector<float> nan_vid = input_vid;
        nan_vid[0] = std::nanf("");
        in.emb = ncnn::Mat(six_c, 1);
        std::memcpy(in.emb.data, emb.data(), emb.size() * sizeof(float));
        in.vid = ncnn::Mat(dim, static_cast<int>(nan_vid.size() / dim));
        std::memcpy(in.vid.data, nan_vid.data(), nan_vid.size() * sizeof(float));
        check(!runtime.run(in, out, &t, stats) && !stats.error_message.empty(), "vid NaN rejected");

        // txt NaN
        std::vector<float> nan_txt = input_txt;
        nan_txt[0] = std::nanf("");
        in.vid = ncnn::Mat(dim, static_cast<int>(input_vid.size() / dim));
        std::memcpy(in.vid.data, input_vid.data(), input_vid.size() * sizeof(float));
        in.txt = ncnn::Mat(dim, static_cast<int>(nan_txt.size() / dim));
        std::memcpy(in.txt.data, nan_txt.data(), nan_txt.size() * sizeof(float));
        check(!runtime.run(in, out, &t, stats) && !stats.error_message.empty(), "txt NaN rejected");

        // empty txt: zero-length txt_lengths
        in.txt = ncnn::Mat(dim, 0);
        in.txt_lengths = {0};
        check(!runtime.run(in, out, &t, stats) && !stats.error_message.empty(), "empty txt rejected");
    }

    std::cout << "\n=== ada_single contract tests: " << (g_failures == 0 ? "PASS" : "FAIL")
              << " (" << g_checks << " checks, " << g_failures << " failures) ===\n";
    return g_failures == 0 ? 0 : 1;
}
