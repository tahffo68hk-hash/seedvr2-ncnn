#include "dynamic_rope3d.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace attention_dispatcher;
namespace fs = std::filesystem;

static DynamicAttentionContract make_contract(int wt, int wh, int ww)
{
    DynamicAttentionContract c{};
    c.batch = c.grid_t = c.grid_h = c.grid_w = 1;
    c.time = wt; c.height = wh; c.width = ww; c.channels = 32;
    c.window_t = wt; c.window_h = wh; c.window_w = ww;
    c.num_windows = 1; c.tokens_per_window = wt * wh * ww;
    c.heads = 4; c.head_dim = 8; c.rope_dim = 6;
    return c;
}

static std::vector<float> read_floats(const fs::path& path, size_t count)
{
    std::vector<float> values(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    assert(file.gcount() == static_cast<std::streamsize>(count * sizeof(float)));
    return values;
}

int main()
{
    Rope3DConfig config{};
    config.theta = 10000.0;
    config.axis_dims = {2, 2, 2};
    config.axis_scales = {1.0, 1.0 / 100.0, 1.0 / 10000.0};
    Rope3DOutput output{};
    std::string error;
    assert(generate_dynamic_rope3d(make_contract(4, 4, 4), config, output, error));
    assert(output.tokens == 64 && output.pairs == 4 && output.cos.size() == 256 && output.sin.size() == 256);
    for (int pair = 0; pair < 4; ++pair) { assert(output.cos[pair] == 1.f); assert(output.sin[pair] == 0.f); }
    const auto at = [&output](int token, int pair) { return output.sin[token * output.pairs + pair]; };
    assert(at(1, 2) != 0.f && at(1, 0) == 0.f && at(1, 1) == 0.f);
    assert(at(4, 1) != 0.f && at(4, 0) == 0.f && at(4, 2) == 0.f);
    assert(at(16, 0) != 0.f && at(16, 1) == 0.f && at(16, 2) == 0.f);
    assert(at(63, 0) != 0.f && at(63, 1) != 0.f && at(63, 2) != 0.f);

    Rope3DOutput rectangular{};
    assert(generate_dynamic_rope3d(make_contract(3, 4, 5), config, rectangular, error));
    assert(rectangular.tokens == 60 && rectangular.pairs == 4 && rectangular.cos.size() == 240);
    assert(rectangular.sin[(59 * 4) + 0] != 0.f && rectangular.sin[(59 * 4) + 1] != 0.f && rectangular.sin[(59 * 4) + 2] != 0.f);

    const fs::path reference = fs::path(__FILE__).parent_path().parent_path() / "seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_fidelity_rms_rope444_grid_1x2x2/reference";
    const auto expected_cos = read_floats(reference / "rope_cos.bin", 256);
    const auto expected_sin = read_floats(reference / "rope_sin.bin", 256);
    float cos_max = 0.f, sin_max = 0.f;
    for (size_t i = 0; i < 256; ++i) { cos_max = std::max(cos_max, std::fabs(output.cos[i] - expected_cos[i])); sin_max = std::max(sin_max, std::fabs(output.sin[i] - expected_sin[i])); }
    assert(cos_max < 1e-6f && sin_max < 1e-6f);

    Rope3DConfig invalid = config;
    invalid.axis_dims = {2, 2, 3};
    assert(!generate_dynamic_rope3d(make_contract(4, 4, 4), invalid, output, error));
    assert(output.cos.empty() && output.sin.empty() && error.find("axis_dims") != std::string::npos);
    invalid = config; invalid.axis_dims = {-2, 4, 4};
    assert(!generate_dynamic_rope3d(make_contract(4, 4, 4), invalid, output, error) && output.cos.empty());
    invalid = config; invalid.axis_dims = {2, 2, 4};
    assert(!generate_dynamic_rope3d(make_contract(4, 4, 4), invalid, output, error) && output.sin.empty());
    invalid = config; invalid.theta = 0.0;
    assert(!generate_dynamic_rope3d(make_contract(4, 4, 4), invalid, output, error) && output.cos.empty());
    invalid = config; invalid.theta = std::numeric_limits<double>::quiet_NaN();
    assert(!generate_dynamic_rope3d(make_contract(4, 4, 4), invalid, output, error) && output.sin.empty());
    invalid = config; invalid.axis_scales[1] = 0.0;
    assert(!generate_dynamic_rope3d(make_contract(4, 4, 4), invalid, output, error) && output.cos.empty());
    auto malformed = make_contract(4, 4, 4); malformed.window_t = 0;
    assert(!generate_dynamic_rope3d(malformed, config, output, error) && output.cos.empty());
    malformed = make_contract(4, 4, 4); malformed.rope_dim = 7;
    assert(!generate_dynamic_rope3d(malformed, config, output, error) && output.sin.empty());
    malformed = make_contract(4, 4, 4); malformed.rope_dim = 10;
    assert(!generate_dynamic_rope3d(malformed, config, output, error) && output.cos.empty());
    assert(generate_dynamic_rope3d(make_contract(1, 2, 3), config, output, error));

    // Phase 10: rope_dim=128 parity against the synthetic HD128-S reference.
    DynamicAttentionContract hd{};
    hd.batch = 1; hd.time = 4; hd.height = 4; hd.width = 4; hd.channels = 512;
    hd.window_t = 4; hd.window_h = 4; hd.window_w = 4;
    hd.grid_t = 1; hd.grid_h = 1; hd.grid_w = 1;
    hd.pad_t = 0; hd.pad_h = 0; hd.pad_w = 0;
    hd.num_windows = 1; hd.tokens_per_window = 64;
    hd.heads = 4; hd.head_dim = 128; hd.rope_dim = 128;
    Rope3DConfig hd_config{};
    hd_config.theta = 10000.0;
    hd_config.axis_dims = {44, 42, 42};
    hd_config.axis_scales = {1.0, 1.0 / 100.0, 1.0 / 10000.0};
    Rope3DOutput hd_output{};
    assert(generate_dynamic_rope3d(hd, hd_config, hd_output, error));
    assert(hd_output.tokens == 64 && hd_output.pairs == 64 && hd_output.cos.size() == 4096 && hd_output.sin.size() == 4096);
    const fs::path hd_reference = fs::path(__FILE__).parent_path().parent_path() / "seedvr2_attention_fidelity/artifacts/attention_core_fidelity/hybrid_synthetic_hd128_rope444_grid_1x1x1/reference";
    const auto hd_expected_cos = read_floats(hd_reference / "rope_cos.bin", 4096);
    const auto hd_expected_sin = read_floats(hd_reference / "rope_sin.bin", 4096);
    float hd_cos_max = 0.f, hd_sin_max = 0.f;
    for (size_t i = 0; i < 4096; ++i)
    {
        hd_cos_max = std::max(hd_cos_max, std::fabs(hd_output.cos[i] - hd_expected_cos[i]));
        hd_sin_max = std::max(hd_sin_max, std::fabs(hd_output.sin[i] - hd_expected_sin[i]));
    }
    assert(hd_cos_max < 1e-6f && hd_sin_max < 1e-6f);
    // All 64 pairs are active for rope_dim=128; token 0 (origin) is identity.
    assert(hd_output.sin[0] == 0.f && hd_output.sin[63] == 0.f);
    assert(hd_output.sin[1 * 64 + 43] != 0.f);   // token 1: W pair 43
    assert(hd_output.sin[16 * 64 + 0] != 0.f);   // token 16 (1,0,0): T pair 0
    assert(hd_output.sin[4 * 64 + 22] != 0.f);    // token 4 (0,1,0): H pair 22
    assert(hd_output.sin[63 * 64 + 63] != 0.f);  // token 63: all dims non-zero
    // Negative contract checks for the synthetic rope_dim=128 contract.
    DynamicAttentionContract hd_bad = hd; hd_bad.rope_dim = 256;
    assert(!generate_dynamic_rope3d(hd_bad, hd_config, output, error) && output.cos.empty());
    hd_bad = hd; hd_bad.rope_dim = 64;
    assert(!generate_dynamic_rope3d(hd_bad, hd_config, output, error) && output.sin.empty());
    Rope3DConfig hd_bad_config = hd_config; hd_bad_config.axis_dims = {44, 42, 44};
    assert(!generate_dynamic_rope3d(hd, hd_bad_config, output, error) && output.cos.empty());
    hd_bad_config = hd_config; hd_bad_config.axis_dims = {43, 43, 42};
    assert(!generate_dynamic_rope3d(hd, hd_bad_config, output, error) && output.sin.empty());
    hd_bad_config = hd_config; hd_bad_config.axis_scales[2] = 0.0;
    assert(!generate_dynamic_rope3d(hd, hd_bad_config, output, error) && output.cos.empty());

    // Phase 10.1: exact upstream rotary-embedding-torch 0.5.3 MM-RoPE3D oracle parity.
    // SeedVR2 commit e4de8c24441a67e1b7df56abea10645059bb1185 configures rope_dim=128 and
    // builds RotaryEmbedding(dim=128 // 3 = 42, freqs_for="lang", theta=10000); three axial
    // 42-dim blocks concatenate to effective_rotary_dim=126 with identity_tail_dim=2.
    struct OracleCase
    {
        const char* name;
        int wt, wh, ww;
    };
    const OracleCase oracle_cases[] = {{"a", 1, 1, 1}, {"b", 4, 4, 4}, {"c", 2, 3, 5}};
    const fs::path oracle_root = fs::path(__FILE__).parent_path().parent_path() / "seedvr2_attention_fidelity/artifacts/mmrope_oracle";
    float oracle_cos_max = 0.f, oracle_sin_max = 0.f, oracle_q_max = 0.f, oracle_k_max = 0.f;
    bool oracle_identity_ok = true;
    for (const OracleCase& c : oracle_cases)
    {
        const int tokens = c.wt * c.wh * c.ww;
        DynamicAttentionContract real{};
        real.batch = 1; real.time = c.wt; real.height = c.wh; real.width = c.ww; real.channels = 512;
        real.window_t = c.wt; real.window_h = c.wh; real.window_w = c.ww;
        real.grid_t = real.grid_h = real.grid_w = 1;
        real.num_windows = 1; real.tokens_per_window = tokens;
        real.heads = 4; real.head_dim = 128; real.rope_dim = 126;
        Rope3DConfig real_config{};
        real_config.theta = 10000.0;
        real_config.axis_dims = {42, 42, 42};
        real_config.axis_scales = {1.0, 1.0, 1.0};
        real_config.freq_mode = "lang_rope";
        Rope3DOutput real_output{};
        assert(generate_dynamic_rope3d(real, real_config, real_output, error));
        assert(real_output.tokens == tokens && real_output.pairs == 64 && real_output.cos.size() == size_t(tokens) * 64);

        const fs::path case_dir = oracle_root / c.name;
        const auto expected_cos = read_floats(case_dir / "cos.bin", size_t(tokens) * 64);
        const auto expected_sin = read_floats(case_dir / "sin.bin", size_t(tokens) * 64);
        for (size_t i = 0; i < real_output.cos.size(); ++i)
        {
            oracle_cos_max = std::max(oracle_cos_max, std::fabs(real_output.cos[i] - expected_cos[i]));
            oracle_sin_max = std::max(oracle_sin_max, std::fabs(real_output.sin[i] - expected_sin[i]));
        }
        // Identity tail: the last pair (dims 126..127) must be exactly cos=1/sin=0.
        for (int t = 0; t < tokens; ++t)
            oracle_identity_ok = oracle_identity_ok && real_output.cos[static_cast<size_t>(t) * 64 + 63] == 1.f &&
                                 real_output.sin[static_cast<size_t>(t) * 64 + 63] == 0.f;

        // Rotated Q/K parity: host interleaved-pair rotation with the generated cos/sin must
        // reproduce the upstream apply_rotary_emb output (partial rotary, identity tail).
        const auto q_in = read_floats(case_dir / "q_input.bin", size_t(tokens) * 4 * 128);
        const auto k_in = read_floats(case_dir / "k_input.bin", size_t(tokens) * 4 * 128);
        const auto q_ref = read_floats(case_dir / "q_rotated.bin", size_t(tokens) * 4 * 128);
        const auto k_ref = read_floats(case_dir / "k_rotated.bin", size_t(tokens) * 4 * 128);
        for (int token = 0; token < tokens; ++token)
        {
            for (int head = 0; head < 4; ++head)
            {
                const size_t row = (static_cast<size_t>(token) * 4 + head) * 128;
            for (int pair = 0; pair < 64; ++pair)
            {
                const size_t base = row + 2 * pair;
                const float c = real_output.cos[static_cast<size_t>(token) * 64 + pair];
                const float s = real_output.sin[static_cast<size_t>(token) * 64 + pair];
                const float qr0 = q_in[base] * c - q_in[base + 1] * s;
                const float qr1 = q_in[base] * s + q_in[base + 1] * c;
                const float kr0 = k_in[base] * c - k_in[base + 1] * s;
                const float kr1 = k_in[base] * s + k_in[base + 1] * c;
                oracle_q_max = std::max(oracle_q_max, std::fabs(qr0 - q_ref[base]));
                oracle_q_max = std::max(oracle_q_max, std::fabs(qr1 - q_ref[base + 1]));
                oracle_k_max = std::max(oracle_k_max, std::fabs(kr0 - k_ref[base]));
                oracle_k_max = std::max(oracle_k_max, std::fabs(kr1 - k_ref[base + 1]));
            }
            }
        }
        // Effective rope_dim / axis sums / odd pairs / freq_mode negative checks.
        Rope3DOutput bad_output{};
        DynamicAttentionContract bad_real = real; bad_real.rope_dim = 130;
        assert(!generate_dynamic_rope3d(bad_real, real_config, bad_output, error) && bad_output.cos.empty());
        bad_real = real; bad_real.rope_dim = 124;
        assert(!generate_dynamic_rope3d(bad_real, real_config, bad_output, error) && bad_output.sin.empty());
        Rope3DConfig bad_real_config = real_config; bad_real_config.axis_dims = {42, 42, 40};
        assert(!generate_dynamic_rope3d(real, bad_real_config, bad_output, error) && bad_output.cos.empty());
        bad_real_config = real_config; bad_real_config.axis_dims = {41, 43, 42};
        assert(!generate_dynamic_rope3d(real, bad_real_config, bad_output, error) && bad_output.sin.empty());
        bad_real_config = real_config; bad_real_config.freq_mode = "bogus";
        assert(!generate_dynamic_rope3d(real, bad_real_config, bad_output, error) && bad_output.cos.empty());
        bad_real_config = real_config; bad_real_config.freq_mode = "lang_rope"; bad_real_config.theta = 0.0;
        assert(!generate_dynamic_rope3d(real, bad_real_config, bad_output, error) && bad_output.sin.empty());
    }
    assert(oracle_cos_max < 1e-6f && oracle_sin_max < 1e-6f);
    assert(oracle_q_max < 1e-6f && oracle_k_max < 1e-6f);
    assert(oracle_identity_ok);

    std::cout << "dynamic_rope3d_tests=PASS cos_max=" << cos_max << " sin_max=" << sin_max
              << " hd128_cos_max=" << hd_cos_max << " hd128_sin_max=" << hd_sin_max
              << " oracle_cos_max=" << oracle_cos_max << " oracle_sin_max=" << oracle_sin_max
              << " oracle_q_max=" << oracle_q_max << " oracle_k_max=" << oracle_k_max << '\n';
}
