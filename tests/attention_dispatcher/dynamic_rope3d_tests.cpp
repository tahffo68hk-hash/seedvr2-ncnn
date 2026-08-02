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

    std::cout << "dynamic_rope3d_tests=PASS cos_max=" << cos_max << " sin_max=" << sin_max
              << " hd128_cos_max=" << hd_cos_max << " hd128_sin_max=" << hd_sin_max << '\n';
}
