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
    std::cout << "dynamic_rope3d_tests=PASS cos_max=" << cos_max << " sin_max=" << sin_max << '\n';
}
