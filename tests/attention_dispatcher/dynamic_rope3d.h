#pragma once

#include "dynamic_attention_inputs.h"

#include <array>
#include <string>
#include <vector>

namespace attention_dispatcher
{
struct Rope3DConfig
{
    double theta = 10000.0;
    std::array<int, 3> axis_dims{};
    std::array<double, 3> axis_scales{};
};

struct Rope3DOutput
{
    int tokens = 0;
    int pairs = 0;
    std::vector<float> cos;
    std::vector<float> sin;
};

bool generate_dynamic_rope3d(const DynamicAttentionContract& contract, const Rope3DConfig& config, Rope3DOutput& output, std::string& error_message);
}
