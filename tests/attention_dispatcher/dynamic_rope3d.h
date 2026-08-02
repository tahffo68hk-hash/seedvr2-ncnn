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
    // "global_pairs": legacy contract, inv_freq = theta^(-pair/active_pairs) over the whole rope.
    // "lang_rope": rotary-embedding-torch 0.5.3 RotaryEmbedding(dim=axis_dims[a], freqs_for="lang")
    // per axis: inv_freq(local k) = theta^(-2k/axis_dims[a]); pairs beyond rope_dim stay identity.
    std::string freq_mode = "global_pairs";
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
