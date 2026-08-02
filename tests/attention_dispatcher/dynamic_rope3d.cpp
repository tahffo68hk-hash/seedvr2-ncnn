#include "dynamic_rope3d.h"

#include <cmath>
#include <limits>

namespace attention_dispatcher
{
namespace
{
void clear(Rope3DOutput& output)
{
    output.tokens = 0;
    output.pairs = 0;
    output.cos.clear();
    output.sin.clear();
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t& result)
{
    if (left && right > std::numeric_limits<std::size_t>::max() / left) return false;
    result = left * right;
    return true;
}
}

bool generate_dynamic_rope3d(const DynamicAttentionContract& c, const Rope3DConfig& config, Rope3DOutput& output, std::string& error)
{
    clear(output);
    if (!validate_dynamic_attention_contract(c, error)) return false;
    if (!std::isfinite(config.theta) || config.theta <= 0.0) { error = "theta must be finite and > 0"; return false; }
    int axis_total = 0;
    for (int axis_dim : config.axis_dims)
    {
        if (axis_dim < 0 || axis_dim % 2) { error = "axis_dims must contain non-negative even real dimensions"; return false; }
        if (axis_dim > std::numeric_limits<int>::max() - axis_total) { error = "axis_dims sum overflow"; return false; }
        axis_total += axis_dim;
    }
    if (axis_total != c.rope_dim) { error = "axis_dims sum must equal rope_dim"; return false; }
    for (double axis_scale : config.axis_scales)
        if (!std::isfinite(axis_scale) || axis_scale <= 0.0) { error = "axis_scales must be finite and > 0"; return false; }
    const bool lang_rope = config.freq_mode == "lang_rope";
    if (config.freq_mode != "global_pairs" && !lang_rope) { error = "freq_mode must be global_pairs or lang_rope"; return false; }
    const int pairs = c.head_dim / 2;
    const int active_pairs = c.rope_dim / 2;
    std::size_t element_count = 0;
    if (!checked_multiply(static_cast<std::size_t>(c.tokens_per_window), static_cast<std::size_t>(pairs), element_count)) { error = "RoPE output element count overflow"; return false; }

    output.tokens = c.tokens_per_window;
    output.pairs = pairs;
    output.cos.assign(element_count, 1.f);
    output.sin.assign(element_count, 0.f);
    int pair_offset = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const int axis_pairs = config.axis_dims[axis] / 2;
        for (int local_pair = 0; local_pair < axis_pairs; ++local_pair)
        {
            const int pair = pair_offset + local_pair;
            const float inv_freq = lang_rope
                ? std::exp(-std::log(static_cast<float>(config.theta)) * static_cast<float>(local_pair) / static_cast<float>(axis_pairs))
                : std::exp(-std::log(static_cast<float>(config.theta)) * static_cast<float>(pair) / static_cast<float>(active_pairs));
            for (int lt = 0; lt < c.window_t; ++lt) for (int lh = 0; lh < c.window_h; ++lh) for (int lw = 0; lw < c.window_w; ++lw)
            {
                const int coordinate = axis == 0 ? lt : (axis == 1 ? lh : lw);
                const int token = ((lt * c.window_h) + lh) * c.window_w + lw;
                const float angle = static_cast<float>(coordinate) * static_cast<float>(config.axis_scales[axis]) * inv_freq;
                output.cos[static_cast<std::size_t>(token) * pairs + pair] = std::cos(angle);
                output.sin[static_cast<std::size_t>(token) * pairs + pair] = std::sin(angle);
            }
        }
        pair_offset += axis_pairs;
    }
    error.clear();
    return true;
}
}
