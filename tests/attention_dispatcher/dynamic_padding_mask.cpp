#include "dynamic_padding_mask.h"

#include <limits>

namespace attention_dispatcher
{
bool generate_padding_validity(const DynamicAttentionContract& c, std::vector<unsigned char>& validity, std::string& error)
{
    validity.clear();
    if (!validate_dynamic_attention_contract(c, error)) return false;
    const std::size_t count = static_cast<std::size_t>(c.num_windows) * c.tokens_per_window;
    if (c.num_windows > 0 && count / static_cast<std::size_t>(c.num_windows) != static_cast<std::size_t>(c.tokens_per_window)) { error = "validity element count overflow"; return false; }
    validity.assign(count, 0);
    for (int b = 0; b < c.batch; ++b) for (int gt = 0; gt < c.grid_t; ++gt) for (int gh = 0; gh < c.grid_h; ++gh) for (int gw = 0; gw < c.grid_w; ++gw)
    {
        const int window = ((b * c.grid_t + gt) * c.grid_h + gh) * c.grid_w + gw;
        for (int lt = 0; lt < c.window_t; ++lt) for (int lh = 0; lh < c.window_h; ++lh) for (int lw = 0; lw < c.window_w; ++lw)
        {
            const int token = ((lt * c.window_h) + lh) * c.window_w + lw;
            validity[static_cast<std::size_t>(window) * c.tokens_per_window + token] = gt * c.window_t + lt < c.time && gh * c.window_h + lh < c.height && gw * c.window_w + lw < c.width;
        }
    }
    error.clear();
    return true;
}

int count_valid_tokens(const std::vector<unsigned char>& validity, int window, int tokens_per_window)
{
    if (window < 0 || tokens_per_window <= 0 || static_cast<std::size_t>(window + 1) * tokens_per_window > validity.size()) return -1;
    int count = 0;
    for (int token = 0; token < tokens_per_window; ++token) count += validity[static_cast<std::size_t>(window) * tokens_per_window + token] != 0;
    return count;
}

bool generate_additive_key_mask(const std::vector<unsigned char>& validity, int windows, int tokens, std::vector<float>& mask, std::string& error)
{
    mask.clear();
    if (windows <= 0 || tokens <= 0 || validity.size() != static_cast<std::size_t>(windows) * tokens) { error = "validity shape mismatch"; return false; }
    if (static_cast<std::size_t>(windows) > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(tokens) / static_cast<std::size_t>(tokens)) { error = "mask element count overflow"; return false; }
    mask.assign(static_cast<std::size_t>(windows) * tokens * tokens, 0.f);
    for (int window = 0; window < windows; ++window) for (int query = 0; query < tokens; ++query) for (int key = 0; key < tokens; ++key)
        if (!validity[static_cast<std::size_t>(window) * tokens + key]) mask[(static_cast<std::size_t>(window) * tokens + query) * tokens + key] = -10000.f;
    error.clear();
    return true;
}
}
