#include "dynamic_attention_inputs.h"

#include <limits>

namespace attention_dispatcher
{
namespace
{
bool checked_multiply(std::size_t left, std::size_t right, std::size_t& result)
{
    if (left && right > std::numeric_limits<std::size_t>::max() / left) return false;
    result = left * right;
    return true;
}
}

bool validate_dynamic_attention_contract(const DynamicAttentionContract& c, std::string& error)
{
    const int* dimensions[] = {&c.batch, &c.time, &c.height, &c.width, &c.channels, &c.window_t, &c.window_h, &c.window_w,
                               &c.grid_t, &c.grid_h, &c.grid_w, &c.num_windows, &c.tokens_per_window, &c.heads, &c.head_dim, &c.rope_dim};
    for (const int* value : dimensions) if (*value <= 0) { error = "dynamic contract dimensions must be positive"; return false; }
    if (c.pad_t < 0 || c.pad_h < 0 || c.pad_w < 0) { error = "dynamic contract padding must be non-negative"; return false; }
    if (c.grid_t * c.window_t != c.time + c.pad_t || c.grid_h * c.window_h != c.height + c.pad_h || c.grid_w * c.window_w != c.width + c.pad_w) { error = "dynamic contract padded shape mismatch"; return false; }
    if (c.tokens_per_window != c.window_t * c.window_h * c.window_w) { error = "dynamic contract token count mismatch"; return false; }
    if (c.num_windows != c.batch * c.grid_t * c.grid_h * c.grid_w) { error = "dynamic contract window count mismatch"; return false; }
    if (c.channels != c.heads * c.head_dim) { error = "dynamic contract channels must equal heads*head_dim"; return false; }
    if (c.rope_dim > c.head_dim || c.rope_dim % 2) { error = "dynamic contract rope_dim must be even and <= head_dim"; return false; }
    std::size_t ignored = 0;
    if (!checked_dynamic_element_count(c, ignored, error)) return false;
    error.clear();
    return true;
}

bool checked_dynamic_element_count(const DynamicAttentionContract& c, std::size_t& count, std::string& error)
{
    count = 1;
    for (int value : {c.batch, c.grid_t, c.grid_h, c.grid_w, c.window_t, c.window_h, c.window_w, c.channels})
        if (value <= 0 || !checked_multiply(count, static_cast<std::size_t>(value), count)) { error = "dynamic contract element count overflow"; return false; }
    error.clear();
    return true;
}
}
