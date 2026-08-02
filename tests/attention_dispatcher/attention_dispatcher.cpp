#include "attention_dispatcher.h"

#include <stdexcept>
#include <sstream>

namespace attention_dispatcher
{
namespace
{
int ceil_div(int value, int divisor) { return (value + divisor - 1) / divisor; }

std::string core_name(const Metadata& m)
{
    return "core_" + std::to_string(m.window_t) + std::to_string(m.window_h) + std::to_string(m.window_w)
        + "_grid_" + std::to_string(m.grid_t) + "x" + std::to_string(m.grid_h) + "x" + std::to_string(m.grid_w);
}

std::size_t bthwc_offset(const Shape& s, int b, int t, int h, int w, int c)
{
    return (((((static_cast<std::size_t>(b) * s.time + t) * s.height + h) * s.width + w) * s.channels) + c);
}
}

Metadata dispatch_shape(Shape shape)
{
    if (shape.batch <= 0 || shape.time <= 0 || shape.height <= 0 || shape.width <= 0 || shape.channels != 32)
        throw std::invalid_argument("expected positive BTHW dimensions and C=32");
    Metadata m{};
    m.input = shape;
    m.window_t = shape.time < 4 ? shape.time : 4;
    m.window_h = 4;
    m.window_w = 4;
    m.grid_t = ceil_div(shape.time, m.window_t);
    m.grid_h = ceil_div(shape.height, m.window_h);
    m.grid_w = ceil_div(shape.width, m.window_w);
    m.pad_t = m.grid_t * m.window_t - shape.time;
    m.pad_h = m.grid_h * m.window_h - shape.height;
    m.pad_w = m.grid_w * m.window_w - shape.width;
    m.num_windows = shape.batch * m.grid_t * m.grid_h * m.grid_w;
    m.tokens_per_window = m.window_t * m.window_h * m.window_w;
    return m;
}

CoreKey make_core_key(const Metadata& m) noexcept
{
    return {m.window_t, m.window_h, m.window_w, m.grid_t, m.grid_h, m.grid_w, "baseline"};
}

bool validate_attention_request(const AttentionRequest& request, std::string& error_message)
{
    if (request.batch != 1) { error_message = "request batch must be 1"; return false; }
    if (request.channels != 32) { error_message = "request channels must be 32"; return false; }
    if (request.dtype != "fp32") { error_message = "request dtype must be fp32"; return false; }
    if (request.variant.empty()) { error_message = "request variant must be non-empty"; return false; }
    if (request.time <= 0 || request.height <= 0 || request.width <= 0) { error_message = "request time, height, and width must be positive"; return false; }
    error_message.clear(); return true;
}

bool dispatch_request(const AttentionRequest& request, const ArtifactRegistry& registry,
                      Metadata& metadata, const CoreInfo*& core, std::string& error_message)
{
    core = nullptr;
    if (!validate_attention_request(request, error_message)) return false;
    try { metadata = dispatch_shape({request.batch, request.time, request.height, request.width, request.channels}); }
    catch (const std::exception& error) { error_message = error.what(); return false; }
    CoreKey key = make_core_key(metadata);
    key.variant = request.variant;
    core = registry.lookup(key);
    if (core) { error_message.clear(); return true; }
    std::ostringstream message;
    message << "no core for key window=" << key.window_t << 'x' << key.window_h << 'x' << key.window_w
            << " grid=" << key.grid_t << 'x' << key.grid_h << 'x' << key.grid_w << "; available:";
    for (const CoreInfo& item : registry.list())
        message << " [window=" << item.key.window_t << 'x' << item.key.window_h << 'x' << item.key.window_w
                << " grid=" << item.key.grid_t << 'x' << item.key.grid_h << 'x' << item.key.grid_w << ']';
    error_message = message.str();
    return false;
}

CoreRegistry::CoreRegistry(std::vector<CoreEntry> entries) : entries_(std::move(entries)) {}

const CoreEntry& CoreRegistry::find(const Metadata& metadata) const
{
    const std::string wanted = core_name(metadata);
    for (const CoreEntry& entry : entries_)
        if (entry.name == wanted)
            return entry;
    throw std::runtime_error("no static core registered for " + wanted);
}

CoreRegistry make_default_registry()
{
    return CoreRegistry({
        {"core_444_grid_1x4x4", true},
        {"core_344_grid_1x4x4", true},
        {"core_444_grid_1x5x5", true},
        {"core_444_grid_2x8x8", true},
        {"core_444_grid_2x5x6", true},
    });
}

CoreEntry select_default_core(const Metadata& metadata)
{
    CoreRegistry registry = make_default_registry();
    return registry.find(metadata);
}

std::vector<float> make_additive_key_mask(const Metadata& m)
{
    const int n = m.tokens_per_window;
    std::vector<float> mask(static_cast<std::size_t>(m.num_windows) * n * n, 0.f);
    for (int window = 0; window < m.num_windows; ++window)
    {
        const int local = window % (m.grid_t * m.grid_h * m.grid_w);
        const int gt = local / (m.grid_h * m.grid_w);
        const int gh = (local / m.grid_w) % m.grid_h;
        const int gw = local % m.grid_w;
        for (int key = 0; key < n; ++key)
        {
            const int kt = key / (m.window_h * m.window_w);
            const int kh = (key / m.window_w) % m.window_h;
            const int kw = key % m.window_w;
            const bool valid = gt * m.window_t + kt < m.input.time && gh * m.window_h + kh < m.input.height && gw * m.window_w + kw < m.input.width;
            if (!valid)
                for (int query = 0; query < n; ++query)
                    mask[(static_cast<std::size_t>(window) * n + query) * n + key] = -10000.f;
        }
    }
    return mask;
}

std::vector<float> partition_bthwc(const std::vector<float>& input, const Metadata& m)
{
    const Shape& s = m.input;
    if (input.size() != static_cast<std::size_t>(s.batch) * s.time * s.height * s.width * s.channels)
        throw std::invalid_argument("input size does not match BTHWC metadata");
    std::vector<float> windows(static_cast<std::size_t>(m.num_windows) * m.tokens_per_window * s.channels, 0.f);
    for (int b = 0; b < s.batch; ++b)
        for (int gt = 0; gt < m.grid_t; ++gt)
            for (int gh = 0; gh < m.grid_h; ++gh)
                for (int gw = 0; gw < m.grid_w; ++gw)
                {
                    const int window = ((b * m.grid_t + gt) * m.grid_h + gh) * m.grid_w + gw;
                    for (int token = 0; token < m.tokens_per_window; ++token)
                    {
                        const int lt = token / (m.window_h * m.window_w);
                        const int lh = (token / m.window_w) % m.window_h;
                        const int lw = token % m.window_w;
                        const int t = gt * m.window_t + lt, h = gh * m.window_h + lh, w = gw * m.window_w + lw;
                        if (t < s.time && h < s.height && w < s.width)
                            for (int c = 0; c < s.channels; ++c)
                                windows[(static_cast<std::size_t>(window) * m.tokens_per_window + token) * s.channels + c] = input[bthwc_offset(s, b, t, h, w, c)];
                    }
                }
    return windows;
}

std::vector<float> reverse_crop_bthwc(const std::vector<float>& windows, const Metadata& m)
{
    const Shape& s = m.input;
    if (windows.size() != static_cast<std::size_t>(m.num_windows) * m.tokens_per_window * s.channels)
        throw std::invalid_argument("window size does not match metadata");
    std::vector<float> output(static_cast<std::size_t>(s.batch) * s.time * s.height * s.width * s.channels);
    for (int b = 0; b < s.batch; ++b)
        for (int gt = 0; gt < m.grid_t; ++gt)
            for (int gh = 0; gh < m.grid_h; ++gh)
                for (int gw = 0; gw < m.grid_w; ++gw)
                {
                    const int window = ((b * m.grid_t + gt) * m.grid_h + gh) * m.grid_w + gw;
                    for (int token = 0; token < m.tokens_per_window; ++token)
                    {
                        const int lt = token / (m.window_h * m.window_w);
                        const int lh = (token / m.window_w) % m.window_h;
                        const int lw = token % m.window_w;
                        const int t = gt * m.window_t + lt, h = gh * m.window_h + lh, w = gw * m.window_w + lw;
                        if (t < s.time && h < s.height && w < s.width)
                            for (int c = 0; c < s.channels; ++c)
                                output[bthwc_offset(s, b, t, h, w, c)] = windows[(static_cast<std::size_t>(window) * m.tokens_per_window + token) * s.channels + c];
                    }
                }
    return output;
}
}
