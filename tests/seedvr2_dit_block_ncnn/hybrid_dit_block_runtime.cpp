#include "hybrid_dit_block_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>

namespace seedvr2_dit_block
{
namespace
{
using attention_dispatcher::DiTBlockConfig;
using attention_dispatcher::NcnnModelCache;
using attention_dispatcher::NcnnModelFiles;

constexpr double kTheta = 10000.0;

bool checked_mul(std::size_t a, std::size_t b, std::size_t& r)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a) return false;
    r = a * b;
    return true;
}

std::vector<float> mat_flat(const ncnn::Mat& m)
{
    return std::vector<float>(static_cast<const float*>(m.data),
                              static_cast<const float*>(m.data) + m.total());
}

ncnn::Mat mat_from(const std::vector<float>& v, int w, int h)
{
    ncnn::Mat m(w, h);
    std::memcpy(m.data, v.data(), v.size() * sizeof(float));
    return m;
}

bool read_bin(const std::filesystem::path& path, std::vector<float>& out, std::size_t count, std::string& error)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { error = "params file missing: " + path.string(); return false; }
    const auto bytes = file.tellg();
    if (bytes != static_cast<std::streamoff>(count * sizeof(float))) { error = "params size mismatch " + path.string(); return false; }
    file.seekg(0);
    out.resize(count);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return bool(file);
}

double elapsed_ms(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int num_tokens(const std::vector<VideoShape>& shapes)
{
    int total = 0;
    for (const auto& s : shapes) total += s.t * s.h * s.w;
    return total;
}

// Python round() is round-half-to-even; match it exactly for window sizes.
int py_round(double value)
{
    const double floor_value = std::floor(value);
    const double diff = value - floor_value;
    if (diff > 0.5) return static_cast<int>(floor_value + 1.0);
    if (diff < 0.5) return static_cast<int>(floor_value);
    const int even = static_cast<int>(floor_value);
    return (even % 2 == 0) ? even : even + 1;
}
} // namespace

bool ada_single_in(const std::vector<float>& hid, int dim,
                   const std::vector<float>& emb_shift, const std::vector<float>& emb_scale,
                   const std::vector<int>& token_counts,
                   const std::vector<float>& shift_b, const std::vector<float>& scale_b,
                   std::vector<float>& out, std::string& error)
{
    out.clear();
    int total = 0;
    for (int c : token_counts)
    {
        if (c < 0) { error = "ada token count negative"; return false; }
        if (c && c > std::numeric_limits<int>::max() - total) { error = "ada token count overflow"; return false; }
        total += c;
    }
    const std::size_t batch = token_counts.size();
    std::size_t need = 0;
    if (!checked_mul(static_cast<std::size_t>(total), static_cast<std::size_t>(dim), need)) { error = "ada size overflow"; return false; }
    if (hid.size() != need) { error = "ada hidden token*dim mismatch"; return false; }
    std::size_t emb_need = 0;
    if (!checked_mul(batch, static_cast<std::size_t>(dim), emb_need)) { error = "ada emb size overflow"; return false; }
    if (emb_shift.size() != emb_need || emb_scale.size() != emb_need) { error = "ada emb must be [B,dim] shift/scale"; return false; }
    if (shift_b.size() != static_cast<std::size_t>(dim) || scale_b.size() != static_cast<std::size_t>(dim)) { error = "ada learned params must be [dim]"; return false; }
    out.resize(need);
    std::size_t token = 0;
    for (std::size_t b = 0; b < batch; ++b)
    {
        const float* e_shift = emb_shift.data() + b * static_cast<std::size_t>(dim);
        const float* e_scale = emb_scale.data() + b * static_cast<std::size_t>(dim);
        for (int t = 0; t < token_counts[b]; ++t, ++token)
        {
            float* dst = out.data() + token * static_cast<std::size_t>(dim);
            const float* src = hid.data() + token * static_cast<std::size_t>(dim);
            for (int c = 0; c < dim; ++c)
            {
                const float value = src[c] * (e_scale[c] + scale_b[c]) + (e_shift[c] + shift_b[c]);
                if (!std::isfinite(value)) { error = "ada output non-finite"; return false; }
                dst[c] = value;
            }
        }
    }
    error.clear();
    return true;
}

bool ada_single_gate(const std::vector<float>& hid, int dim,
                     const std::vector<float>& emb_gate,
                     const std::vector<int>& token_counts,
                     const std::vector<float>& gate_b,
                     std::vector<float>& out, std::string& error)
{
    out.clear();
    int total = 0;
    for (int c : token_counts)
    {
        if (c < 0) { error = "ada token count negative"; return false; }
        if (c && c > std::numeric_limits<int>::max() - total) { error = "ada token count overflow"; return false; }
        total += c;
    }
    const std::size_t batch = token_counts.size();
    std::size_t need = 0;
    if (!checked_mul(static_cast<std::size_t>(total), static_cast<std::size_t>(dim), need)) { error = "ada size overflow"; return false; }
    if (hid.size() != need) { error = "ada hidden token*dim mismatch"; return false; }
    std::size_t emb_need = 0;
    if (!checked_mul(batch, static_cast<std::size_t>(dim), emb_need)) { error = "ada emb size overflow"; return false; }
    if (emb_gate.size() != emb_need) { error = "ada emb must be [B,dim] gate"; return false; }
    if (gate_b.size() != static_cast<std::size_t>(dim)) { error = "ada gate param must be [dim]"; return false; }
    out.resize(need);
    std::size_t token = 0;
    for (std::size_t b = 0; b < batch; ++b)
    {
        const float* e_gate = emb_gate.data() + b * static_cast<std::size_t>(dim);
        for (int t = 0; t < token_counts[b]; ++t, ++token)
        {
            float* dst = out.data() + token * static_cast<std::size_t>(dim);
            const float* src = hid.data() + token * static_cast<std::size_t>(dim);
            for (int c = 0; c < dim; ++c)
            {
                const float value = src[c] * (e_gate[c] + gate_b[c]);
                if (!std::isfinite(value)) { error = "ada output non-finite"; return false; }
                dst[c] = value;
            }
        }
    }
    error.clear();
    return true;
}

bool compute_windows(const std::vector<VideoShape>& shapes,
                     int window_t, int window_h, int window_w,
                     WindowInfo& out, std::string& error)
{
    out = WindowInfo{};
    if (window_t <= 0 || window_h <= 0 || window_w <= 0) { error = "window must be positive"; return false; }
    int sample_offset = 0;
    for (const auto& s : shapes)
    {
        if (s.t <= 0 || s.h <= 0 || s.w <= 0) { error = "video shape must be positive"; return false; }
        if (s.t > std::numeric_limits<int>::max() / (s.h * s.w)) { error = "video token overflow"; return false; }
        const double scale = std::sqrt(3600.0 / (static_cast<double>(s.h) * s.w));
        const int resized_h = py_round(s.h * scale);
        const int resized_w = py_round(s.w * scale);
        const int wh = static_cast<int>(std::ceil(resized_h / static_cast<double>(window_h)));
        const int ww = static_cast<int>(std::ceil(resized_w / static_cast<double>(window_w)));
        const int wt = static_cast<int>(std::ceil(std::min(s.t, 30) / static_cast<double>(window_t)));
        const int nt = (wt > 0) ? (s.t + wt - 1) / wt : 0;
        const int nh = (wh > 0) ? (s.h + wh - 1) / wh : 0;
        const int nw = (ww > 0) ? (s.w + ww - 1) / ww : 0;
        int count = 0;
        for (int iw = 0; iw < nw; ++iw)
        {
            const int sw0 = iw * ww;
            const int sw1 = std::min((iw + 1) * ww, s.w);
            if (sw1 <= sw0) continue;
            for (int ih = 0; ih < nh; ++ih)
            {
                const int sh0 = ih * wh;
                const int sh1 = std::min((ih + 1) * wh, s.h);
                if (sh1 <= sh0) continue;
                for (int it = 0; it < nt; ++it)
                {
                    const int st0 = it * wt;
                    const int st1 = std::min((it + 1) * wt, s.t);
                    if (st1 <= st0) continue;
                    out.window_shapes.push_back({st1 - st0, sh1 - sh0, sw1 - sw0});
                    out.window_sizes.push_back((st1 - st0) * (sh1 - sh0) * (sw1 - sw0));
                    for (int t = st0; t < st1; ++t)
                        for (int h = sh0; h < sh1; ++h)
                            for (int w = sw0; w < sw1; ++w)
                                out.partition_index.push_back(sample_offset + (t * s.h + h) * s.w + w);
                    ++count;
                }
            }
        }
        out.window_counts.push_back(count);
        sample_offset += s.t * s.h * s.w;
    }
    out.reverse_index.assign(static_cast<std::size_t>(sample_offset), 0);
    for (std::size_t i = 0; i < out.partition_index.size(); ++i)
        out.reverse_index[static_cast<std::size_t>(out.partition_index[i])] = static_cast<int>(i);
    error.clear();
    return true;
}

namespace
{
void make_pair_table(int rows, int pairs, const std::vector<float>& angles, std::vector<float>& cos, std::vector<float>& sin)
{
    cos.assign(static_cast<std::size_t>(rows) * pairs, 1.f);
    sin.assign(static_cast<std::size_t>(rows) * pairs, 0.f);
    for (int token = 0; token < rows; ++token)
        for (int p = 0; p < pairs; ++p)
        {
            const float angle = angles[static_cast<std::size_t>(token) * pairs + p];
            cos[static_cast<std::size_t>(token) * pairs + p] = std::cos(angle);
            sin[static_cast<std::size_t>(token) * pairs + p] = std::sin(angle);
        }
}
} // namespace

bool generate_vid_rope_cos_sin(const std::vector<VideoShape>& shapes,
                               const std::vector<int>& txt_lengths,
                               const WindowInfo& windows,
                               const DiTBlockConfig& block,
                               std::vector<float>& cos, std::vector<float>& sin,
                               std::string& error)
{
    if (shapes.size() != txt_lengths.size()) { error = "vid rope batch mismatch"; return false; }
    if (block.effective_rotary_dim % 6 != 0) { error = "vid rope effective_rotary_dim must be divisible by 6"; return false; }
    const int axis_pairs = block.effective_rotary_dim / 6;
    const int pairs = block.head_pairs;
    std::size_t total = 0;
    for (int s : windows.window_sizes) total += static_cast<std::size_t>(s);
    std::vector<float> angles(total * pairs, 0.f);
    std::size_t token = 0;
    int win = 0;
    for (std::size_t b = 0; b < shapes.size(); ++b)
    {
        const int l = txt_lengths[b];
        for (int n = 0; n < windows.window_counts[b]; ++n, ++win)
        {
            const auto& ws = windows.window_shapes[static_cast<std::size_t>(win)];
            for (int t = 0; t < ws[0]; ++t)
                for (int h = 0; h < ws[1]; ++h)
                    for (int w = 0; w < ws[2]; ++w, ++token)
                    {
                        const int positions[3] = {l + t, h, w};
                        for (int axis = 0; axis < 3; ++axis)
                            for (int k = 0; k < axis_pairs; ++k)
                            {
                                const float inv = static_cast<float>(std::exp(-std::log(kTheta) * static_cast<double>(k) / axis_pairs));
                                const int pair = axis * axis_pairs + k;
                                angles[token * pairs + pair] = static_cast<float>(positions[axis]) * inv;
                            }
                    }
        }
    }
    make_pair_table(static_cast<int>(total), pairs, angles, cos, sin);
    error.clear();
    return true;
}

bool generate_txt_rope_cos_sin(const std::vector<int>& txt_lengths,
                               const WindowInfo& windows,
                               const DiTBlockConfig& block,
                               std::vector<float>& cos, std::vector<float>& sin,
                               std::string& error)
{
    if (txt_lengths.size() != windows.window_counts.size()) { error = "txt rope batch mismatch"; return false; }
    if (block.effective_rotary_dim % 6 != 0) { error = "txt rope effective_rotary_dim must be divisible by 6"; return false; }
    const int axis_pairs = block.effective_rotary_dim / 6;
    const int pairs = block.head_pairs;
    std::size_t total = 0;
    for (std::size_t b = 0; b < txt_lengths.size(); ++b)
        total += static_cast<std::size_t>(txt_lengths[b]) * windows.window_counts[b];
    std::vector<float> angles(total * pairs, 0.f);
    std::size_t token = 0;
    for (std::size_t b = 0; b < txt_lengths.size(); ++b)
    {
        const int l = txt_lengths[b];
        for (int n = 0; n < windows.window_counts[b]; ++n)
            for (int j = 0; j < l; ++j, ++token)
                for (int axis = 0; axis < 3; ++axis)
                    for (int k = 0; k < axis_pairs; ++k)
                    {
                        const float inv = static_cast<float>(std::exp(-std::log(kTheta) * static_cast<double>(k) / axis_pairs));
                        const int pair = axis * axis_pairs + k;
                        angles[token * pairs + pair] = static_cast<float>(j) * inv;
                    }
    }
    make_pair_table(static_cast<int>(total), pairs, angles, cos, sin);
    error.clear();
    return true;
}

HybridDiTBlockRuntime::HybridDiTBlockRuntime(NcnnModelCache& cache) : cache_(cache) {}

ncnn::Net* HybridDiTBlockRuntime::model(const std::string& name) const
{
    auto it = models_.find(name);
    return it == models_.end() ? nullptr : it->second;
}

bool HybridDiTBlockRuntime::initialize(const attention_dispatcher::CoreInfo& core, std::string& error)
{
    const auto* artifact = core.dit_block();
    if (!artifact) { error = "core execution is not staged_dit_block"; return false; }
    core_ = &core;
    block_ = artifact->block;
    const char* sides[] = {"vid", "txt"};
    const char* layers[] = {"attn", "mlp"};
    const char* parts[] = {"shift", "scale", "gate"};
    for (const char* side : sides)
    {
        if (block_.is_last_layer && std::string(side) == "txt") continue; // last layer is vid_only
        for (const char* layer : layers)
            for (int p = 0; p < 3; ++p)
            {
                const std::string name = std::string("ada_") + side + "_" + layer + "_" + parts[p] + ".bin";
                std::vector<float> values;
                if (!read_bin(artifact->params_dir / name, values, static_cast<std::size_t>(block_.hidden_dim), error)) return false;
                params_[side][layer][p] = std::move(values);
            }
    }
    for (const auto& entry : artifact->models)
    {
        attention_dispatcher::CacheLoadStats stats;
        ncnn::Net* net = cache_.get_or_load(entry.second, stats, error);
        if (!net) { error = "stage model load failed: " + entry.first + ": " + error; return false; }
        models_[entry.first] = net;
    }
    initialized_ = true;
    error.clear();
    return true;
}

bool HybridDiTBlockRuntime::run_linear(const std::string& name, const std::vector<float>& input, int in_dim,
                                       std::vector<float>& output, int& out_dim, std::string& error)
{
    ncnn::Net* net = model(name);
    if (!net) { error = "stage model missing: " + name; return false; }
    if (input.size() % static_cast<std::size_t>(in_dim) != 0) { error = name + " input dim mismatch"; return false; }
    const int rows = static_cast<int>(input.size() / in_dim);
    if (rows == 0) { error = name + " empty input"; return false; }
    ncnn::Mat in = mat_from(input, in_dim, rows);
    ncnn::Mat out;
    auto ex = net->create_extractor();
    if (ex.input("in0", in) || ex.extract("out0", out)) { error = name + " inference failed"; return false; }
    if (!out.data || out.total() % static_cast<std::size_t>(rows) != 0) { error = name + " output shape invalid"; return false; }
    out_dim = static_cast<int>(out.total() / rows);
    output = mat_flat(out);
    if (output.size() != static_cast<std::size_t>(rows) * out_dim) { error = name + " output size mismatch"; return false; }
    error.clear();
    return true;
}

bool HybridDiTBlockRuntime::run_rms(const std::string& name, const std::vector<float>& input, int dim,
                                    std::vector<float>& output, std::string& error)
{
    int out_dim = 0;
    if (!run_linear(name, input, dim, output, out_dim, error)) return false;
    if (out_dim != dim) { error = name + " rms output dim mismatch"; return false; }
    error.clear();
    return true;
}

bool HybridDiTBlockRuntime::run_rope_chunked(const std::vector<float>& q, int rows, int head_dim,
                                             const std::vector<float>& cos, const std::vector<float>& sin,
                                             std::vector<float>& out, std::string& error)
{
    ncnn::Net* net = model("rope_apply");
    if (!net) { error = "stage model missing: rope_apply"; return false; }
    const int pairs = head_dim / 2;
    if (q.size() != static_cast<std::size_t>(rows) * head_dim) { error = "rope q shape mismatch"; return false; }
    if (cos.size() != static_cast<std::size_t>(rows) * pairs || sin.size() != static_cast<std::size_t>(rows) * pairs) { error = "rope cos/sin shape mismatch"; return false; }
    const int cap = block_.q_max;
    if (cap < head_dim) { error = "rope q_max too small"; return false; }
    std::vector<float> padded_q(static_cast<std::size_t>(cap) * head_dim, 0.f);
    std::vector<float> padded_cos(static_cast<std::size_t>(cap) * pairs, 1.f);
    std::vector<float> padded_sin(static_cast<std::size_t>(cap) * pairs, 0.f);
    out.assign(static_cast<std::size_t>(rows) * head_dim, 0.f);
    for (int begin = 0; begin < rows; begin += cap)
    {
        const int chunk = std::min(cap, rows - begin);
        std::memcpy(padded_q.data(), q.data() + static_cast<std::size_t>(begin) * head_dim, static_cast<std::size_t>(chunk) * head_dim * sizeof(float));
        std::memcpy(padded_cos.data(), cos.data() + static_cast<std::size_t>(begin) * pairs, static_cast<std::size_t>(chunk) * pairs * sizeof(float));
        std::memcpy(padded_sin.data(), sin.data() + static_cast<std::size_t>(begin) * pairs, static_cast<std::size_t>(chunk) * pairs * sizeof(float));
        ncnn::Mat qm = mat_from(padded_q, head_dim, cap);
        ncnn::Mat cm = mat_from(padded_cos, pairs, cap);
        ncnn::Mat sm = mat_from(padded_sin, pairs, cap);
        ncnn::Mat om;
        auto ex = net->create_extractor();
        if (ex.input("in0", qm) || ex.input("in1", cm) || ex.input("in2", sm) || ex.extract("out0", om)) { error = "rope_apply inference failed"; return false; }
        if (om.total() != static_cast<std::size_t>(cap) * head_dim) { error = "rope_apply output shape invalid"; return false; }
        std::memcpy(out.data() + static_cast<std::size_t>(begin) * head_dim, om.data, static_cast<std::size_t>(chunk) * head_dim * sizeof(float));
    }
    error.clear();
    return true;
}

bool HybridDiTBlockRuntime::run_head_segment(const std::vector<float>& q, const std::vector<float>& k,
                                             const std::vector<float>& v, int seg_len, int head_dim,
                                             std::vector<float>& out, std::string& error)
{
    ncnn::Net* net = model("head0");
    if (!net) { error = "stage model missing: head0"; return false; }
    const int cap = block_.q_max;
    if (seg_len <= 0 || seg_len > cap) { error = "packed segment length out of range for fixed-capacity head"; return false; }
    const std::size_t need = static_cast<std::size_t>(seg_len) * head_dim;
    if (q.size() != need || k.size() != need || v.size() != need) { error = "head segment q/k/v shape mismatch"; return false; }
    std::vector<float> qp(static_cast<std::size_t>(cap) * head_dim, 0.f);
    std::vector<float> kp(static_cast<std::size_t>(cap) * head_dim, 0.f);
    std::vector<float> vp(static_cast<std::size_t>(cap) * head_dim, 0.f);
    std::memcpy(qp.data(), q.data(), need * sizeof(float));
    std::memcpy(kp.data(), k.data(), need * sizeof(float));
    std::memcpy(vp.data(), v.data(), need * sizeof(float));
    std::vector<float> mask(static_cast<std::size_t>(cap) * cap, -10000.f);
    for (int i = 0; i < seg_len; ++i)
        for (int j = 0; j < seg_len; ++j)
            mask[static_cast<std::size_t>(i) * cap + j] = 0.f;
    ncnn::Mat qm = mat_from(qp, head_dim, cap);
    ncnn::Mat km = mat_from(kp, head_dim, cap);
    ncnn::Mat vm = mat_from(vp, head_dim, cap);
    ncnn::Mat mm = mat_from(mask, cap, cap);
    ncnn::Mat om;
    auto ex = net->create_extractor();
    if (ex.input("in0", qm) || ex.input("in1", km) || ex.input("in2", vm) || ex.input("in3", mm) || ex.extract("out0", om)) { error = "head inference failed"; return false; }
    if (om.total() != static_cast<std::size_t>(cap) * head_dim) { error = "head output shape invalid"; return false; }
    out.resize(need);
    std::memcpy(out.data(), om.data, need * sizeof(float));
    error.clear();
    return true;
}

bool HybridDiTBlockRuntime::apply_ada(const std::string& side, const std::string& layer, const std::string& mode,
                                      const std::vector<float>& hid, int tokens,
                                      const ncnn::Mat& emb, const std::vector<int>& token_counts,
                                      std::vector<float>& out, std::string& error)
{
    if (block_.is_last_layer && side == "txt") { out = hid; error.clear(); return true; }
    auto side_it = params_.find(side);
    if (side_it == params_.end()) { error = "ada params missing side " + side; return false; }
    auto layer_it = side_it->second.find(layer);
    if (layer_it == side_it->second.end()) { error = "ada params missing layer " + layer; return false; }
    const int dim = block_.hidden_dim;
    const int layer_index = (layer == "attn") ? 0 : 1;
    const int offset = layer_index * 3 * dim;
    if (emb.w != 6 * dim || emb.h <= 0) { error = "ada emb must be [B,6*C]"; return false; }
    const int batch = emb.h;
    if (batch != static_cast<int>(token_counts.size())) { error = "ada batch mismatch"; return false; }
    const float* ebase = static_cast<const float*>(emb.data);
    // Real SeedVR2 AdaSingle layout: emb [B,6*C] is rearrange("b (d l g) -> b d l g",
    // l=2, g=3); per dim d the six values are
    // [attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate].
    std::vector<float> e_shift, e_scale, e_gate;
    e_shift.resize(static_cast<std::size_t>(batch) * dim);
    e_scale.resize(static_cast<std::size_t>(batch) * dim);
    e_gate.resize(static_cast<std::size_t>(batch) * dim);
    for (int b = 0; b < batch; ++b)
    {
        const float* row = ebase + static_cast<std::size_t>(b) * 6 * dim;
        for (int c = 0; c < dim; ++c)
        {
            const float* base = row + static_cast<std::size_t>(6 * c + layer_index * 3);
            e_shift[static_cast<std::size_t>(b) * dim + c] = base[0];
            e_scale[static_cast<std::size_t>(b) * dim + c] = base[1];
            e_gate[static_cast<std::size_t>(b) * dim + c] = base[2];
        }
    }
    if (mode == "in")
        return ada_single_in(hid, dim, e_shift, e_scale, token_counts, layer_it->second[0], layer_it->second[1], out, error);
    if (mode == "out")
        return ada_single_gate(hid, dim, e_gate, token_counts, layer_it->second[2], out, error);
    error = "ada mode must be in or out";
    return false;
}

bool HybridDiTBlockRuntime::swiglu_forward(const std::string& side, const ncnn::Mat& input,
                                           SwigluStageOutput& out, std::string& error)
{
    if (block_.is_last_layer && side == "txt") { error = "last-layer SwiGLU is vid_only"; return false; }
    const int rows = input.h;
    if (rows <= 0 || input.w != block_.hidden_dim) { error = "swiglu input shape invalid"; return false; }
    const std::vector<float> flat = mat_flat(input);
    if (!std::all_of(flat.begin(), flat.end(), [](float v) { return std::isfinite(v); })) { error = "swiglu input non-finite"; return false; }
    int out_dim = 0;
    if (!run_linear("swiglu_gate_" + side, flat, block_.hidden_dim, out.gate, out_dim, error)) return false;
    if (out_dim != block_.mlp_hidden_dim) { error = "swiglu gate out dim mismatch"; return false; }
    if (!run_linear("swiglu_up_" + side, flat, block_.hidden_dim, out.up, out_dim, error)) return false;
    if (out_dim != block_.mlp_hidden_dim) { error = "swiglu up out dim mismatch"; return false; }
    ncnn::Net* act = model("swiglu_act");
    if (!act) { error = "stage model missing: swiglu_act"; return false; }
    ncnn::Mat gm = mat_from(out.gate, block_.mlp_hidden_dim, rows);
    ncnn::Mat um = mat_from(out.up, block_.mlp_hidden_dim, rows);
    ncnn::Mat pm;
    auto ex = act->create_extractor();
    if (ex.input("in0", gm) || ex.input("in1", um) || ex.extract("out0", pm)) { error = "swiglu_act inference failed"; return false; }
    out.product = mat_flat(pm);
    if (out.product.size() != static_cast<std::size_t>(rows) * block_.mlp_hidden_dim) { error = "swiglu product shape mismatch"; return false; }
    if (!std::all_of(out.product.begin(), out.product.end(), [](float v) { return std::isfinite(v); })) { error = "swiglu product non-finite"; return false; }
    if (!run_linear("swiglu_out_" + side, out.product, block_.mlp_hidden_dim, out.output, out_dim, error)) return false;
    if (out_dim != block_.hidden_dim) { error = "swiglu out dim mismatch"; return false; }
    if (!std::all_of(out.output.begin(), out.output.end(), [](float v) { return std::isfinite(v); })) { error = "swiglu output non-finite"; return false; }
    error.clear();
    return true;
}

bool HybridDiTBlockRuntime::run(const DiTBlockInput& input, DiTBlockOutput& output,
                                DiTBlockTrace* trace, DiTBlockRunStats& stats)
{
    stats = DiTBlockRunStats{};
    const auto total_start = std::chrono::steady_clock::now();
    if (!initialized_) { stats.error_message = "runtime not initialized"; return false; }
    if (!core_) { stats.error_message = "runtime core missing"; return false; }
    const int dim = block_.hidden_dim;
    const int heads = block_.heads;
    const int head_dim = block_.head_dim;
    const int qkv_dim = 3 * dim;
    const int mlp = block_.mlp_hidden_dim;
    std::string& error = stats.error_message;
    stats.cache_size_before = cache_.size();

    const int batch = static_cast<int>(input.vid_shapes.size());
    if (batch <= 0) { error = "batch must be positive"; return false; }
    if (input.txt_lengths.size() != static_cast<std::size_t>(batch)) { error = "batch mismatch between vid_shapes and txt_lengths"; return false; }
    const int vid_tokens = num_tokens(input.vid_shapes);
    int txt_tokens = 0;
    for (int l : input.txt_lengths)
    {
        if (l < 0) { error = "negative txt length"; return false; }
        txt_tokens += l;
    }
    if (input.vid.h != vid_tokens || input.vid.w != dim) { error = "vid token/dim mismatch"; return false; }
    if (input.txt.h != txt_tokens || input.txt.w != dim) { error = "txt token/dim mismatch"; return false; }
    if (input.emb.w != 6 * dim || input.emb.h != batch) { error = "emb must be [B,6*C]"; return false; }
    const std::vector<float> vid0 = mat_flat(input.vid);
    const std::vector<float> txt0 = mat_flat(input.txt);
    if (!std::all_of(vid0.begin(), vid0.end(), [](float v) { return std::isfinite(v); })) { error = "vid input non-finite"; return false; }
    if (!std::all_of(txt0.begin(), txt0.end(), [](float v) { return std::isfinite(v); })) { error = "txt input non-finite"; return false; }
    if (!std::all_of(static_cast<const float*>(input.emb.data), static_cast<const float*>(input.emb.data) + input.emb.total(), [](float v) { return std::isfinite(v); })) { error = "emb input non-finite"; return false; }

    std::vector<int> vid_token_counts;
    for (const auto& s : input.vid_shapes) vid_token_counts.push_back(s.t * s.h * s.w);
    std::vector<int> txt_token_counts = input.txt_lengths;

    std::vector<float> attn_norm_vid, attn_norm_txt;
    if (!run_rms("attn_norm", vid0, dim, attn_norm_vid, error)) return false;
    if (!run_rms("attn_norm", txt0, dim, attn_norm_txt, error)) return false;

    std::vector<float> ada_attn_in_vid, ada_attn_in_txt;
    if (!apply_ada("vid", "attn", "in", attn_norm_vid, vid_tokens, input.emb, vid_token_counts, ada_attn_in_vid, error)) return false;
    if (!apply_ada("txt", "attn", "in", attn_norm_txt, txt_tokens, input.emb, txt_token_counts, ada_attn_in_txt, error)) return false;

    std::vector<float> qkv_vid, qkv_txt;
    int out_dim = 0;
    if (!run_linear("qkv_vid", ada_attn_in_vid, dim, qkv_vid, out_dim, error)) return false;
    if (out_dim != qkv_dim) { error = "qkv_vid out dim mismatch"; return false; }
    if (!run_linear("qkv_txt", ada_attn_in_txt, dim, qkv_txt, out_dim, error)) return false;
    if (out_dim != qkv_dim) { error = "qkv_txt out dim mismatch"; return false; }

    // Real layout: qkv output rows are [token, (o h d)] with o=3 and d=head_dim,
    // so q/k/v are strided 512-wide slices of each 1536-wide token row.
    const std::ptrdiff_t vd = static_cast<std::ptrdiff_t>(dim);
    const std::ptrdiff_t qkv_stride = static_cast<std::ptrdiff_t>(qkv_dim);
    std::vector<float> q_vid(static_cast<std::size_t>(vid_tokens) * vd);
    std::vector<float> k_vid(static_cast<std::size_t>(vid_tokens) * vd);
    std::vector<float> v_vid(static_cast<std::size_t>(vid_tokens) * vd);
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(vid_tokens); ++i)
    {
        std::memcpy(q_vid.data() + i * vd, qkv_vid.data() + i * qkv_stride, static_cast<std::size_t>(vd) * sizeof(float));
        std::memcpy(k_vid.data() + i * vd, qkv_vid.data() + i * qkv_stride + vd, static_cast<std::size_t>(vd) * sizeof(float));
        std::memcpy(v_vid.data() + i * vd, qkv_vid.data() + i * qkv_stride + 2 * vd, static_cast<std::size_t>(vd) * sizeof(float));
    }
    std::vector<float> q_txt(static_cast<std::size_t>(txt_tokens) * vd);
    std::vector<float> k_txt(static_cast<std::size_t>(txt_tokens) * vd);
    std::vector<float> v_txt(static_cast<std::size_t>(txt_tokens) * vd);
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(txt_tokens); ++i)
    {
        std::memcpy(q_txt.data() + i * vd, qkv_txt.data() + i * qkv_stride, static_cast<std::size_t>(vd) * sizeof(float));
        std::memcpy(k_txt.data() + i * vd, qkv_txt.data() + i * qkv_stride + vd, static_cast<std::size_t>(vd) * sizeof(float));
        std::memcpy(v_txt.data() + i * vd, qkv_txt.data() + i * qkv_stride + 2 * vd, static_cast<std::size_t>(vd) * sizeof(float));
    }

    std::vector<float> q_norm_vid, k_norm_vid, q_norm_txt, k_norm_txt;
    // q/k RMSNorm is per-head (dim=head_dim=128); the flat [tokens*heads*128]
    // buffer viewed as [tokens*heads, 128] is row-major per (token, head).
    if (!run_rms("q_norm_vid", q_vid, head_dim, q_norm_vid, error)) return false;
    if (!run_rms("k_norm_vid", k_vid, head_dim, k_norm_vid, error)) return false;
    if (!run_rms("q_norm_txt", q_txt, head_dim, q_norm_txt, error)) return false;
    if (!run_rms("k_norm_txt", k_txt, head_dim, k_norm_txt, error)) return false;

    WindowInfo win;
    if (!compute_windows(input.vid_shapes, core_->key.window_t, core_->key.window_h, core_->key.window_w, win, error)) return false;
    const int win_tokens = static_cast<int>(win.partition_index.size());
    int txt_repeat_tokens = 0;
    for (std::size_t b = 0; b < txt_token_counts.size(); ++b) txt_repeat_tokens += txt_token_counts[b] * win.window_counts[b];
    std::vector<float> vid_q_win(static_cast<std::size_t>(win_tokens) * dim);
    std::vector<float> vid_k_win(static_cast<std::size_t>(win_tokens) * dim);
    std::vector<float> vid_v_win(static_cast<std::size_t>(win_tokens) * dim);
    for (std::size_t i = 0; i < win.partition_index.size(); ++i)
    {
        const std::size_t src = static_cast<std::size_t>(win.partition_index[i]) * dim;
        std::memcpy(vid_q_win.data() + i * static_cast<std::size_t>(dim), q_norm_vid.data() + src, static_cast<std::size_t>(dim) * sizeof(float));
        std::memcpy(vid_k_win.data() + i * static_cast<std::size_t>(dim), k_norm_vid.data() + src, static_cast<std::size_t>(dim) * sizeof(float));
        std::memcpy(vid_v_win.data() + i * static_cast<std::size_t>(dim), v_vid.data() + src, static_cast<std::size_t>(dim) * sizeof(float));
    }
    std::vector<float> txt_q_rep(static_cast<std::size_t>(txt_repeat_tokens) * dim);
    std::vector<float> txt_k_rep(static_cast<std::size_t>(txt_repeat_tokens) * dim);
    std::vector<float> txt_v_rep(static_cast<std::size_t>(txt_repeat_tokens) * dim);
    {
        std::size_t dst = 0;
        for (std::size_t b = 0; b < txt_token_counts.size(); ++b)
        {
            const std::size_t sample_off = std::accumulate(txt_token_counts.begin(), txt_token_counts.begin() + static_cast<std::ptrdiff_t>(b), std::size_t(0)) * dim;
            for (int n = 0; n < win.window_counts[b]; ++n)
                for (int j = 0; j < txt_token_counts[b]; ++j, dst += static_cast<std::size_t>(dim))
                {
                    const std::size_t src = sample_off + static_cast<std::size_t>(j) * dim;
                    std::memcpy(txt_q_rep.data() + dst, q_norm_txt.data() + src, static_cast<std::size_t>(dim) * sizeof(float));
                    std::memcpy(txt_k_rep.data() + dst, k_norm_txt.data() + src, static_cast<std::size_t>(dim) * sizeof(float));
                    std::memcpy(txt_v_rep.data() + dst, v_txt.data() + src, static_cast<std::size_t>(dim) * sizeof(float));
                }
        }
    }

    std::vector<float> vid_cos, vid_sin, txt_cos, txt_sin;
    if (!generate_vid_rope_cos_sin(input.vid_shapes, input.txt_lengths, win, block_, vid_cos, vid_sin, error)) return false;
    if (!generate_txt_rope_cos_sin(input.txt_lengths, win, block_, txt_cos, txt_sin, error)) return false;
    // The ncnn rope_apply stage rotates one head ([m,head_dim] + [m,pairs] cos/sin);
    // apply it head-by-head over the [rows, heads*head_dim] buffers.
    const auto rope_all_heads = [&](const std::vector<float>& qin, int rows,
                                    const std::vector<float>& cos, const std::vector<float>& sin,
                                    std::vector<float>& out) -> bool {
        std::string err;
        out.assign(static_cast<std::size_t>(rows) * dim, 0.f);
        std::vector<float> hbuf(static_cast<std::size_t>(rows) * head_dim);
        std::vector<float> hout;
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < rows; ++i)
                std::memcpy(hbuf.data() + static_cast<std::size_t>(i) * head_dim,
                            qin.data() + static_cast<std::size_t>(i) * dim + static_cast<std::size_t>(h) * head_dim,
                            static_cast<std::size_t>(head_dim) * sizeof(float));
            if (!run_rope_chunked(hbuf, rows, head_dim, cos, sin, hout, err)) { error = err; return false; }
            for (int i = 0; i < rows; ++i)
                std::memcpy(out.data() + static_cast<std::size_t>(i) * dim + static_cast<std::size_t>(h) * head_dim,
                            hout.data() + static_cast<std::size_t>(i) * head_dim,
                            static_cast<std::size_t>(head_dim) * sizeof(float));
        }
        return true;
    };
    std::vector<float> rope_q_vid, rope_k_vid, rope_q_txt, rope_k_txt;
    if (!rope_all_heads(vid_q_win, win_tokens, vid_cos, vid_sin, rope_q_vid)) return false;
    if (!rope_all_heads(vid_k_win, win_tokens, vid_cos, vid_sin, rope_k_vid)) return false;
    if (!rope_all_heads(txt_q_rep, txt_repeat_tokens, txt_cos, txt_sin, rope_q_txt)) return false;
    if (!rope_all_heads(txt_k_rep, txt_repeat_tokens, txt_cos, txt_sin, rope_k_txt)) return false;

    std::vector<int> seg_lens;
    int total_packed = 0;
    {
        int win_idx = 0;
        for (std::size_t b = 0; b < static_cast<std::size_t>(batch); ++b)
            for (int n = 0; n < win.window_counts[b]; ++n, ++win_idx)
            {
                const int seg = win.window_sizes[static_cast<std::size_t>(win_idx)] + txt_token_counts[b];
                seg_lens.push_back(seg);
                total_packed += seg;
            }
    }
    std::vector<int> cu(seg_lens.size() + 1, 0);
    for (std::size_t i = 0; i < seg_lens.size(); ++i) cu[i + 1] = cu[i] + seg_lens[i];
    std::vector<float> packed_q(static_cast<std::size_t>(total_packed) * dim);
    std::vector<float> packed_k(static_cast<std::size_t>(total_packed) * dim);
    std::vector<float> packed_v(static_cast<std::size_t>(total_packed) * dim);
    {
        // Real repeat_concat_idx builds tgt_idx from the ORIGINAL per-sample txt
        // offsets (txt_idx = vid_len_sum + [cum_b, cum_b + l_b)) but the q/k tensors
        // actually concatenated are the window-repeated rope outputs, so every window
        // of sample b reuses repeated-txt rows [cum_b, cum_b + l_b). v is not roped
        // (original txt), so its per-window copies are identical rows and the
        // advancing txt_src reproduces the reference for v exactly.
        std::size_t dst = 0, vid_src = 0, txt_src = 0, qk_txt_off = 0;
        int win_idx = 0;
        for (std::size_t b = 0; b < static_cast<std::size_t>(batch); ++b)
        {
            const int tn = txt_token_counts[b];
            for (int n = 0; n < win.window_counts[b]; ++n, ++win_idx)
            {
                const int vn = win.window_sizes[static_cast<std::size_t>(win_idx)];
                std::memcpy(packed_q.data() + dst, rope_q_vid.data() + vid_src, static_cast<std::size_t>(vn) * dim * sizeof(float));
                std::memcpy(packed_k.data() + dst, rope_k_vid.data() + vid_src, static_cast<std::size_t>(vn) * dim * sizeof(float));
                std::memcpy(packed_v.data() + dst, vid_v_win.data() + vid_src, static_cast<std::size_t>(vn) * dim * sizeof(float));
                dst += static_cast<std::size_t>(vn) * dim;
                vid_src += static_cast<std::size_t>(vn) * dim;
                std::memcpy(packed_q.data() + dst, rope_q_txt.data() + qk_txt_off, static_cast<std::size_t>(tn) * dim * sizeof(float));
                std::memcpy(packed_k.data() + dst, rope_k_txt.data() + qk_txt_off, static_cast<std::size_t>(tn) * dim * sizeof(float));
                std::memcpy(packed_v.data() + dst, txt_v_rep.data() + txt_src, static_cast<std::size_t>(tn) * dim * sizeof(float));
                dst += static_cast<std::size_t>(tn) * dim;
                txt_src += static_cast<std::size_t>(tn) * dim;
            }
            qk_txt_off += static_cast<std::size_t>(tn) * dim;
        }
    }

    std::vector<float> attn_packed(static_cast<std::size_t>(total_packed) * dim);
    {
        std::size_t off = 0;
        for (int seg = 0; seg < static_cast<int>(seg_lens.size()); ++seg)
        {
            const int sl = seg_lens[static_cast<std::size_t>(seg)];
            for (int h = 0; h < heads; ++h)
            {
                ncnn::Net* head_net = model("head" + std::to_string(h));
                if (!head_net) { error = "stage model missing: head" + std::to_string(h); return false; }
                std::vector<float> qh, kh, vh, oh;
                qh.reserve(static_cast<std::size_t>(sl) * head_dim);
                kh.reserve(static_cast<std::size_t>(sl) * head_dim);
                vh.reserve(static_cast<std::size_t>(sl) * head_dim);
                for (int i = 0; i < sl; ++i)
                {
                    const std::size_t row = off + static_cast<std::size_t>(i) * dim + static_cast<std::size_t>(h) * head_dim;
                    qh.insert(qh.end(), packed_q.begin() + static_cast<std::ptrdiff_t>(row), packed_q.begin() + static_cast<std::ptrdiff_t>(row + head_dim));
                    kh.insert(kh.end(), packed_k.begin() + static_cast<std::ptrdiff_t>(row), packed_k.begin() + static_cast<std::ptrdiff_t>(row + head_dim));
                    vh.insert(vh.end(), packed_v.begin() + static_cast<std::ptrdiff_t>(row), packed_v.begin() + static_cast<std::ptrdiff_t>(row + head_dim));
                }
                if (!run_head_segment(qh, kh, vh, sl, head_dim, oh, error)) return false;
                for (int i = 0; i < sl; ++i)
                {
                    const std::size_t row = off + static_cast<std::size_t>(i) * dim + static_cast<std::size_t>(h) * head_dim;
                    std::memcpy(attn_packed.data() + row, oh.data() + static_cast<std::size_t>(i) * head_dim, static_cast<std::size_t>(head_dim) * sizeof(float));
                }
            }
            off += static_cast<std::size_t>(sl) * dim;
        }
    }

    std::vector<float> vid_windows, txt_windows;
    std::vector<float> vid_merged(static_cast<std::size_t>(vid_tokens) * dim);
    std::vector<float> txt_mean_merged(static_cast<std::size_t>(txt_tokens) * dim);
    {
        std::size_t off = 0;
        for (int seg = 0; seg < static_cast<int>(seg_lens.size()); ++seg)
        {
            const int vn = win.window_sizes[static_cast<std::size_t>(seg)];
            const int tn = seg_lens[static_cast<std::size_t>(seg)] - vn;
            const std::size_t seg_base = off;
            vid_windows.insert(vid_windows.end(), attn_packed.begin() + static_cast<std::ptrdiff_t>(seg_base),
                               attn_packed.begin() + static_cast<std::ptrdiff_t>(seg_base + static_cast<std::size_t>(vn) * dim));
            txt_windows.insert(txt_windows.end(), attn_packed.begin() + static_cast<std::ptrdiff_t>(seg_base + static_cast<std::size_t>(vn) * dim),
                               attn_packed.begin() + static_cast<std::ptrdiff_t>(seg_base + static_cast<std::size_t>(vn + tn) * dim));
            off = seg_base + static_cast<std::size_t>(vn + tn) * dim;
        }
        for (std::size_t i = 0; i < win.reverse_index.size(); ++i)
        {
            const std::size_t src = static_cast<std::size_t>(win.reverse_index[i]) * dim;
            std::memcpy(vid_merged.data() + i * static_cast<std::size_t>(dim), vid_windows.data() + src, static_cast<std::size_t>(dim) * sizeof(float));
        }
        std::size_t src = 0, dst = 0;
        for (std::size_t b = 0; b < static_cast<std::size_t>(batch); ++b)
        {
            const int l = txt_token_counts[b];
            const int n = win.window_counts[b];
            // Real unconcat_coalesce: argsort reorders the repeated txt to
            // token-major (t0 across windows, then t1 ...) and reshape(-1, n).mean(1)
            // averages the same token across windows; txt_windows is window-major
            // ([win0 t0..tl, win1 t0..tl, ...]) so the same-token rows are rep*l + j.
            std::vector<double> sums(static_cast<std::size_t>(l) * dim, 0.0);
            for (int rep = 0; rep < n; ++rep)
                for (int j = 0; j < l; ++j)
                    for (int c = 0; c < dim; ++c)
                        sums[static_cast<std::size_t>(j) * dim + c] += txt_windows[src + static_cast<std::size_t>(rep * l + j) * dim + c];
            for (int j = 0; j < l; ++j)
                for (int c = 0; c < dim; ++c)
                    txt_mean_merged[dst + static_cast<std::size_t>(j) * dim + c] = static_cast<float>(sums[static_cast<std::size_t>(j) * dim + c] / n);
            src += static_cast<std::size_t>(l) * n * dim;
            dst += static_cast<std::size_t>(l) * dim;
        }
    }

    std::vector<float> attn_proj_vid, attn_proj_txt;
    if (!run_linear("attn_proj_vid", vid_merged, dim, attn_proj_vid, out_dim, error)) return false;
    if (out_dim != dim) { error = "attn_proj_vid out dim mismatch"; return false; }
    if (!run_linear("attn_proj_txt", txt_mean_merged, dim, attn_proj_txt, out_dim, error)) return false;
    if (out_dim != dim) { error = "attn_proj_txt out dim mismatch"; return false; }

    std::vector<float> ada_attn_gate_vid, ada_attn_gate_txt;
    if (!apply_ada("vid", "attn", "out", attn_proj_vid, vid_tokens, input.emb, vid_token_counts, ada_attn_gate_vid, error)) return false;
    if (!apply_ada("txt", "attn", "out", attn_proj_txt, txt_tokens, input.emb, txt_token_counts, ada_attn_gate_txt, error)) return false;
    std::vector<float> attn_residual_vid(static_cast<std::size_t>(vid_tokens) * dim);
    std::vector<float> attn_residual_txt(static_cast<std::size_t>(txt_tokens) * dim);
    for (std::size_t i = 0; i < ada_attn_gate_vid.size(); ++i) attn_residual_vid[i] = ada_attn_gate_vid[i] + vid0[i];
    for (std::size_t i = 0; i < ada_attn_gate_txt.size(); ++i) attn_residual_txt[i] = ada_attn_gate_txt[i] + txt0[i];

    std::vector<float> mlp_norm_vid, mlp_norm_txt;
    if (!run_rms("mlp_norm", attn_residual_vid, dim, mlp_norm_vid, error)) return false;
    if (block_.is_last_layer) { mlp_norm_txt = attn_residual_txt; }
    else if (!run_rms("mlp_norm", attn_residual_txt, dim, mlp_norm_txt, error)) return false;

    std::vector<float> ada_mlp_in_vid, ada_mlp_in_txt;
    if (!apply_ada("vid", "mlp", "in", mlp_norm_vid, vid_tokens, input.emb, vid_token_counts, ada_mlp_in_vid, error)) return false;
    if (block_.is_last_layer) { ada_mlp_in_txt = mlp_norm_txt; }
    else if (!apply_ada("txt", "mlp", "in", mlp_norm_txt, txt_tokens, input.emb, txt_token_counts, ada_mlp_in_txt, error)) return false;

    SwigluStageOutput so_vid, so_txt;
    std::vector<float> swiglu_out_vid, swiglu_out_txt;
    {
        ncnn::Mat in = mat_from(ada_mlp_in_vid, dim, vid_tokens);
        if (!swiglu_forward("vid", in, so_vid, error)) return false;
        swiglu_out_vid = so_vid.output;
    }
    if (block_.is_last_layer) { swiglu_out_txt = ada_mlp_in_txt; }
    else
    {
        ncnn::Mat in = mat_from(ada_mlp_in_txt, dim, txt_tokens);
        if (!swiglu_forward("txt", in, so_txt, error)) return false;
        swiglu_out_txt = so_txt.output;
    }

    std::vector<float> ada_mlp_gate_vid, ada_mlp_gate_txt;
    if (!apply_ada("vid", "mlp", "out", swiglu_out_vid, vid_tokens, input.emb, vid_token_counts, ada_mlp_gate_vid, error)) return false;
    if (block_.is_last_layer) { ada_mlp_gate_txt = swiglu_out_txt; }
    else if (!apply_ada("txt", "mlp", "out", swiglu_out_txt, txt_tokens, input.emb, txt_token_counts, ada_mlp_gate_txt, error)) return false;
    std::vector<float> final_vid(static_cast<std::size_t>(vid_tokens) * dim);
    std::vector<float> final_txt(static_cast<std::size_t>(txt_tokens) * dim);
    for (std::size_t i = 0; i < ada_mlp_gate_vid.size(); ++i) final_vid[i] = ada_mlp_gate_vid[i] + attn_residual_vid[i];
    for (std::size_t i = 0; i < ada_mlp_gate_txt.size(); ++i) final_txt[i] = ada_mlp_gate_txt[i] + attn_residual_txt[i];
    if (!std::all_of(final_vid.begin(), final_vid.end(), [](float v) { return std::isfinite(v); }) ||
        !std::all_of(final_txt.begin(), final_txt.end(), [](float v) { return std::isfinite(v); })) { error = "final output non-finite"; return false; }

    output.vid = mat_from(final_vid, dim, vid_tokens);
    output.txt = mat_from(final_txt, dim, txt_tokens);
    if (trace)
    {
        trace->attn_norm_vid = attn_norm_vid; trace->attn_norm_txt = attn_norm_txt;
        trace->ada_attn_in_vid = ada_attn_in_vid; trace->ada_attn_in_txt = ada_attn_in_txt;
        trace->qkv_vid = qkv_vid; trace->qkv_txt = qkv_txt;
        trace->q_vid = q_vid; trace->k_vid = k_vid; trace->v_vid = v_vid;
        trace->q_txt = q_txt; trace->k_txt = k_txt; trace->v_txt = v_txt;
        trace->q_norm_vid = q_norm_vid; trace->k_norm_vid = k_norm_vid;
        trace->q_norm_txt = q_norm_txt; trace->k_norm_txt = k_norm_txt;
        trace->rope_before_q_vid = vid_q_win; trace->rope_before_k_vid = vid_k_win;
        trace->rope_after_q_vid = rope_q_vid; trace->rope_after_k_vid = rope_k_vid;
        trace->rope_after_q_txt = rope_q_txt; trace->rope_after_k_txt = rope_k_txt;
        trace->rope_cos_vid = vid_cos; trace->rope_sin_vid = vid_sin;
        trace->rope_cos_txt = txt_cos; trace->rope_sin_txt = txt_sin;
        trace->packed_q = packed_q; trace->packed_k = packed_k; trace->packed_v = packed_v;
        trace->attention_packed_output = attn_packed;
        trace->attention_vid_windows = vid_windows;
        trace->attention_txt_windows = txt_windows;
        trace->attention_vid_merged = vid_merged;
        trace->attention_txt_mean_merged = txt_mean_merged;
        trace->attention_proj_vid = attn_proj_vid; trace->attention_proj_txt = attn_proj_txt;
        trace->ada_attn_gate_vid = ada_attn_gate_vid; trace->ada_attn_gate_txt = ada_attn_gate_txt;
        trace->attention_residual_vid = attn_residual_vid; trace->attention_residual_txt = attn_residual_txt;
        trace->mlp_norm_vid = mlp_norm_vid; trace->mlp_norm_txt = mlp_norm_txt;
        trace->ada_mlp_in_vid = ada_mlp_in_vid; trace->ada_mlp_in_txt = ada_mlp_in_txt;
        trace->swiglu_gate_vid = so_vid.gate; trace->swiglu_up_vid = so_vid.up;
        trace->swiglu_product_vid = so_vid.product; trace->swiglu_output_vid = so_vid.output;
        if (!block_.is_last_layer)
        {
            trace->swiglu_gate_txt = so_txt.gate; trace->swiglu_up_txt = so_txt.up;
            trace->swiglu_product_txt = so_txt.product; trace->swiglu_output_txt = so_txt.output;
        }
        trace->ada_mlp_gate_vid = ada_mlp_gate_vid; trace->ada_mlp_gate_txt = ada_mlp_gate_txt;
        trace->final_vid = final_vid; trace->final_txt = final_txt;
    }

    stats.windows = win;
    stats.segment_lengths = seg_lens;
    stats.cu_seqlens = cu;
    stats.total_packed = total_packed;
    stats.cache_size_after = cache_.size();
    stats.total_ms = elapsed_ms(total_start);
    stats.success = true;
    error.clear();
    return true;
}
} // namespace seedvr2_dit_block
