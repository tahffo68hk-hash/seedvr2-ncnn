#pragma once
#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <net.h>

#include "artifact_registry.h"
#include "model_cache.h"

namespace seedvr2_dit_block
{
struct VideoShape
{
    int t = 0;
    int h = 0;
    int w = 0;
};

// Row-major float32 contract matching the reference logical shapes:
// vid  [L_vid, C], txt [L_txt, C], emb [B, 6*C].
struct DiTBlockInput
{
    ncnn::Mat vid;
    ncnn::Mat txt;
    std::vector<VideoShape> vid_shapes;
    std::vector<int> txt_lengths;
    ncnn::Mat emb;
};

struct DiTBlockOutput
{
    ncnn::Mat vid;
    ncnn::Mat txt;
};

struct WindowInfo
{
    std::vector<int> window_counts;                     // per sample
    std::vector<std::array<int, 3>> window_shapes;      // per window (t,h,w)
    std::vector<int> window_sizes;                      // per window (tokens)
    std::vector<int> partition_index;                   // packed order over vid tokens
    std::vector<int> reverse_index;                     // argsort(partition_index)
};

struct DiTBlockRunStats
{
    bool success = false;
    std::string error_message;
    std::size_t cache_size_before = 0;
    std::size_t cache_size_after = 0;
    int cache_hit_count = 0;
    int cache_miss_count = 0;
    double total_ms = 0.0;
    WindowInfo windows;
    std::vector<int> segment_lengths;                   // per window segment
    std::vector<int> cu_seqlens;
    int total_packed = 0;
};

// Every stage of the real NaMMSRTransformerBlock forward, kept row-major
// float32 in the reference logical layout.  "C" = hidden_dim, "HD" = head_dim.
struct DiTBlockTrace
{
    std::vector<float> attn_norm_vid, attn_norm_txt;          // [L,C]
    std::vector<float> ada_attn_in_vid, ada_attn_in_txt;      // [L,C]
    std::vector<float> qkv_vid, qkv_txt;                      // [L,3*C]
    std::vector<float> q_vid, k_vid, v_vid;                   // [L,C] = [L,H,HD]
    std::vector<float> q_txt, k_txt, v_txt;                   // [L,C]
    std::vector<float> q_norm_vid, k_norm_vid;                // [L,C]
    std::vector<float> q_norm_txt, k_norm_txt;                // [L,C]
    std::vector<float> rope_before_q_vid, rope_before_k_vid;  // windowed [W,C]
    std::vector<float> rope_after_q_vid, rope_after_k_vid;    // [W,C]
    std::vector<float> rope_after_q_txt, rope_after_k_txt;    // [T*N,C]
    std::vector<float> rope_cos_vid, rope_sin_vid;            // [W,head_pairs]
    std::vector<float> rope_cos_txt, rope_sin_txt;            // [T*N,head_pairs]
    std::vector<float> packed_q, packed_k, packed_v;          // [P,H,HD]
    std::vector<float> attention_packed_output;               // [P,H,HD]
    std::vector<float> attention_vid_windows;                 // [sum(W),H,HD]
    std::vector<float> attention_txt_windows;                 // [T*N,H,HD]
    std::vector<float> attention_vid_merged;                  // [L,C]
    std::vector<float> attention_txt_mean_merged;             // [T,H,HD]
    std::vector<float> attention_proj_vid, attention_proj_txt;// [L,C]
    std::vector<float> ada_attn_gate_vid, ada_attn_gate_txt;  // [L,C]
    std::vector<float> attention_residual_vid, attention_residual_txt; // [L,C]
    std::vector<float> mlp_norm_vid, mlp_norm_txt;            // [L,C]
    std::vector<float> ada_mlp_in_vid, ada_mlp_in_txt;        // [L,C]
    std::vector<float> swiglu_gate_vid, swiglu_up_vid;        // [L,MLP]
    std::vector<float> swiglu_product_vid, swiglu_output_vid; // [L,MLP]/[L,C]
    std::vector<float> swiglu_gate_txt, swiglu_up_txt;        // [L,MLP]
    std::vector<float> swiglu_product_txt, swiglu_output_txt; // [L,MLP]/[L,C]
    std::vector<float> ada_mlp_gate_vid, ada_mlp_gate_txt;    // [L,C]
    std::vector<float> final_vid, final_txt;                  // [L,C]
};

// Real SeedVR2 AdaSingle math (models/dit_v2/modulation.py):
//   in : hid * (scaleA + scaleB) + (shiftA + shiftB)
//   out: hid * (gateA + gateB)
// The caller extracts the per-layer [B, C] shift/scale/gate slices from the
// [B, 6*C] embedding (interleaved per-dim: per dim d the six values are
// [attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate], layer
// idx = attn 0 / mlp 1).  emb_shift/emb_scale/emb_gate are [B, C] flat
// row-major, repeated per token of each sample (token_counts).  Learned params
// are the [C] AdaSingle parameters.  All buffers are [tokens*C] flat row-major.
bool ada_single_in(const std::vector<float>& hid, int dim,
                   const std::vector<float>& emb_shift, const std::vector<float>& emb_scale,
                   const std::vector<int>& token_counts,
                   const std::vector<float>& shift_b, const std::vector<float>& scale_b,
                   std::vector<float>& out, std::string& error);
bool ada_single_gate(const std::vector<float>& hid, int dim,
                     const std::vector<float>& emb_gate,
                     const std::vector<int>& token_counts,
                     const std::vector<float>& gate_b,
                     std::vector<float>& out, std::string& error);

// Real 720p window partition (models/dit_v2/window.py) over [B,3] video shapes
// with num_windows = (window_t, window_h, window_w) and T-major/H-mid/W-fastest
// token layout.
bool compute_windows(const std::vector<VideoShape>& shapes,
                     int window_t, int window_h, int window_w,
                     WindowInfo& out, std::string& error);

// Real MM-RoPE3D cos/sin tables (na/rope contract, rotary-embedding-torch 0.5.3
// freqs_for="lang", theta=10000, per-axis dims = effective_rotary_dim/3 with
// local pairs k=0..(axis_pairs-1): inv_freq = theta^(-k/axis_pairs)).  vid
// tokens are window-packed T-major/H-mid/W-fastest at positions (l+t,h,w); txt
// tokens are window-repeated at positions j (local index).  Pair 63
// (head dims 126..127) is identity.  Outputs are [rows, head_pairs].
bool generate_vid_rope_cos_sin(const std::vector<VideoShape>& shapes,
                               const std::vector<int>& txt_lengths,
                               const WindowInfo& windows,
                               const attention_dispatcher::DiTBlockConfig& block,
                               std::vector<float>& cos, std::vector<float>& sin,
                               std::string& error);
bool generate_txt_rope_cos_sin(const std::vector<int>& txt_lengths,
                               const WindowInfo& windows,
                               const attention_dispatcher::DiTBlockConfig& block,
                               std::vector<float>& cos, std::vector<float>& sin,
                               std::string& error);

struct SwigluStageOutput
{
    std::vector<float> gate, up, product, output;
};

// Staged hybrid runtime for a single real SeedVR2 NaMMSRTransformerBlock at a
// scaled config.  Host executes modulation/window/packing/RoPE/residual; ncnn
// executes norm/QKV/head/rope/projection/SwiGLU stage artifacts through the
// shared model cache.
class HybridDiTBlockRuntime
{
public:
    explicit HybridDiTBlockRuntime(attention_dispatcher::NcnnModelCache& cache);

    // Loads every stage model through the shared cache and reads AdaSingle
    // params from the artifact params_dir.
    bool initialize(const attention_dispatcher::CoreInfo& core, std::string& error);

    // One full block forward.  trace may be null to skip stage capture.
    bool run(const DiTBlockInput& input, DiTBlockOutput& output,
             DiTBlockTrace* trace, DiTBlockRunStats& stats);

    // SwiGLU stage helper used by the block and by the contract tests.
    bool swiglu_forward(const std::string& side, const ncnn::Mat& input,
                        SwigluStageOutput& out, std::string& error);

    const attention_dispatcher::DiTBlockConfig& block() const { return block_; }
    const attention_dispatcher::CoreInfo* core() const { return core_; }
    bool initialized() const { return initialized_; }

private:
    ncnn::Net* model(const std::string& name) const;

    bool run_linear(const std::string& name, const std::vector<float>& input, int in_dim,
                    std::vector<float>& output, int& out_dim, std::string& error);
    bool run_rms(const std::string& name, const std::vector<float>& input, int dim,
                 std::vector<float>& output, std::string& error);
    bool run_rope_chunked(const std::vector<float>& q, int rows, int head_dim,
                          const std::vector<float>& cos, const std::vector<float>& sin,
                          std::vector<float>& out, std::string& error);
    bool run_head_segment(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, int seg_len, int head_dim,
                          std::vector<float>& out, std::string& error);

    bool apply_ada(const std::string& side, const std::string& layer, const std::string& mode,
                   const std::vector<float>& hid, int tokens,
                   const ncnn::Mat& emb, const std::vector<int>& token_counts,
                   std::vector<float>& out, std::string& error);

    attention_dispatcher::NcnnModelCache& cache_;
    const attention_dispatcher::CoreInfo* core_ = nullptr;
    attention_dispatcher::DiTBlockConfig block_;
    bool initialized_ = false;
    std::map<std::string, ncnn::Net*> models_;
    // learned AdaSingle params indexed [side][layer][p] p=0 shift,1 scale,2 gate
    std::map<std::string, std::map<std::string, std::array<std::vector<float>, 3>>> params_;
};
}
