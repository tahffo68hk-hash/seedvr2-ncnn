#include "hybrid_attention_runtime.h"
#include "hybrid_window_adapter.h"
#include "dynamic_padding_mask.h"
#include "dynamic_rope3d.h"
#include "ncnn_mat_contract.h"
#include "../seedvr2_attention_fidelity/host_head_merge.h"
#include <chrono>
#include <cstring>
#include <fstream>
namespace attention_dispatcher {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
bool read_exact(const std::filesystem::path& path, std::vector<float>& values, size_t expected, std::string& error)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { error = "contract file missing: " + path.string(); return false; }
    const auto bytes = file.tellg();
    const auto expected_bytes = static_cast<std::streamoff>(expected * sizeof(float));
    if (bytes != expected_bytes) { error = "contract " + path.filename().string() + " expected_bytes=" + std::to_string(expected_bytes) + " actual_bytes=" + std::to_string(bytes); return false; }
    file.seekg(0); values.resize(expected); file.read(reinterpret_cast<char*>(values.data()), expected_bytes);
    if (!file) { error = "contract file read failed: " + path.string(); return false; }
    return true;
}
bool validate_override(const char* name, const std::vector<float>* values, size_t expected, std::string& error)
{
    if (!values) return true;
    if (values->size() != expected) { error = std::string(name) + " expected_floats=" + std::to_string(expected) + " actual_floats=" + std::to_string(values->size()); return false; }
    return true;
}
}
HybridRuntimeResult run_hybrid_attention(const AttentionRequest& q, const std::vector<float>& input, const ArtifactRegistry& registry, NcnnModelCache& cache, const HybridRuntimeOptions& options)
{
    const auto total_start = Clock::now(); HybridRuntimeResult r; r.variant=q.variant; std::string e;
    if (!validate_attention_request(q,e)) { r.error_message="validate: "+e; return r; }
    try { r.metadata=dispatch_shape({q.batch,q.time,q.height,q.width,q.channels}); } catch(const std::exception& x) { r.error_message=x.what(); return r; }
    CoreKey key=make_core_key(r.metadata); key.variant=q.variant; const CoreInfo* core=registry.lookup(key);
    if(!core) { r.error_message="lookup: variant="+q.variant; return r; }
    r.selected_artifact=core->name; if(!core->staged_head_merge()) { r.error_message="executor requires staged_host_head_merge"; return r; }
    if(input.size()!=size_t(q.batch)*q.time*q.height*q.width*q.channels) { r.error_message="input size mismatch"; return r; }
    HybridModelBundle bundle; if(!acquire_hybrid_bundle(*core,cache,bundle,r.cache_stats,e)) { r.error_message=e; return r; }
    HybridWindowContract c{q.batch,q.time,q.height,q.width,q.channels,r.metadata.window_t,r.metadata.window_h,r.metadata.window_w,r.metadata.grid_t,r.metadata.grid_h,r.metadata.grid_w,r.metadata.pad_t,r.metadata.pad_h,r.metadata.pad_w};
    std::vector<float> windows; auto timer=Clock::now(); if(!pack_hybrid_windows(input,c,windows,e)) { r.error_message=e; return r; } r.timings.pack_ms=elapsed_ms(timer);
    const size_t mask_count=size_t(r.metadata.num_windows)*r.metadata.tokens_per_window*r.metadata.tokens_per_window; std::vector<float> mask,cos,sin; timer=Clock::now();
    const auto* aux=core->auxiliary(); if(!aux) { r.error_message="contract: staged artifact missing auxiliary_inputs"; return r; }
    DynamicAttentionContract dc{q.batch,q.time,q.height,q.width,q.channels,r.metadata.window_t,r.metadata.window_h,r.metadata.window_w,r.metadata.grid_t,r.metadata.grid_h,r.metadata.grid_w,r.metadata.pad_t,r.metadata.pad_h,r.metadata.pad_w,r.metadata.num_windows,r.metadata.tokens_per_window,4,8,0};
    if (aux->rope_mode==RopeMode::ArtifactReference) dc.rope_dim=8; else dc.rope_dim=std::get<DynamicMMRoPE3DConfig>(aux->rope).rope_dim;
    const size_t rope_count=size_t(r.metadata.tokens_per_window)*(dc.head_dim/2);
    if (!validate_override("mask",options.mask_override,mask_count,e) || !validate_override("rope_cos",options.rope_cos_override,rope_count,e) || !validate_override("rope_sin",options.rope_sin_override,rope_count,e)) { r.error_message="contract: "+e; return r; }
    if (options.mask_override) mask=*options.mask_override;
    else if(aux->mask_mode==MaskMode::ArtifactReference) { const auto* cfg=std::get_if<ArtifactReferenceMaskConfig>(&aux->mask); if(!cfg||!read_exact(cfg->path,mask,mask_count,e)) { r.error_message="contract: "+e; return r; } }
    else { std::vector<unsigned char> validity; if(!generate_padding_validity(dc,validity,e)||!generate_additive_key_mask(validity,dc.num_windows,dc.tokens_per_window,mask,e)){r.error_message="contract: "+e;return r;} r.validity.assign(validity.begin(), validity.end()); }
    if (options.rope_cos_override) cos=*options.rope_cos_override; else if(aux->rope_mode==RopeMode::ArtifactReference) { const auto* cfg=std::get_if<ArtifactReferenceRopeConfig>(&aux->rope); if(!cfg||!read_exact(cfg->cos_path,cos,rope_count,e)){r.error_message="contract: "+e;return r;} } else { Rope3DOutput rope; const auto& cfg=std::get<DynamicMMRoPE3DConfig>(aux->rope); Rope3DConfig rc{cfg.theta,cfg.axis_dims,cfg.axis_scales}; if(!generate_dynamic_rope3d(dc,rc,rope,e)){r.error_message="contract: "+e;return r;} cos=std::move(rope.cos); sin=std::move(rope.sin); }
    if (options.rope_sin_override) sin=*options.rope_sin_override; else if(aux->rope_mode==RopeMode::ArtifactReference) { const auto* cfg=std::get_if<ArtifactReferenceRopeConfig>(&aux->rope); if(!cfg||!read_exact(cfg->sin_path,sin,rope_count,e)){r.error_message="contract: "+e;return r;} }
    r.timings.contract_read_ms=elapsed_ms(timer);
    r.mask_input=mask; r.rope_cos=cos; r.rope_sin=sin;
    ncnn::Mat mask_mat, cos_mat, sin_mat;
    if (!make_mask_mat(mask, r.metadata.num_windows, r.metadata.tokens_per_window, mask_mat, e) || !make_rope_mat(cos, r.metadata.tokens_per_window, dc.head_dim / 2, cos_mat, e) || !make_rope_mat(sin, r.metadata.tokens_per_window, dc.head_dim / 2, sin_mat, e)) { r.error_message="contract: "+e; return r; }
    for(int h=0;h<4;++h) { timer=Clock::now(); r.head_outputs[h].resize(size_t(r.metadata.num_windows)*64*8); for(int w=0;w<r.metadata.num_windows;++w) { ncnn::Mat x(32,64),m=mask_mat.channel(w),co=cos_mat,si=sin_mat,out; memcpy(x.data,windows.data()+w*2048,2048*sizeof(float)); auto ex=bundle.heads[h]->create_extractor(); if(ex.input("in0",x)||ex.input("in1",m)||ex.input("in2",co)||ex.input("in3",si)||ex.extract("out0",out)||out.total()!=512) { r.error_message="head inference="+std::to_string(h); return r; } memcpy(r.head_outputs[h].data()+w*512,out.data,512*sizeof(float)); } r.timings.head_ms[h]=elapsed_ms(timer); }
    timer=Clock::now(); std::vector<std::vector<float>> hs(4); for(int i=0;i<4;++i) hs[i]=r.head_outputs[i]; if(!seedvr2::merge_attention_heads(hs,r.metadata.num_windows,64,8,r.merged_output,e)) { r.error_message=e; return r; } r.timings.merge_ms=elapsed_ms(timer);
    timer=Clock::now(); ncnn::Mat pi(32,r.metadata.num_windows*64),po; memcpy(pi.data,r.merged_output.data(),r.merged_output.size()*sizeof(float)); auto ex=bundle.output_projection->create_extractor(); if(ex.input("in0",pi)||ex.extract("out0",po)||po.total()!=r.merged_output.size()) { r.error_message="projection inference"; return r; } r.projection_output.assign((float*)po.data,(float*)po.data+po.total()); r.windows_output=r.projection_output; r.timings.projection_ms=elapsed_ms(timer);
    timer=Clock::now(); if(!unpack_hybrid_windows(r.windows_output,c,r.bthwc_output,e)) { r.error_message=e; return r; } r.timings.unpack_ms=elapsed_ms(timer); r.timings.total_ms=elapsed_ms(total_start); r.success=true; return r;
}
}
