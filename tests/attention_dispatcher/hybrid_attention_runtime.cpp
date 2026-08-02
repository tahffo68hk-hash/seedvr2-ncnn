#include "hybrid_attention_runtime.h"
#include "hybrid_window_adapter.h"
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
    const size_t mask_count=size_t(r.metadata.num_windows)*64*64, rope_count=256; std::vector<float> mask,cos,sin; timer=Clock::now();
    const auto ref=core->directory/"reference";
    if (!validate_override("mask",options.mask_override,mask_count,e) || !validate_override("rope_cos",options.rope_cos_override,rope_count,e) || !validate_override("rope_sin",options.rope_sin_override,rope_count,e)) { r.error_message="contract: "+e; return r; }
    if (options.mask_override) mask=*options.mask_override; else if(!read_exact(ref/"mask.bin",mask,mask_count,e)) { r.error_message="contract: "+e; return r; }
    if (options.rope_cos_override) cos=*options.rope_cos_override; else if(!read_exact(ref/"rope_cos.bin",cos,rope_count,e)) { r.error_message="contract: "+e; return r; }
    if (options.rope_sin_override) sin=*options.rope_sin_override; else if(!read_exact(ref/"rope_sin.bin",sin,rope_count,e)) { r.error_message="contract: "+e; return r; }
    r.timings.contract_read_ms=elapsed_ms(timer);
    for(int h=0;h<4;++h) { timer=Clock::now(); r.head_outputs[h].resize(size_t(r.metadata.num_windows)*64*8); for(int w=0;w<r.metadata.num_windows;++w) { ncnn::Mat x(32,64),m(64,64),co(4,64),si(4,64),out; memcpy(x.data,windows.data()+w*2048,2048*sizeof(float)); memcpy(m.data,mask.data()+w*4096,4096*sizeof(float)); memcpy(co.data,cos.data(),1024); memcpy(si.data,sin.data(),1024); auto ex=bundle.heads[h]->create_extractor(); if(ex.input("in0",x)||ex.input("in1",m)||ex.input("in2",co)||ex.input("in3",si)||ex.extract("out0",out)||out.total()!=512) { r.error_message="head inference="+std::to_string(h); return r; } memcpy(r.head_outputs[h].data()+w*512,out.data,512*sizeof(float)); } r.timings.head_ms[h]=elapsed_ms(timer); }
    timer=Clock::now(); std::vector<std::vector<float>> hs(4); for(int i=0;i<4;++i) hs[i]=r.head_outputs[i]; if(!seedvr2::merge_attention_heads(hs,r.metadata.num_windows,64,8,r.merged_output,e)) { r.error_message=e; return r; } r.timings.merge_ms=elapsed_ms(timer);
    timer=Clock::now(); ncnn::Mat pi(32,r.metadata.num_windows*64),po; memcpy(pi.data,r.merged_output.data(),r.merged_output.size()*sizeof(float)); auto ex=bundle.output_projection->create_extractor(); if(ex.input("in0",pi)||ex.extract("out0",po)||po.total()!=r.merged_output.size()) { r.error_message="projection inference"; return r; } r.projection_output.assign((float*)po.data,(float*)po.data+po.total()); r.windows_output=r.projection_output; r.timings.projection_ms=elapsed_ms(timer);
    timer=Clock::now(); if(!unpack_hybrid_windows(r.windows_output,c,r.bthwc_output,e)) { r.error_message=e; return r; } r.timings.unpack_ms=elapsed_ms(timer); r.timings.total_ms=elapsed_ms(total_start); r.success=true; return r;
}
}
