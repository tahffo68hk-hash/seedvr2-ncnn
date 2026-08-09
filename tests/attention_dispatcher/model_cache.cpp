#include "model_cache.h"
#include <chrono>
#include <filesystem>
namespace attention_dispatcher {
namespace { std::string key_for(const SingleModelArtifact& c) { return std::filesystem::absolute(c.model.param_path).lexically_normal().string() + "|" + std::filesystem::absolute(c.model.bin_path).lexically_normal().string(); } }
ncnn::Net* NcnnModelCache::get_or_load(const NcnnModelFiles& files, CacheLoadStats& stats, std::string& error_message) {
    stats={}; stats.cache_size_before=models_.size(); const std::string key=std::filesystem::absolute(files.param_path).lexically_normal().string()+"|"+std::filesystem::absolute(files.bin_path).lexically_normal().string();
    auto found=models_.find(key); if(found!=models_.end()){stats.cache_hit=true;stats.cache_size_after=models_.size();error_message.clear();return found->second.get();}
    stats.cache_miss=true; const auto begin=std::chrono::steady_clock::now(); auto net=std::make_unique<ncnn::Net>();net->opt.use_vulkan_compute=false;net->opt.num_threads=num_threads_;net->opt.use_packing_layout=false;
    if(net->load_param(files.param_path.string().c_str())!=0||net->load_model(files.bin_path.string().c_str())!=0){stats.cache_size_after=models_.size();error_message="cache load failed stage=model name="+files.name+" param="+files.param_path.string()+" bin="+files.bin_path.string();return nullptr;}
    stats.load_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();auto*result=net.get();models_.emplace(key,std::move(net));stats.cache_size_after=models_.size();error_message.clear();
    std::error_code ec; const auto sz=std::filesystem::file_size(files.bin_path, ec); bytes_by_key_[key]=ec?0:static_cast<std::size_t>(sz);
    return result;
}
ncnn::Net* NcnnModelCache::get_or_load(const CoreInfo& core, double& load_ms, std::string& error_message) {
    const SingleModelArtifact* single = core.single_model(); if (!single) { load_ms = 0.0; error_message = "NcnnModelCache single-model loader rejected core=" + core.name + " variant=" + core.key.variant + " execution=" + (core.execution_type == ExecutionType::StagedHostHeadMerge ? "staged_host_head_merge" : "single_model"); return nullptr; }
    CacheLoadStats stats; ncnn::Net* result=get_or_load(single->model,stats,error_message);load_ms=stats.load_ms;return result;
}
std::size_t NcnnModelCache::size() const noexcept { return models_.size(); }
void NcnnModelCache::clear() { models_.clear(); bytes_by_key_.clear(); }
std::size_t NcnnModelCache::erase_by_prefix(const std::string& abs_prefix) noexcept {
    std::size_t erased = 0;
    for (auto it = models_.begin(); it != models_.end();) {
        if (it->first.compare(0, abs_prefix.size(), abs_prefix) == 0) {
            bytes_by_key_.erase(it->first);
            it = models_.erase(it);
            ++erased;
        } else {
            ++it;
        }
    }
    return erased;
}
std::size_t NcnnModelCache::resident_bytes_estimate() const noexcept {
    std::size_t total = 0;
    for (const auto& kv : bytes_by_key_) total += kv.second;
    return total;
}
}
