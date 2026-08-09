#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <net.h>
#include "artifact_registry.h"
namespace attention_dispatcher {
struct CacheLoadStats { bool cache_hit=false; bool cache_miss=false; double load_ms=0.0; std::size_t cache_size_before=0,cache_size_after=0; };
class NcnnModelCache {
public:
    explicit NcnnModelCache(int num_threads = 1) noexcept
        : num_threads_(num_threads > 0 ? num_threads : 1) {}
    ncnn::Net* get_or_load(const CoreInfo& core, double& load_ms, std::string& error_message);
    ncnn::Net* get_or_load(const NcnnModelFiles& files, CacheLoadStats& stats, std::string& error_message);
    std::size_t size() const noexcept;
    void clear();

    // Phase 15 streaming manager support: block-scoped release and accounting.
    // The cache key is "<abs param path>|<abs bin path>"; a block's artifact
    // directory is unique, so erase_by_prefix/release_block use the absolute
    // artifact-directory prefix of the variant.
    std::size_t erase_by_prefix(const std::string& abs_prefix) noexcept;
    std::size_t release_block(const std::string& abs_variant_dir) noexcept { return erase_by_prefix(abs_variant_dir); }
    std::size_t resident_bytes_estimate() const noexcept;
    std::size_t load_count() const noexcept { return models_.size(); }
private:
    int num_threads_ = 1;
    std::unordered_map<std::string, std::unique_ptr<ncnn::Net>> models_;
    std::unordered_map<std::string, std::size_t> bytes_by_key_;
};
}
