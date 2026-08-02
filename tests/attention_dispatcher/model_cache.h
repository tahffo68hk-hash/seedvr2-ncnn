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
    ncnn::Net* get_or_load(const CoreInfo& core, double& load_ms, std::string& error_message);
    ncnn::Net* get_or_load(const NcnnModelFiles& files, CacheLoadStats& stats, std::string& error_message);
    std::size_t size() const noexcept;
    void clear();
private:
    std::unordered_map<std::string, std::unique_ptr<ncnn::Net>> models_;
};
}
