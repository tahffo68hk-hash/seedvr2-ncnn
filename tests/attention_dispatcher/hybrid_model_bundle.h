#pragma once
#include <array>
#include "model_cache.h"
namespace attention_dispatcher {
struct HybridModelBundle { std::array<ncnn::Net*,4> heads{}; ncnn::Net* output_projection=nullptr; bool valid()const noexcept {return output_projection&&heads[0]&&heads[1]&&heads[2]&&heads[3];} };
struct HybridBundleLoadStats { std::array<CacheLoadStats,4> heads{}; CacheLoadStats projection{}; double load_ms=0.; };
bool acquire_hybrid_bundle(const CoreInfo&,NcnnModelCache&,HybridModelBundle&,HybridBundleLoadStats&,std::string&);
}
