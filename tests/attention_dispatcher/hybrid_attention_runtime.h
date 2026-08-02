#pragma once
#include <array>
#include <chrono>
#include "attention_runtime.h"
#include "hybrid_model_bundle.h"
namespace attention_dispatcher {
struct HybridRuntimeTimings { double total_ms=0, pack_ms=0, contract_read_ms=0, merge_ms=0, projection_ms=0, unpack_ms=0; std::array<double,4> head_ms{}; };
// Test hooks intentionally remain data-only: production callers leave all overrides null.
struct HybridRuntimeOptions { const std::vector<float>* mask_override=nullptr; const std::vector<float>* rope_cos_override=nullptr; const std::vector<float>* rope_sin_override=nullptr; };
struct HybridRuntimeResult {bool success=false;std::string selected_artifact,variant,error_message;Metadata metadata{};std::array<std::vector<float>,4> head_outputs;std::vector<float> merged_output,projection_output,windows_output,bthwc_output;HybridBundleLoadStats cache_stats{};HybridRuntimeTimings timings{};};
HybridRuntimeResult run_hybrid_attention(const AttentionRequest&,const std::vector<float>&,const ArtifactRegistry&,NcnnModelCache&,const HybridRuntimeOptions& = {});
}
