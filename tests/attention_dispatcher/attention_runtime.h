#pragma once
#include <string>
#include <vector>
#include "attention_dispatcher.h"
#include "model_cache.h"
namespace attention_dispatcher {
struct RuntimeResult {
    bool success = false; std::string selected_core; Metadata metadata{}; std::vector<float> output; std::string error_message;
    double model_load_ms = 0.0, metadata_ms = 0.0, registry_lookup_ms = 0.0, partition_ms = 0.0, mask_generation_ms = 0.0, ncnn_inference_ms = 0.0, reverse_crop_ms = 0.0, total_runtime_ms = 0.0;
};
RuntimeResult run_attention(const AttentionRequest& request, const std::vector<float>& input, const ArtifactRegistry& registry, NcnnModelCache& model_cache);
}
