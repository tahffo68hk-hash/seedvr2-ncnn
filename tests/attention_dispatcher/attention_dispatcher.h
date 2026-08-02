#pragma once

#include <string>
#include <vector>

#include "metadata.h"
#include "artifact_registry.h"

namespace attention_dispatcher
{
struct CoreEntry
{
    std::string name;
    bool available;
};

struct AttentionRequest
{
    int batch = 0;
    int time = 0;
    int height = 0;
    int width = 0;
    int channels = 0;
    std::string dtype;
    std::string variant = "baseline";
};

class CoreRegistry
{
public:
    explicit CoreRegistry(std::vector<CoreEntry> entries);
    const CoreEntry& find(const Metadata& metadata) const;

private:
    std::vector<CoreEntry> entries_;
};

Metadata dispatch_shape(Shape shape);
bool validate_attention_request(const AttentionRequest& request, std::string& error_message);
CoreKey make_core_key(const Metadata& metadata) noexcept;
bool dispatch_request(const AttentionRequest& request, const ArtifactRegistry& registry,
                      Metadata& metadata, const CoreInfo*& core, std::string& error_message);
CoreRegistry make_default_registry();
CoreEntry select_default_core(const Metadata& metadata);
std::vector<float> make_additive_key_mask(const Metadata& metadata);
std::vector<float> partition_bthwc(const std::vector<float>& input, const Metadata& metadata);
std::vector<float> reverse_crop_bthwc(const std::vector<float>& windows, const Metadata& metadata);
}
