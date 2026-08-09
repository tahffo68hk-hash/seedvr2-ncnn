#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace seededge::phase18d {

struct TileConfig
{
    int tile_size = 128;
    int overlap = 32;
    int stride = 96;
};

struct TileSpec
{
    int tile_index = -1;
    int grid_x = -1;
    int grid_y = -1;
    int source_left = 0;
    int source_top = 0;
    int source_right = 0;
    int source_bottom = 0;
    int valid_width = 0;
    int valid_height = 0;
    int padding_left = 0;
    int padding_top = 0;
    int padding_right = 0;
    int padding_bottom = 0;
};

struct TilePlan
{
    int image_width = 0;
    int image_height = 0;
    TileConfig config;
    std::vector<TileSpec> tiles;
};

struct PaddedTile
{
    std::vector<uint8_t> rgb;
    TileSpec spec;
    std::string padding_mode_x;
    std::string padding_mode_y;
};

struct FusionResult
{
    std::vector<float> rgb;
    std::vector<float> weight_sum;
    size_t uncovered_pixel_count = 0;
    size_t zero_weight_pixel_count = 0;
    float max_abs_error = 0.f;
};

struct PhotoRuntimeLimits
{
    // 16 megapixels covers the target 4032x3024 photos without claiming
    // unbounded allocations.  Callers may lower either limit per device.
    size_t max_pixels = 16u * 1024u * 1024u;
    size_t max_tiles = 2048u;
};

struct PhotoStorageEstimate
{
    size_t input_rgb_bytes = 0;
    size_t output_rgb_bytes = 0;
    size_t tile_rgb_bytes = 0;
    size_t checkpoint_bytes = 0;
    size_t fusion_bytes = 0;
    size_t metadata_bytes = 0;
    size_t atomic_write_reserve_bytes = 0;
    size_t required_bytes = 0;
};

struct CheckpointIdentity
{
    std::string task_id;
    std::string normalized_image_sha256;
    std::string tile_input_sha256;
    std::string model_manifest_sha256;
    std::string dtype;
    std::string checkpoint_sha256;
    TileSpec tile;
    std::vector<int> shape;
    size_t byte_count = 0;
    bool finite = false;
};

bool plan_tiles(int image_width, int image_height, const TileConfig& config,
                TilePlan* plan, std::string* error);

bool validate_photo_limits(int image_width, int image_height,
                           const TilePlan& plan,
                           const PhotoRuntimeLimits& limits,
                           std::string* error);

bool estimate_photo_storage(int image_width, int image_height,
                            size_t tile_count,
                            PhotoStorageEstimate* estimate,
                            std::string* error);

bool has_required_free_space(size_t required_bytes, size_t available_bytes,
                             std::string* error);

std::string tile_plan_sha256(const TilePlan& plan);
std::string tile_plan_json(const TilePlan& plan);

bool extract_and_pad_rgb8(const uint8_t* image_rgb, size_t byte_count,
                           int image_width, int image_height, const TileSpec& tile,
                           PaddedTile* padded, std::string* error);

std::string sha256_hex(const uint8_t* data, size_t byte_count);

bool identity_fuse_rgb8(const TilePlan& plan, const uint8_t* image_rgb,
                        size_t byte_count, FusionResult* result, std::string* error);

// Builds one-dimensional fusion weights for a valid tile extent.
// Leading overlap uses a pixel-centered sin^2 ramp.
// Trailing overlap uses a pixel-centered cos^2 ramp.
bool make_axis_fusion_weights(
    int extent,
    int leading_overlap,
    int trailing_overlap,
    std::vector<float>* weights,
    std::string* error);

// Fuses valid RGB8 tile outputs into a full-resolution float RGB image.
// tile_outputs[i] corresponds to plan.tiles[i] and contains exactly
// valid_width * valid_height * 3 bytes without padded pixels.
bool weighted_fuse_rgb8(
    const TilePlan& plan,
    const std::vector<std::vector<uint8_t>>& tile_outputs,
    FusionResult* result,
    std::string* error);

bool validate_checkpoint_identity(const CheckpointIdentity& expected,
                                  const CheckpointIdentity& actual,
                                  std::string* error_code, std::string* error);

bool is_valid_tile_state_transition(const std::string& from, const std::string& to);

}  // namespace seededge::phase18d
