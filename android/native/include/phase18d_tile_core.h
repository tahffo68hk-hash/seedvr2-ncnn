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

bool extract_and_pad_rgb8(const uint8_t* image_rgb, size_t byte_count,
                           int image_width, int image_height, const TileSpec& tile,
                           PaddedTile* padded, std::string* error);

std::string sha256_hex(const uint8_t* data, size_t byte_count);

bool identity_fuse_rgb8(const TilePlan& plan, const uint8_t* image_rgb,
                        size_t byte_count, FusionResult* result, std::string* error);

bool validate_checkpoint_identity(const CheckpointIdentity& expected,
                                  const CheckpointIdentity& actual,
                                  std::string* error_code, std::string* error);

bool is_valid_tile_state_transition(const std::string& from, const std::string& to);

}  // namespace seededge::phase18d
