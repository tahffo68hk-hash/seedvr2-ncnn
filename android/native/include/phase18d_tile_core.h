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

bool plan_tiles(int image_width, int image_height, const TileConfig& config,
                TilePlan* plan, std::string* error);

bool extract_and_pad_rgb8(const uint8_t* image_rgb, size_t byte_count,
                           int image_width, int image_height, const TileSpec& tile,
                           PaddedTile* padded, std::string* error);

}  // namespace seededge::phase18d
